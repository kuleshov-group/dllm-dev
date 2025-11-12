import logging
from typing import Any, List, Literal, Union

from composer import Event, Logger, State, Time, TimeUnit
from composer.core.algorithm import Algorithm
from composer.utils import reproducibility

log = logging.getLogger(__name__)


def _is_eval_event(event: Event, state: State) -> bool:
    evaluators_executing = []
    for evaluator in state.evaluators:
        evaluators_executing.append(evaluator.eval_interval(state, event))
    return any(evaluators_executing)


class SetFixedRngStateForEval(Algorithm):
    def __init__(
        self,
        seed: int = 1234,
    ):
        super().__init__()
        self.seed = seed
        self.fixed_rng_state_active = False
        self._og_rng_state = None

        self.set_fixed_rng_state_events = [
            Event.EVAL_START,
            Event.EVAL_STANDALONE_START,
        ]
        self.restore_rng_state_events = [
            Event.EVAL_END,
            Event.EVAL_STANDALONE_END,
        ]

    def _set_rng_state(self):
        if self.fixed_rng_state_active:
            raise ValueError("Fixed RNG state is already active.")
        # Store current rng state
        self._og_rng_state = reproducibility.get_rng_state()
        reproducibility.seed_all(self.seed)
        self.fixed_rng_state_active = True
        log.info(f"Fixed RNG state set with seed {self.seed}")

    def _restore_rng_state(self):
        if not self.fixed_rng_state_active:
            raise ValueError("Fixed RNG state is not currently active.")
        reproducibility.load_rng_state(self._og_rng_state)
        self.fixed_rng_state_active = False
        log.info("Original RNG state restored.")

    def match(self, event: Event, state: State) -> bool:
        if (
            event in self.set_fixed_rng_state_events
            or event in self.restore_rng_state_events
        ):
            return True
        return False

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        if event in self.set_fixed_rng_state_events:
            self._set_rng_state()
        if event in self.restore_rng_state_events:
            self._restore_rng_state()

    def state_dict(self) -> dict[str, Any]:
        state_dict = super().state_dict()
        state_dict["fixed_rng_state_active"] = self.fixed_rng_state_active
        state_dict["og_rng_state"] = self._og_rng_state
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.fixed_rng_state_active = state_dict["fixed_rng_state_active"]
        self._og_rng_state = state_dict["og_rng_state"]


class BlockSizeAnnealing(Algorithm):
    def __init__(
        self,
        max_block_size: int,
        schedule: Union[List[str], str],
        factor: int = 2,
        increase_via_add_or_multiply: Literal["add", "multiply"] = "multiply",
    ):
        super().__init__()
        self.max_block_size = max_block_size
        if isinstance(schedule, list):
            self.schedule = [Time.from_timestring(s) for s in schedule]
            for s in self.schedule:
                assert s.unit == TimeUnit.BATCH or s.unit == TimeUnit.EPOCH, (
                    "Only batch or epoch events are supported."
                )
        else:
            self.schedule = Time.from_timestring(schedule)
            assert (
                self.schedule.unit == TimeUnit.BATCH
                or self.schedule.unit == TimeUnit.EPOCH
            ), "Only batch or epoch intervals are supported."
        self.factor = factor
        self.increase_via_add_or_multiply = increase_via_add_or_multiply
        # -1: sentinel value to indicate schedule has not started
        self.block_size = -1
        self._increase_deferred_until_eval_end = False

    def _increase_block_size(self, current_block_size):
        if self.increase_via_add_or_multiply == "add":
            return current_block_size + self.factor
        return current_block_size * self.factor

    def match(self, event: Event, state: State) -> bool:
        if event == Event.AFTER_LOAD:
            return True

        # Execute the "deferred" `apply`
        if event == Event.EVAL_END and self._increase_deferred_until_eval_end:
            self._increase_deferred_until_eval_end = False  # reset
            return True

        # Currently, only batch or epoch are supported for scheduling increase
        if event not in [Event.BATCH_END, Event.EPOCH_END]:
            return False

        if isinstance(self.schedule, list):
            for s in self.schedule:
                current_time = state.timestamp.get(s.unit).value
                if current_time == s.value:
                    # If trainer will execute eval, wait to apply block increase until
                    # after eval loop end
                    if _is_eval_event(event, state):
                        self._increase_deferred_until_eval_end = True
                        return False
                    return True
            return False
        current_time = state.timestamp.get(self.schedule.unit).value
        if current_time > 0 and current_time % self.schedule.value == 0:
            # If trainer will execute eval, wait to apply block increase until after
            # eval loop end
            if _is_eval_event(event, state):
                self._increase_deferred_until_eval_end = True
                return False
            return True

        return False

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        if event == Event.AFTER_LOAD:
            if self.block_size > 0:
                if hasattr(state.model, "module"):
                    if self.block_size != state.model.module.model.config.block_size:
                        state.model.module.model.config.block_size = self.block_size
                        state.model.module.model.update_static_mask(
                            state.model.module.model.generate_static_mask()
                        )
                elif self.block_size != state.model.model.config.block_size:
                    state.model.model.config.block_size = self.block_size
                    state.model.model.update_static_mask(
                        state.model.model.generate_static_mask()
                    )
                log.info(f"Restored block size value to {self.block_size}.")
                # Update collators' block size
                if self.block_size != state.train_dataloader.collate_fn.block_size:
                    state.train_dataloader.collate_fn.update_block_size(self.block_size)
                for e in state.evaluators:
                    if self.block_size != e.dataloader.dataloader.collate_fn.block_size:
                        e.dataloader.dataloader.collate_fn.update_block_size(
                            self.block_size
                        )
            else:
                if hasattr(state.model, "module"):
                    self.block_size = state.model.module.model.config.block_size
                else:
                    self.block_size = state.model.model.config.block_size
            return

        # TODO: Will this work with FSDP?
        if hasattr(state.model, "module"):
            if not hasattr(state.model.module.model.config, "block_size"):
                raise ValueError(
                    "Model config does not contain `block_size` parameter."
                )
            if state.model.module.model.config.block_size >= self.max_block_size:
                return
            new_block_size = self._increase_block_size(
                state.model.module.model.config.block_size
            )
            state.model.module.model.config.block_size = new_block_size
            state.model.module.model.update_static_mask(
                state.model.module.model.generate_static_mask()
            )
        else:
            if not hasattr(state.model.model.config, "block_size"):
                raise ValueError(
                    "Model config does not contain `block_size` parameter."
                )
            if state.model.model.config.block_size >= self.max_block_size:
                return
            new_block_size = self._increase_block_size(
                state.model.model.config.block_size
            )
            state.model.model.config.block_size = new_block_size
            state.model.model.update_static_mask(
                state.model.model.generate_static_mask()
            )
        # Update collators' block size
        state.train_dataloader.collate_fn.update_block_size(new_block_size)
        for e in state.evaluators:
            e.dataloader.dataloader.collate_fn.update_block_size(new_block_size)

        log.info(f"Block size updated: {self.block_size} --> {new_block_size}")
        self.block_size = new_block_size
        self._increase_deferred_until_eval_end = False

    def state_dict(self) -> dict[str, Any]:
        state_dict = super().state_dict()
        state_dict["block_size"] = self.block_size
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.block_size = state_dict["block_size"]
