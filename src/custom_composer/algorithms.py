import logging
from typing import Any

from composer import Event, Logger, State
from composer.core.algorithm import Algorithm
from composer.utils import reproducibility

log = logging.getLogger(__name__)


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
