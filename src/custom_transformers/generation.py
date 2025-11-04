import re

import torch
from transformers.generation import (
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
)


class HydraCompatibleLogitsProcessorList(LogitsProcessorList):
    """Hydra-compatible version of LogitsProcessorList.

    Initialized using dict[str, LogitsProcessor], which in turn initializes
    the parent object as: LogitsProcessorList(list(dict.values())).
    """

    def __init__(self, logits_processor_dict: dict[str, LogitsProcessor]):
        super().__init__(list(logits_processor_dict.values()))


class HydraCompatibleStoppingCriteriaList(StoppingCriteriaList):
    """Hydra-compatible version of StoppingCriteriaList.

    Initialized using dict[str, StoppingCriteria], which in turn initializes
    the parent object as: StoppingCriteriaList(list(dict.values())).
    """

    def __init__(self, stopping_criteria_dict: dict[str, StoppingCriteria]):
        super().__init__(list(stopping_criteria_dict.values()))


class RegexStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, pattern):
        self.tokenizer = tokenizer
        self.pattern = pattern
        assert (self.tokenizer.eos_token_id == self.tokenizer.bos_token_id) or self.tokenizer.bos_token_id is None, "Assumes EOS and BOS are the same token"

    def __call__(
        self, input_ids: torch.LongTensor, scores: None | torch.FloatTensor, **kwargs
    ) -> torch.BoolTensor:
        if input_ids.numel() == 0:
            return torch.tensor([False], device=input_ids.device, dtype=torch.bool)
        matches = []
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        for i in range(input_ids.shape[0]):
            prompt_offset = torch.where(input_ids[i] == self.tokenizer.eos_token_id)[0][1] + 1
            text = self.tokenizer.decode(input_ids[i][prompt_offset:])
            matches.append(len(re.findall(self.pattern, text)) > 0)
        return torch.tensor(matches, device=input_ids.device, dtype=torch.bool)
