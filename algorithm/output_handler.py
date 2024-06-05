from typing import List, Tuple
from exception import UnsatException


class OutputHandler:
    def __init__(self, append_outputs):
        self._append_outputs = append_outputs

    def handle(self, outputs: List[Tuple[int, int]]) -> List[int]:
        new_outputs = []
        results = []
        for res, negation_mark in outputs:
            if res + negation_mark > 1:
                new_outputs.append(res + negation_mark)
            else:
                results.append(res + negation_mark)
        if self._append_outputs:
            all_zeroes_in_out = all([r == 0 for r in results])
            if len(new_outputs) == 0 and all_zeroes_in_out:
                raise UnsatException
            if not all_zeroes_in_out:
                self._append_outputs = False
        return new_outputs

    @property
    def append_outputs(self):
        return self._append_outputs


__all__ = [
    'OutputHandler'
]
