from util import Status
from typing import Optional


class CncReport:
    def __init__(self, status: Status, satisfying_assignment: str = None):
        self._status = status
        self._satisfying_assignment = satisfying_assignment

    @property
    def satisfying_assignment(self) -> Optional[str]:
        return self._satisfying_assignment

    @property
    def status(self) -> Status:
        return self._status


__all__ = [
    'CncReport'
]
