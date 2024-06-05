# Abstract error we encounter when our task was solved during the solution process
class StatusException(Exception):
    pass


class SatException(Exception):
    def __init__(self, satisfying_assignment: str = None):
        self._satisfying_assignment = satisfying_assignment

    # If NP-hard task is solved, then satisfying assignment is not None
    # Otherwise NP-completeness is proved
    @property
    def satisfying_assignment(self) -> str:
        return self._satisfying_assignment


class UnsatException(Exception):
    pass


class IndetException(Exception):
    pass


__all__ = [
    'SatException',
    'UnsatException',
    'IndetException',
    'StatusException'
]
