from typing import List


class IncompatibleSubstitution(Exception):
    def __init__(self, time, conflicts):
        self.time = time
        self.conflicts = conflicts


class ConflictAssignment(Exception):
    def __init__(self, vertex: int, suggestion: int):
        self._vertex = vertex
        self._suggestion = suggestion


# Controversial assignment: we are trying to assign value to vertex which has been already assigned with 1 - value
class ControversialAssignment(ConflictAssignment):
    def __str__(self) -> str:
        return (f'Attempt to assign {self._vertex} with {self._suggestion}. '
                f'But it was already assigned with {1 - self._suggestion}.')


# Constraint violation
class NonExistentAssignment(ConflictAssignment):
    def __str__(self) -> str:
        return (f'Attempt to assign {self._vertex} with {self._suggestion}. '
                f'But {self._vertex} never takes {self._suggestion} value.')


# Non-existent assignment: the assignment which can't be made because vertex always takes other value
class ConstraintViolationAssignment(ConflictAssignment):
    def __init__(
            self,
            vertex: int,
            suggestion: int,
            constraint: List[int],
            first_node_negation: int,
            second_node_negation: int
    ):
        super().__init__(vertex, suggestion)
        self._constraint = constraint
        self._first_node_negation = first_node_negation
        self._second_node_negation = second_node_negation

    def __str__(self) -> str:
        if self._first_node_negation == 1:
            self._constraint[0] = -self._constraint[0]
        if self._second_node_negation == 1:
            self._constraint[1] = -self._constraint[1]
        return (f'Attempt to assign {self._vertex} with {self._suggestion}. '
                f'But it violates {self._constraint} constraint.')


# Don't know the case in which we can encounter such an error, so it's for self-checking
class ControversialConstraintChange(Exception):
    def __init__(self, first_node: int, first_neg_mark: int, second_node: int):
        self._first_node = first_node
        self._first_neg_mark = first_neg_mark
        self._second_node = second_node

    def __str__(self) -> str:
        if self._first_neg_mark == 1:
            return (f'Attempt to change [-{self._first_node}, {self._second_node}] with [-{self._first_node}, '
                    f'-{self._second_node}] or vise versa.')
        else:
            return (f'Attempt to change [{self._first_node}, {self._second_node}] with [{self._first_node}, '
                    f'-{self._second_node}] or vise versa.')
