from typing import List


# Abstract error we encounter when attempting to make conflict assignment among one of the following types
class ConflictAssignmentException(Exception):
    def __init__(self, vertex: int, suggestion: int):
        self._vertex = vertex
        self._suggestion = suggestion


# Controversial assignment: we are trying to assign value to vertex which has been already assigned with 1 - value
class ControversialAssignmentException(ConflictAssignmentException):
    def __str__(self) -> str:
        return (f'Attempt to assign {self._vertex} with {self._suggestion}. '
                f'But it was already assigned with {1 - self._suggestion}.')


# Non-existent assignment: the assignment which can't be made because vertex always takes other value
class NonExistentAssignmentException(ConflictAssignmentException):
    def __str__(self) -> str:
        return (f'Attempt to assign {self._vertex} with {self._suggestion}. '
                f'But {self._vertex} never takes {self._suggestion} value.')


# Constraint violation assignment
class ConstraintViolationAssignmentException(ConflictAssignmentException):
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


__all__ = [
    'ConflictAssignmentException',
    'ControversialAssignmentException',
    'NonExistentAssignmentException',
    'ConstraintViolationAssignmentException'
]
