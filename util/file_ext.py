from enum import Enum


class FileExtension(Enum):
    [
        AAG,
        AIG,
        CNF,
        DOT
    ] = [
        '.aag',
        '.aig',
        '.cnf',
        '.dot'
    ]

    def __str__(self):
        return self.value


__all__ = [
    'FileExtension'
]
