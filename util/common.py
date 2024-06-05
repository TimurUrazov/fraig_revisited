from typing import List
from datetime import datetime


def date_now() -> str:
    return datetime.today().strftime("%Y.%m.%d-%H:%M:%S")


def string_to_bits(x) -> List[int]:
    return [int(i) for i in x]


def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type("Enum", (), enums)


def make_n_ranges(input_vars, n_of_ranges):
    l = range(2 ** len(input_vars))
    n = len(l)
    k = n_of_ranges
    return [l[i * (n // k) + min(i, n % k): (i + 1) * (n // k) + min(i + 1, n % k)] for i in range(k)]


__all__ = [
    'enum',
    'make_n_ranges',
    'date_now',
    'string_to_bits'
]
