import argparse


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-nf', '--namefirst', nargs='?', default='first.aig', help='First circuit file in AIGER (ASCII) format'
    )
    parser.add_argument(
        '-ns', '--namesecond', nargs='?', default='second.aig', help='First circuit file in AIGER (ASCII) format'
    )
    parser.add_argument("-lim", "--limit", nargs="?", type=int, default=100)
    return parser


__all__ = [
    'create_parser'
]
