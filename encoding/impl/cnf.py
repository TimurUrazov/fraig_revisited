from typing import List, Tuple

from ..encoding import Encoding, EncodingData

Assumptions = List[int]
Constraints = List[List[int]]
Supplements = Tuple[Assumptions, Constraints]

Clause = List[int]
Clauses = List[Clause]


class CNFData(EncodingData):
    def __init__(self, clauses: Clauses, lines: str = None, max_lit: int = None, comments=None):
        self._lines = lines
        self._clauses = clauses
        self._max_lit = max_lit
        if comments is None:
            comments = {
                'outputs': []
            }
        self._comments = comments

    def _get_lines_and_max_lit(self) -> Tuple[str, int]:
        if not self._lines or not self._max_lit:
            lines, max_lit = [], 0
            for clause in self._clauses:
                max_lit = max(max_lit, *map(abs, clause))
                lines.append(' '.join(map(str, clause)) + ' 0\n')
            self._lines, self._max_lit = ''.join(lines), max_lit
        return self._lines, self._max_lit

    def _get_source_header(self, payload_len: int) -> str:
        lines, max_lit = self._get_lines_and_max_lit()
        return f'p cnf {max_lit} {len(self._clauses) + payload_len}\n'

    def clauses(self, constraints: Constraints = ()) -> Clauses:
        return [*self._clauses, *constraints]

    def add_comment(self, comment_name: str, comment_content: str):
        self._comments[comment_name] = comment_content

    def source(self, supplements: Supplements = ((), ())) -> str:
        assumptions, constraints = supplements
        lines, max_lit = self._get_lines_and_max_lit()
        payload_len = len(constraints) + len(assumptions)
        outputs = 'c outputs: ' + ' '.join([str(x) for x in self.get_outputs()])
        comments = []
        for comment_name in self._comments:
            if 'outputs' != comment_name and 'inputs' != comment_name:
                comments.append(f'c {comment_name}: {self._comments[comment_name]}')
        return ''.join([
            self._get_source_header(payload_len),
            outputs + '\n',
            *(f'{x}\n' for x in comments),
            lines, *(f'{x} 0\n' for x in assumptions),
            *(' '.join(map(str, c)) + ' 0\n' for c in constraints),
        ])

    @property
    def max_literal(self) -> int:
        return self._get_lines_and_max_lit()[1]

    def get_outputs(self) -> List[int]:
        return self._comments['outputs']

    def get_inputs(self) -> List[int]:
        return self._comments['inputs']


class CNF(Encoding):
    comment_lead = ['p', 'c']

    def __init__(self, from_clauses: Clauses = None, from_file: str = None):
        super().__init__(from_file)
        self.clauses = from_clauses
        self._outputs = None
        self._cnf_data = None

    def _parse_raw_data(self, raw_data: str):
        process_line = 1
        try:
            lines, clauses, max_lit, comments = [], [], 0, {}
            for line in raw_data.splitlines(keepends=True):
                if line[0] not in self.comment_lead:
                    clause = [int(n) for n in line.split()[:-1]]
                    max_lit = max(max_lit, *map(abs, clause))
                    clauses.append(clause)
                    lines.append(line)
                elif "c outputs" in line:
                    comments['outputs'] = list(map(int, line.split(":")[1].split()))
                elif "c inputs" in line:
                    comments['inputs'] = list(map(int, line.split(":")[1].split()))
                process_line += 1

            self._cnf_data = CNFData(
                clauses, ''.join(lines), max_lit, comments
            )
        except Exception as exc:
            msg = f'Error while parsing encoding file in line {process_line}'
            raise ValueError(msg) from exc

    def _process_raw_data(self):
        if self._cnf_data is not None:
            return

        data = self.get_raw_data()
        self._parse_raw_data(data)

    def get_data(self) -> CNFData:
        if self.clauses is not None:
            return CNFData(self.clauses)
        elif self._cnf_data is not None:
            return self._cnf_data

        self._process_raw_data()
        return self._cnf_data

    def __copy__(self):
        return CNF(
            from_file=self.filepath,
            from_clauses=self.clauses
        )


dummy = CNF(
    from_clauses=([]),
)

__all__ = [
    'CNF',
    'CNFData',
    # types
    'Clause',
    'Clauses',
    'dummy'
]
