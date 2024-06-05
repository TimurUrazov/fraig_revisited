from typing import List, Tuple, Iterable,Dict
from ..encoding import Encoding, EncodingData
from .cnf import CNF, dummy
from itertools import accumulate
from util import FileExtension, CONFIG, script, AAG2CNF
from copy import copy
import json


OutputAssumptions = Iterable[int]
AndGates = List[Tuple[int, int, int]]
Inputs = List[int]
OutputGates = List[int]

Assignment = Tuple[int, int]
Assignments = List[Assignment]

Constraint = List[int]
Constraints = List[Constraint]


class AndGate:
    def __init__(self, gate_num: int, fan_ins: Tuple[int, int]):
        self._gate_num = gate_num
        self._fan_ins = fan_ins

    @property
    def gate_num(self) -> int:
        return self._gate_num

    @property
    def fan_ins(self) -> Tuple[int, int]:
        return self._fan_ins

    def __iter__(self):
        yield self._gate_num
        for fan_in in self._fan_ins:
            yield fan_in


GatesAndInputs = Tuple[Inputs, OutputGates, List[AndGate]]


class AAGData(EncodingData):
    comment_lead = 'aag'

    def __init__(
            self,
            gates_and_inputs: GatesAndInputs = None,
            constraints: List[Tuple[int, int]] = None,
            max_gate: int = None,
            lines: str = None
    ):
        self._max_gate = max_gate
        self._inputs, self._outputs, self._and_gates = gates_and_inputs
        self._n_of_inputs = len(self._inputs)
        self._n_of_outputs = len(self._outputs)
        self._n_of_and_gates = len(self._and_gates)
        self._lines = lines
        self._constraints = constraints
        self._comments = {}

    def _get_source_header(self) -> str:
        _, max_gate = self._get_lines_and_max_gate()
        header_parts = [max_gate, self._n_of_inputs, 0, self._n_of_outputs, self._n_of_and_gates]
        return f'{self.comment_lead} {" ".join(map(str, header_parts))}'

    def _get_lines_and_max_gate(self):
        if not self._lines or not self._max_gate:
            lines, max_gate = [], 0
            lines.extend(map(str, self._inputs + self._outputs))
            for and_gate in self._and_gates:
                lines.append(" ".join(map(str, and_gate)))
                max_gate = max(max_gate, *map(lambda x: x // 2, and_gate))
            lines = [line + '\n' for line in lines]
            self._lines = ''.join(lines)
            self._max_gate = max_gate
        return self._lines, self._max_gate

    @property
    def constraints(self):
        if self._constraints:
            return self._constraints.copy()
        return None

    @constraints.setter
    def constraints(self, constraints):
        self._constraints = constraints

    def and_gates(self) -> List[AndGate]:
        return self._and_gates.copy()

    def inputs(self) -> List[int]:
        return self._inputs.copy()

    def vertices(self) -> List[int]:
        return [*[output - output % 2 for output in self._outputs], *[fan_in for fan_in, _, _ in self._and_gates]]

    def outputs(self, assumptions: OutputAssumptions = ()) -> List[int]:
        return [*self._outputs, *list(set(assumptions) - set(self._outputs))]

    def add_comment(self, comment_name: str, comment_content: str):
        self._comments[comment_name] = comment_content

    def source(self) -> str:
        source = []
        if self._constraints:
            source.append('c constraints: ' + json.dumps(self._constraints))
        for comment_name in self._comments:
            source.append(f'c {comment_name}: {self._comments[comment_name]}')
        lines, max_gate = self._get_lines_and_max_gate()
        source.extend([self._get_source_header(), lines])
        return '\n'.join(source)

    @property
    def max_literal(self) -> int:
        if not self._lines or not self._max_gate:
            _, max_gate = self._get_lines_and_max_gate()
            return max_gate
        return self._max_gate

    @property
    def n_of_and_gates(self) -> int:
        return self._n_of_and_gates

    @property
    def n_of_vertices(self) -> int:
        return len(self.vertices())

    @property
    def n_of_inputs(self) -> int:
        return self._n_of_inputs

    @property
    def n_of_outputs(self) -> int:
        return self._n_of_outputs


class AAG(Encoding):
    comment_lead = ['p', 'c']

    def __init__(
            self, from_file: str = None,
            from_gates_and_inputs: GatesAndInputs = None,
            constraints: List[Tuple[int, int]] = None
    ):
        super().__init__(from_file)
        self._gates_and_inputs = from_gates_and_inputs
        self._constraints = constraints
        self._cnf = None
        self._aag_data = None
        self._input_vars = None

    @staticmethod
    def _decompose_aag_header(aag_header):
        parts_of_header = list(map(int, aag_header.split()[1:]))
        max_gate = parts_of_header[0]
        n_of_inputs = parts_of_header[1]
        n_of_outputs = parts_of_header[3]
        n_of_and_gates = parts_of_header[4]
        return max_gate, n_of_and_gates, n_of_inputs, n_of_outputs

    def _parse_raw_data(self, raw_data: str):
        process_line = 1
        try:
            lines = raw_data.splitlines()
            inputs, outputs, and_gates, constraints = [], [], [], None
            n_of_inputs, n_of_outputs, n_of_and_gates = None, None, None
            split_indices = []
            max_gate = 0
            for index, line in enumerate(lines):
                if line[0] in self.comment_lead:
                    if "c constraints" in line:
                        constraints = json.loads(line.split(":")[1])
                else:
                    if 'aag' in line:
                        max_gate, n_of_and_gates, n_of_inputs, n_of_outputs = AAG._decompose_aag_header(line)
                    else:
                        if process_line == 1:
                            split_indices = list(accumulate([index, n_of_inputs, n_of_outputs, n_of_and_gates]))
                            process_line = index
                        if process_line < split_indices[1]:
                            inputs.append(int(line))
                        elif process_line < split_indices[2]:
                            outputs.append(int(line))
                        elif process_line < split_indices[3]:
                            and_gates.append(list(map(int, line.split())))

                        process_line += 1
            self._aag_data = AAGData(
                gates_and_inputs=(
                    inputs,
                    outputs,
                    list(map(lambda x: AndGate(x[0], (x[1], x[2])), and_gates))
                ),
                max_gate=max_gate,
                constraints=constraints
            )
        except Exception as exc:
            msg = f'Error while parsing encoding file {self.filepath} in line {process_line}'
            raise ValueError(msg) from exc

    def _process_raw_data(self):
        if self._aag_data is not None:
            return

        data = self.get_raw_data()
        self._parse_raw_data(data)

    def get_data(self) -> AAGData:
        if self._aag_data is not None:
            return self._aag_data
        elif self._gates_and_inputs:
            self._aag_data = AAGData(self._gates_and_inputs, self._constraints)
        else:
            self._process_raw_data()
        return self._aag_data

    def to_cnf(self, filename: str = 'cnf') -> CNF:
        if self._cnf is None:
            self.write_raw_data(filepath=CONFIG.path_to_out().to_file(f'{filename}{FileExtension.AAG}'))
            script.call_handmade_aag_to_cnf_from_out(filename)
            self._cnf = CNF(from_file=CONFIG.path_to_out().to_file(f'{filename}{FileExtension.CNF}'))
        return self._cnf

    def to_cnf_with_constraints(self) -> CNF:
        if self._cnf is None:
            data = self.get_data()
            constraints = data.constraints

            self.get_data().constraints = None
            aag_to_cnf = AAG2CNF()
            if self._gates_and_inputs == ([], [], []):
                self._cnf = dummy
            else:
                self._cnf = CNF(
                    from_raw_data=aag_to_cnf.aag_to_cnf_raw_data(self.get_data().source(), constraints)
                )
                self.get_data().constraints = constraints
        return self._cnf

    def to_cnf_with_mapping(self) -> Tuple[CNF, Dict[int, int]]:
        if self._cnf is None:
            data = self.get_data()
            constraints = data.constraints

            self.get_data().constraints = None
            aag_to_cnf = AAG2CNF()
            if self._gates_and_inputs == ([], [], []):
                self._cnf = dummy
            else:
                self._cnf = CNF(
                    from_raw_data=aag_to_cnf.aag_to_cnf_raw_data(self.get_data().source(), constraints)
                )
                self.get_data().constraints = constraints
                self._input_vars = aag_to_cnf.aag_to_cnf_inputs_vars
        return self._cnf, self._input_vars

    def __copy__(self):
        return AAG(
            from_file=self.filepath,
            from_gates_and_inputs=copy(self._gates_and_inputs),
            constraints=copy(self._constraints)
        )


__all__ = [
    'AAG',
    'AAGData',
    'AndGate',
    'GatesAndInputs',
    'Constraints'
]
