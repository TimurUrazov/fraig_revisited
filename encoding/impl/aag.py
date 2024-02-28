from typing import List, Tuple, Iterable, Dict, Set
from ..encoding import Encoding, EncodingData
from .cnf import CNF, dummy
from itertools import accumulate
from util import FileExtension, CONFIG, script, aag_to_cnf, write_cnf
from copy import copy, deepcopy
import json


OutputAssumptions = Iterable[int]
AndGates = List[List[int]]
Inputs = List[int]
OutputGates = List[int]
GatesAndInputs = Tuple[Inputs, OutputGates, AndGates]

Assignment = Tuple[int, int]
Assignments = List[Assignment]

Constraint = List[int]
Constraints = List[Constraint]


class Gate:
    def __init__(self, number: int, is_output_gate: bool = False):
        self._node_number = number
        self._parents = {}
        self._children = {}
        self._out_degree = 0
        self._in_degree = 0
        self._is_output_gate = is_output_gate
        self._assignment = None
        # Node can be included in constraint negated or not (depending on index in self._constraints)
        # Elements are mapping to vertices and their negation marks in constraints
        self._constraints = [{}, {}]

    @property
    def in_degree(self) -> int:
        return self._in_degree

    @property
    def out_degree(self) -> int:
        return self._out_degree

    def add_parent(self, parent: int, is_negative: int) -> bool:
        parent_appeared = False
        if parent in self._parents:
            if self._parents[parent] ^ is_negative == 1:
                self._parents[parent] = 2
                self._in_degree += 1
                parent_appeared = True
        else:
            self._parents[parent] = is_negative
            self._in_degree += 1
            parent_appeared = True
        return parent_appeared

    def add_child(self, child: int, is_negative: int) -> bool:
        child_appeared = False
        if child in self._children:
            if self._children[child] ^ is_negative == 1:
                self._children[child] = 2
                self._out_degree += 1
                child_appeared = True
        else:
            self._children[child] = is_negative
            self._out_degree += 1
            child_appeared = True
        return child_appeared

    def remove_parent(self, parent: int, is_negative: int) -> bool:
        parent_removed = False
        if parent in self._parents:
            parent_removed = True
            self._in_degree -= 1
            if self._parents[parent] == 2:
                self._parents[parent] = is_negative ^ 1
            else:
                del self._parents[parent]
        return parent_removed

    def remove_child(self, child, is_negative: int) -> bool:
        child_removed = False
        if child in self._children:
            child_removed = True
            self._out_degree -= 1
            if self._children[child] == 2:
                self._children[child] = is_negative ^ 1
            else:
                del self._children[child]
        return child_removed

    def remove_children(self):
        self._children = {}
        self._out_degree = 0

    def set_assignment(self, assignment: int):
        self._assignment = assignment

    def _retrieve_constraints(self, negation_mark):
        constraints = []
        for node_n, node_neg in self._constraints[negation_mark].items():
            if node_neg == 2:
                constraints.extend([(node_n, 0), (node_n, 1)])
            else:
                constraints.append((node_n, node_neg))
        return constraints

    @property
    def constraints(self) -> List[Tuple[int, int]]:
        def order_nodes(first_node: int, second_node: int) -> Tuple[int, int]:
            if first_node > second_node:
                return second_node, first_node
            else:
                return first_node, second_node

        constraints = []
        for negation_mark in [0, 1]:
            for node_n, node_neg in self._retrieve_constraints(negation_mark):
                constraints.append(order_nodes(self.node_number + negation_mark, node_n + node_neg))
        return constraints

    @property
    def assignment(self) -> int:
        return self._assignment

    @assignment.setter
    def assignment(self, assignment: int):
        self._assignment = assignment

    @property
    def is_output_gate(self) -> bool:
        return self._assignment

    @property
    def node_number(self) -> int:
        return self._node_number

    @node_number.setter
    def node_number(self, number: int):
        self._node_number = number

    @property
    def children(self) -> List[Tuple[int, int]]:
        children = []
        for child_number, child_value in self._children.items():
            if child_value == 2:
                children.extend([(child_number, 0), (child_number, 1)])
            else:
                children.append((child_number, child_value))
        return children

    @property
    def parents(self) -> List[Tuple[int, int]]:
        parents = []
        for parent_number, parent_value in self._parents.items():
            if parent_value == 2:
                parents.extend([(parent_number, 0), (parent_number, 1)])
            else:
                parents.append((parent_number, parent_value))
        return parents

    # When current_node_negation ^ current_node_value == 0, then other_node_negation ^ other_node_value should be 1 for
    # all other nodes in constraint with x. So we can derive such values and check whether they violate constraints.
    def derive_values_from_constraints(self, assigned_value: int) -> List[Tuple[int, int]]:
        derived_values = []
        for node_num, negation_mark in self._retrieve_constraints(assigned_value):
            derived_values.append((node_num, negation_mark ^ 1))
        return derived_values

    # When current_node_negation ^ current_node_value == 1, then other_node_negation ^ other_node_value can take
    # arbitrary values
    def derive_nodes_to_remove_constraints_with(self, assigned_value: int) -> List[Tuple[int, int]]:
        nodes_to_remove_constraints_with = []
        for node_num, negation_mark in self._retrieve_constraints(assigned_value ^ 1):
            nodes_to_remove_constraints_with.append((node_num, negation_mark))
        return nodes_to_remove_constraints_with

    # The function removes constraint with current node in which it is included with such a negation mark and
    # other node
    def remove_constraint(self, negation_mark: int, other_node: int, other_node_negation_mark: int):
        if self._constraints[negation_mark][other_node] == 2:
            self._constraints[negation_mark][other_node] = other_node_negation_mark ^ 1
        else:
            del self._constraints[negation_mark][other_node]

    # When we add constraint in which one vertex coincides with another but another differs only in negation then
    # we conclude that current_vertex ^ negation_mark should be 1
    def check_constraint_contradiction(self, negation_mark: int, other_node: int, other_node_neg_mark: int) -> Tuple[
        bool, List[int]
    ]:
        bucket = self._constraints[negation_mark]
        contradiction = other_node in bucket and (
                bucket[other_node] == 2 or bucket[other_node] ^ other_node_neg_mark == 1
        )
        constraints_to_remove = []
        if other_node in bucket:
            if bucket[other_node] == 2:
                constraints_to_remove.extend([0, 1])
            else:
                constraints_to_remove.append(bucket[other_node])
        return not contradiction, constraints_to_remove

    # The function adds constraint
    def add_constraint(self, negation_mark: int, other_node: int, other_node_neg_mark: int):
        if other_node not in self._constraints[negation_mark]:
            self._constraints[negation_mark][other_node] = other_node_neg_mark
        elif self._constraints[negation_mark][other_node] ^ other_node_neg_mark == 1:
            self._constraints[negation_mark][other_node] = 2

    # The function removes all constraints with this current node in which it takes such an assigned_value
    def remove_constraints(self, assigned_value: int):
        self._constraints[assigned_value ^ 1] = {}

    # The function returns negation mark of current vertex in constraint and other vertex, and it's negation mark
    # in this constraint
    def pairs_in_constraints(self) -> List[Tuple[int, int, int]]:
        pairs_in_constraints = []
        for negation_mark in [0, 1]:
            for node_n, node_neg in self._retrieve_constraints(negation_mark):
                pairs_in_constraints.append((negation_mark, node_n, node_neg))
        return pairs_in_constraints


class AndInverterGraph(dict):
    def __str__(self):
        graph = []
        for node in self:
            graph.append("node number: {}".format(node))
            graph.append("parents: {}".format(self[node].parents))
            graph.append("children: {}".format(self[node].children))
        return '\n'.join(graph)

    def __getitem__(self, item: int) -> Gate:
        return super().__getitem__(item)

    def __setitem__(self, key: int, value: Gate):
        super().__setitem__(key, value)


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
        self._vertices = None
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

    def constraints(self):
        if self._constraints:
            return self._constraints.copy()
        return None

    def and_gates(self) -> List[List[int]]:
        return self._and_gates.copy()

    def inputs(self) -> List[int]:
        return self._inputs.copy()

    def vertices(self) -> List[int]:
        if self._vertices:
            return self._vertices
        vertices = []
        for and_gate in self._and_gates:
            vertices.append(and_gate[0])
        self._vertices = vertices
        return [*self._vertices, *self._inputs]

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
            constraints: List[Tuple[int, int]] = None,
            graph: Dict[int, Gate] = None
    ):
        super().__init__(from_file)
        self._gates_and_inputs = from_gates_and_inputs
        self._cnf = None
        self._aag_data = None
        self._constraints = constraints
        self._graph = graph

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
                gates_and_inputs=(inputs, outputs, and_gates),
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

    @staticmethod
    def _get_node_number_in_graph(gate):
        if gate % 2 == 1:
            return gate - 1
        return gate

    def _build_graph(self):
        and_gates = self.get_data().and_gates()
        self._graph = AndInverterGraph()
        for outpt in self.get_data().outputs():
            outpt = AAG._get_node_number_in_graph(outpt)
            self._graph[outpt] = Gate(outpt)
        for and_gate in and_gates:
            first = AAG._get_node_number_in_graph(and_gate[1])
            second = AAG._get_node_number_in_graph(and_gate[2])
            if and_gate[0] not in self._graph:
                self._graph[and_gate[0]] = Gate(and_gate[0])
            if first not in self._graph:
                self._graph[first] = Gate(first)
            if second not in self._graph:
                self._graph[second] = Gate(second)
            self._graph[and_gate[0]].add_parent(first, abs(and_gate[1] - first))
            self._graph[and_gate[0]].add_parent(second, abs(and_gate[2] - second))
            self._graph[first].add_child(and_gate[0], abs(and_gate[1] - first))
            self._graph[second].add_child(and_gate[0], abs(and_gate[2] - second))
        if self.get_data().constraints():
            for constraint in self.get_data().constraints():
                first_vertex, second_vertex = constraint
                first_v = first_vertex - first_vertex % 2
                second_v = second_vertex - second_vertex % 2
                first_neg_mark = first_vertex % 2
                second_neg_mark = second_vertex % 2
                self._graph[first_v].add_constraint(first_neg_mark, second_v, second_neg_mark)
                self._graph[second_v].add_constraint(second_neg_mark, first_v, first_neg_mark)

    def get_data(self) -> AAGData:
        if self._gates_and_inputs:
            self._aag_data = AAGData(self._gates_and_inputs, self._constraints)
            return self._aag_data
        elif self._aag_data is not None:
            return self._aag_data

        self._process_raw_data()
        return self._aag_data

    def get_graph(self) -> Dict[int, Gate]:
        if self._graph is None:
            self._build_graph()
        return self._graph

    def to_cnf(self, filename: str = 'cnf') -> CNF:
        if self._cnf is None:
            self.write_raw_data(filepath=CONFIG.path_to_out().to_file(f'{filename}{FileExtension.AAG}'))
            script.call_handmade_aag_to_cnf_from_out(filename)
            self._cnf = CNF(from_file=CONFIG.path_to_out().to_file(f'{filename}{FileExtension.CNF}'))
        return self._cnf

    def to_cnf_with_constraints(self, filename: str = CONFIG.path_to_out().to_file('cnf')) -> CNF:
        if self._cnf is None:
            filepath_in = f'{filename}{FileExtension.AAG}'
            constraints = []
            if self._constraints:
                constraints = self._constraints.copy()
            self._constraints = None
            self.write_raw_data(filepath=filepath_in)
            self._constraints = constraints
            filepath_out = f'{filename}{FileExtension.CNF}'
            if self._gates_and_inputs == ([], [], []):
                self._cnf = dummy
            else:
                aag_to_cnf(filepath_in, filepath_out, self._constraints)
                self._cnf = CNF(from_file=filepath_out)
        return self._cnf

    def __copy__(self):
        return AAG(
            from_file=self.filepath,
            from_gates_and_inputs=copy(self._gates_and_inputs),
            graph=deepcopy(self._graph),
            constraints=copy(self._constraints)
        )


__all__ = [
    'AAG',
    'AAGData',
    'AndInverterGraph',
    'Gate',
    # types
    'AndGates',
    'Inputs',
    'OutputGates',
    'GatesAndInputs'
]
