from encoding import AAG, AndGate
from .gate import Gate
from toposort import toposort
from typing import List, Union, Tuple, Dict, Set
from collections import defaultdict
from numpy.random import RandomState


class GraphInfo:
    def __init__(self, and_inverter_graph: 'AndInverterGraph'):
        self._graph = and_inverter_graph

    def out_degrees(self) -> Dict[int, int]:
        out_degree = {}
        for vertex_num, vertex_info in self._graph.items():
            out_degree[vertex_num] = vertex_info.out_degree
        return out_degree

    def depths(self) -> Tuple[Dict[int, int], int]:
        return self._graph.get_depths()

    def number_of_dominated_variables(self) -> Dict[int, int]:
        return self._graph.shadow_variables_number()

    # The function returns graph representation of outputs without their negation marks
    @property
    def outputs(self) -> List[int]:
        return [output for output, _ in self._graph.outputs]

    def vertex_exists(self, vertex_num: int) -> bool:
        return vertex_num in self._graph
    
    def is_output(self, vertex_num: int) -> bool:
        return self._graph.is_output(vertex_num)

    @property
    def current_input_to_initial(self) -> Dict[int, int]:
        return self._graph.current_input_to_initial

    @property
    def initial_input_to_assignment(self) -> Dict[int, int]:
        return self._graph.initial_input_to_assignment

    @property
    def size(self) -> int:
        return len(self._graph)


class AndInverterGraph(dict):
    def __init__(self, aag: AAG):
        super().__init__()
        self._aag = aag
        self._inputs, self._inputs_to_indices = None, None
        self._outputs, self._outputs_to_indices = None, None
        self._initial_input_to_assignment = {}
        self._current_input_to_initial = None
        self._build_graph()

    @property
    def inputs(self) -> List[int]:
        return self._inputs

    @property
    def outputs(self) -> List[Tuple[int, int]]:
        return self._outputs

    @property
    def current_input_to_initial(self) -> Dict[int, int]:
        return self._current_input_to_initial

    @property
    def initial_input_to_assignment(self) -> Dict[int, int]:
        return self._initial_input_to_assignment

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

    # The function returns graph representation of gate in aag format: node and its negation mark
    @staticmethod
    def decode_vertex(vertex: int):
        return vertex - vertex % 2, vertex % 2

    # The function returns graph node representation in aag format depending on its negation mark
    @staticmethod
    def encode_vertex(graph_node: int, negation_mark: int):
        return graph_node + negation_mark

    def link_vertices(self, child_num: int, parent_num: int, neg_mark: int):
        self[child_num].add_parent(parent_num, neg_mark)
        self[parent_num].add_child(child_num, neg_mark)

    def update_outputs(self, outputs: List[int]):
        self._outputs = []
        self._outputs_to_indices = {}
        for index, output in enumerate(outputs):
            output_gate_num, negation_mark = self.decode_vertex(output)
            self._outputs.append((output_gate_num, negation_mark))
            if output_gate_num in self._outputs_to_indices:
                self._outputs_to_indices[output_gate_num].add(index)
            else:
                self._outputs_to_indices[output_gate_num] = {index}

    def _build_graph(self):
        def create_gate(vertex: int):
            if vertex not in self:
                self[vertex] = Gate(vertex)

        data = self._aag.get_data()

        self._inputs = data.inputs()
        self._inputs_to_indices = {input_num: {index} for index, input_num in enumerate(self._inputs)}
        self._current_input_to_initial = {i: i for i in self._inputs}

        self.update_outputs(data.outputs())

        and_gates = data.and_gates()

        for and_gate in and_gates:
            and_gate_num = and_gate.gate_num
            first_fan_in, second_fan_in = and_gate.fan_ins
            first_gate_num, first_gate_neg = self.decode_vertex(first_fan_in)
            second_gate_num, second_gate_neg = self.decode_vertex(second_fan_in)
            create_gate(and_gate_num)
            create_gate(first_gate_num)
            create_gate(second_gate_num)

            self.link_vertices(and_gate_num, first_gate_num, first_gate_neg)
            self.link_vertices(and_gate_num, second_gate_num, second_gate_neg)

        constraints = data.constraints

        if constraints:
            for constraint in constraints:
                first_vertex, second_vertex = constraint
                first_v = first_vertex - first_vertex % 2
                second_v = second_vertex - second_vertex % 2
                first_neg_mark = first_vertex % 2
                second_neg_mark = second_vertex % 2
                self[first_v].add_constraint(first_neg_mark, second_v, second_neg_mark)
                self[second_v].add_constraint(second_neg_mark, first_v, first_neg_mark)

    # The function replaces node in constraints when we replace vertex_to_replace with vertex_to_set
    def replace_constraints(self, vertex_to_replace_num: int, vertex_to_set_num: int, equivalence_negation_mark: int):
        vertex_to_replace = self[vertex_to_replace_num]
        pairs_in_constraints = vertex_to_replace.pairs_in_constraints()

        constraints_to_add = []
        for vertex_to_replace_negation_mark, vertex_in_pair_num, vertex_in_pair_neg_mark in pairs_in_constraints:
            if vertex_in_pair_num == vertex_to_replace_num:
                # remove helps when vertex_to_set_num == vertex_in_pair == vertex_to_replace_num
                self[vertex_to_replace_num].remove_constraint(
                    vertex_to_replace_negation_mark, vertex_in_pair_num, vertex_in_pair_neg_mark
                )
                constraints_to_add.append((
                    vertex_to_set_num,
                    vertex_to_replace_negation_mark ^ equivalence_negation_mark,
                    vertex_to_set_num,
                    vertex_in_pair_neg_mark ^ equivalence_negation_mark
                ))
            else:
                self[vertex_in_pair_num].remove_constraint(
                    vertex_in_pair_neg_mark, vertex_to_replace_num, vertex_to_replace_negation_mark
                )
                self[vertex_to_replace_num].remove_constraint(
                    vertex_to_replace_negation_mark, vertex_in_pair_num, vertex_in_pair_neg_mark
                )
                constraints_to_add.append((
                    vertex_in_pair_num,
                    vertex_in_pair_neg_mark,
                    vertex_to_set_num,
                    vertex_to_replace_negation_mark ^ equivalence_negation_mark
                ))
                constraints_to_add.append((
                    vertex_to_set_num,
                    vertex_to_replace_negation_mark ^ equivalence_negation_mark,
                    vertex_in_pair_num,
                    vertex_in_pair_neg_mark
                ))
        for v_to_set_value, v_to_set_value_neg_mark, v_in_pair, v_in_pair_neg_mark in constraints_to_add:
            self[v_to_set_value].add_constraint(v_to_set_value_neg_mark, v_in_pair, v_in_pair_neg_mark)

    # The function updates inputs or outputs when we replace one node with another
    def update_inputs_or_outputs(
            self,
            vertex_to_del: int,
            new_vertex: int,
            negation_mark: int
    ):
        def update_elements(
                vertex_to_remove: int,
                vertex_to_set: int,
                elements: List[Union[int, Tuple[int, int]]],
                elements_to_indices: Dict[int, Set[int]],
                is_output: bool
        ):
            if vertex_to_remove in elements_to_indices:
                for vertex_to_replace in elements_to_indices[vertex_to_remove]:
                    if is_output:
                        negation_mark_updated = elements[vertex_to_replace][1] ^ negation_mark
                        elements[vertex_to_replace] = (vertex_to_set, negation_mark_updated)
                    else:
                        self._current_input_to_initial[vertex_to_set] = self._current_input_to_initial[
                            elements[vertex_to_replace]
                        ]
                        del self._current_input_to_initial[elements[vertex_to_replace]]
                        elements[vertex_to_replace] = vertex_to_set
                if vertex_to_set in elements_to_indices:
                    elements_to_indices[vertex_to_set] |= elements_to_indices[vertex_to_remove]
                else:
                    elements_to_indices[vertex_to_set] = elements_to_indices[vertex_to_remove]
                if vertex_to_set != vertex_to_remove:
                    del elements_to_indices[vertex_to_remove]

        update_elements(vertex_to_del, new_vertex, self._outputs, self._outputs_to_indices, True)
        update_elements(vertex_to_del, new_vertex, self._inputs, self._inputs_to_indices, False)

    # The function checks if node to assign belongs to input or output and assign them with value if necessary
    def assign_input_or_output(self, assigned_node: int, assigned_value: int):
        # noinspection PyTypeChecker
        def assign_element(
                elements_to_indices: Dict[int, Set[int]],
                elements: Union[List[Tuple[int, int]], List[int]],
                node: int,
                value: int,
                is_output: bool
        ):
            if node in elements_to_indices:
                for element_index in elements_to_indices[node]:
                    if is_output:
                        elem, mark = elements[element_index]
                        elements[element_index] = (value ^ mark, 0)
                    else:
                        self._initial_input_to_assignment[
                            self._current_input_to_initial[node]
                        ] = value
                        del self._current_input_to_initial[node]
                        elements[element_index] = value
                del elements_to_indices[node]

        assign_element(self._outputs_to_indices, self._outputs, assigned_node, assigned_value, True)
        assign_element(self._inputs_to_indices, self._inputs, assigned_node, assigned_value, False)

    # The function returns layers of graph ordered topologically
    def get_layers(self):
        graph = {}
        for v in self:
            graph[v] = set([vertex for vertex, _ in self[v].parents])
        layers = list(toposort(graph))
        return layers

    # The function counts depths in topological order (starting from inputs) and max depth
    def get_depths(self):
        layers = self.get_layers()

        depth = {}

        max_depth = 0

        for layer_num, layer in enumerate(layers):
            for variable in layer:
                depth[variable] = layer_num
            max_depth = layer_num

        return depth, max_depth

    def shadow_variables_number(self) -> Dict[int, int]:
        shadow_variables_numbers = defaultdict(lambda: 1)

        for layer in reversed(self.get_layers()):
            for vertex_num in layer:
                for parent, _ in self[vertex_num].parents:
                    shadow_variables_numbers[parent] += shadow_variables_numbers[vertex_num]

        return shadow_variables_numbers

    # The function retrieves data from graph representation needed for aag format
    def _retrieve_data(self) -> Tuple[List[AndGate], List[int], List[Tuple[int, int]]]:

        # The function add node constraints to all constraints
        def collect_constraints(global_constraints: Set[Tuple[int, int]], node_constraints: List[Tuple[int, int]]):
            for less_node_num, greater_node_num in node_constraints:
                global_constraints.add((less_node_num, greater_node_num))

        constraints = set()
        layers = self.get_layers()

        no_parent_nodes = []
        and_gates = []
        for layer in layers:
            for gate in layer:
                node = self[gate]
                collect_constraints(constraints, node.constraints)
                parents = node.parents
                if node.in_degree == 0:
                    no_parent_nodes.append(gate)
                    continue

                assert node.in_degree == 2

                and_gates.append(
                    AndGate(
                        gate, (
                            AndInverterGraph.encode_vertex(*parents[0]),
                            AndInverterGraph.encode_vertex(*parents[1])
                        )
                    ))

        # Some inputs may become equal after normalization
        for no_parent_node in no_parent_nodes:
            assert no_parent_node in self._inputs_to_indices

        for input_to_index in self._inputs_to_indices:
            assert input_to_index in no_parent_nodes

        return and_gates, list(no_parent_nodes), list(constraints)

    def generate_inputs_and_outputs(self, seed=42) -> Tuple[str, str]:
        def array_to_str(array) -> str:
            return ''.join(list(map(str, array)))

        rand_input_values = RandomState(seed=seed).randint(0, [2] * len(self._inputs))
        rand_input_dict = dict(zip(self._inputs, rand_input_values))
        generated_outputs_dict = self.generate_outputs_given_inputs(rand_input_dict)
        generated_outputs = [generated_outputs_dict[output ^ neg] for output, neg in self._outputs]
        return array_to_str(rand_input_values), array_to_str(generated_outputs)

    def generate_outputs_given_inputs(self, generated: Dict[int, int]) -> Dict[int, int]:
        def generate_node(node_number: int):
            if node_number in generated:
                return

            node = self[node_number]

            assert node.in_degree == 2

            parents = node.parents

            parent_values = []

            for parent_num, negation in parents:
                if parent_num not in generated:
                    generate_node(parent_num)
                parent_values.append(generated[parent_num] ^ negation)

            current_simulation_vector = parent_values[0] & parent_values[1]

            generated[node_number] = current_simulation_vector

        for node_num in self:
            generate_node(node_num)

        return {node_num ^ neg: generated[node_num] ^ neg for node_num, neg in self._outputs}

    def info(self) -> GraphInfo:
        return GraphInfo(self)

    def is_output(self, vertex_num: int) -> bool:
        return vertex_num in self._outputs_to_indices

    # The function translates graph to aag format
    def to_aag(self) -> AAG:
        and_gates, inputs, constraints = self._retrieve_data()
        return AAG(
            from_gates_and_inputs=(
                inputs,
                [output_ + negation_mark for output_, negation_mark in self._outputs],
                and_gates
            ),
            constraints=constraints
        )


__all__ = [
    'AndInverterGraph',
    'GraphInfo'
]
