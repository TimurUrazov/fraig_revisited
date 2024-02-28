from typing import Set, Tuple, List, Dict, Union
from .aag import Gate
from toposort import toposort
from aig_substitutions import ConflictAssignment


class Normalize:
    def __init__(
            self,
            graph: Dict[int, Gate],
            inputs: List[int],
            outputs: List[Tuple[int, int]],
            inputs_to_indices: Dict[int, Set[int]],
            outputs_to_indices: Dict[int, Set[int]]
    ):
        self._graph = graph
        self._inputs = inputs
        self._inputs_to_indices = inputs_to_indices
        self._outputs = outputs
        self._outputs_to_indices = outputs_to_indices
        self._changed_out_degrees = {}

    def _normalize_graph(self):
        visited_vertices = set()
        graph = self._graph

        def change_pointers(node: Gate, parent: Tuple[int, int], initial_node_num: int, current_node_num: int):
            node.add_parent(*parent)
            parent_number = parent[0]
            parent_negation_mark = parent[1]
            graph[parent_number].remove_child(current_node_num, parent_negation_mark)
            graph[parent_number].add_child(initial_node_num, parent_negation_mark)

        def normalize_graph_impl(initial_node_number: int):
            if initial_node_number in visited_vertices:
                return
            visited_vertices.add(initial_node_number)

            node = graph[initial_node_number]

            if node.in_degree != 1:
                return

            node_parents = node.parents

            current_node_num = initial_node_number
            parents = node_parents
            node_parents_negation_mark = 0

            children_buckets = [node.children, []]
            literal_buckets = [[initial_node_number], []]

            for child_num, child_neg in graph[current_node_num].children:
                graph[child_num].remove_parent(current_node_num, child_neg)

            while len(parents) == 1:
                parent = parents[0]
                parent_negation_mark = parent[1]
                node_parents_negation_mark ^= parent_negation_mark
                parent_node = parent[0]
                parents = graph[parent_node].parents
                for child_num, child_neg in graph[parent_node].children:
                    if current_node_num != child_num:
                        children_buckets[node_parents_negation_mark].append((child_num, child_neg))
                    graph[child_num].remove_parent(parent_node, child_neg)
                    graph[parent_node].remove_child(child_num, child_neg)
                literal_buckets[node_parents_negation_mark].append(parent_node)
                current_node_num = parent_node

            assert len(parents) % 2 == 0

            if len(parents) == 2:
                change_pointers(node, parents[0], initial_node_number, current_node_num)
                change_pointers(node, parents[1], initial_node_number, current_node_num)
            node.remove_children()
            for child_num, child_neg in children_buckets[node_parents_negation_mark ^ 1]:
                graph[initial_node_number].add_child(child_num, child_neg ^ 1)
                graph[child_num].add_parent(initial_node_number, child_neg ^ 1)
            for child_num, child_neg in children_buckets[node_parents_negation_mark]:
                graph[initial_node_number].add_child(child_num, child_neg)
                graph[child_num].add_parent(initial_node_number, child_neg)

            for literal in literal_buckets[node_parents_negation_mark]:
                visited_vertices.add(literal)
                Normalize.update_inputs_or_outputs(
                    literal,
                    initial_node_number,
                    self._outputs,
                    self._outputs_to_indices,
                    self._inputs,
                    self._inputs_to_indices,
                    0
                )
                Normalize.replace_constraints(
                    self._graph, literal, initial_node_number, 0
                )
                if literal != initial_node_number:
                    del graph[literal]

            for literal in literal_buckets[node_parents_negation_mark ^ 1]:
                visited_vertices.add(literal)
                Normalize.update_inputs_or_outputs(
                    literal,
                    initial_node_number,
                    self._outputs,
                    self._outputs_to_indices,
                    self._inputs,
                    self._inputs_to_indices,
                    1
                )
                Normalize.replace_constraints(
                    self._graph, literal, initial_node_number, 1
                )
                if literal != initial_node_number:
                    del graph[literal]

            for node_n, _ in node.parents:
                normalize_graph_impl(node_n)

        # We go layer by layer not to turn into situation in which we got one parent vertex because
        # lower layers we processed later
        for layer in self._get_layers_from_graph():
            for vertex in layer:
                normalize_graph_impl(vertex)

    # The function updates inputs or outputs when we replace one node with another
    @staticmethod
    def update_inputs_or_outputs(
            vertex_to_del: int,
            new_vertex: int,
            outputs: List[Tuple[int, int]],
            outputs_to_indices: Dict[int, Set[int]],
            inputs: List[int],
            inputs_to_indices: Dict[int, Set[int]],
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
                        elements[vertex_to_replace] = vertex_to_set
                if vertex_to_set in elements_to_indices:
                    elements_to_indices[vertex_to_set] |= elements_to_indices[vertex_to_remove]
                else:
                    elements_to_indices[vertex_to_set] = elements_to_indices[vertex_to_remove]
                if vertex_to_set != vertex_to_remove:
                    del elements_to_indices[vertex_to_remove]

        update_elements(vertex_to_del, new_vertex, outputs, outputs_to_indices, True)
        update_elements(vertex_to_del, new_vertex, inputs, inputs_to_indices, False)

    # The function returns layers of graph ordered topologically
    def _get_layers_from_graph(self):
        graph = {}
        for v in self._graph:
            graph[v] = set([vertex for vertex, _ in self._graph[v].parents])
        layers = list(toposort(graph))
        return layers

    # The function replaces node in constraints when we replace vertex_to_replace with vertex_to_set
    @staticmethod
    def replace_constraints(
            graph: Dict[int, Gate], vertex_to_replace_num: int, vertex_to_set_num: int, equivalence_negation_mark: int
    ):
        vertex_to_replace = graph[vertex_to_replace_num]
        pairs_in_constraints = vertex_to_replace.pairs_in_constraints()
        constraints_to_add = []
        for vertex_to_replace_negation_mark, vertex_in_pair_num, vertex_in_pair_neg_mark in pairs_in_constraints:
            if vertex_in_pair_num == vertex_to_replace_num:
                # remove helps when vertex_to_set_num == vertex_in_pair == vertex_to_replace_num
                graph[vertex_to_replace_num].remove_constraint(
                    vertex_to_replace_negation_mark, vertex_in_pair_num, vertex_in_pair_neg_mark
                )
                constraints_to_add.append((
                    vertex_to_set_num,
                    vertex_to_replace_negation_mark ^ equivalence_negation_mark,
                    vertex_to_set_num,
                    vertex_in_pair_neg_mark ^ equivalence_negation_mark
                ))
            else:
                graph[vertex_in_pair_num].remove_constraint(
                    vertex_in_pair_neg_mark, vertex_to_replace_num, vertex_to_replace_negation_mark
                )
                graph[vertex_to_replace_num].remove_constraint(
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
            graph[v_to_set_value].add_constraint(v_to_set_value_neg_mark, v_in_pair, v_in_pair_neg_mark)

    # The function retrieves data from graph representation needed for aag format
    def _retrieve_data_from_graph(self, vertices_with_out_degrees: Dict[int, int]):
        # The function gets gate representation in aag format depending on its negation mark
        def negate_node_number_in_graph(vertex: int, negation_mark: int):
            if negation_mark == 1:
                return vertex + 1
            return vertex

        # The function add node constraints to all constraints
        def collect_constraints(global_constraints: Set[Tuple[int, int]], node_constraints: List[Tuple[int, int]]):
            for less_node_num, greater_node_num in node_constraints:
                global_constraints.add((less_node_num, greater_node_num))

        def count_out_degree_deltas(vertex: int, new_out_degree: int):
            assert vertex in vertices_with_out_degrees
            vertices_with_out_degrees[vertex] = abs(vertices_with_out_degrees[vertex] - new_out_degree)

        constraints = set()
        layers = self._get_layers_from_graph()

        no_parent_nodes = []
        and_gates = []
        for layer in layers:
            for gate in layer:
                node = self._graph[gate]
                collect_constraints(constraints, node.constraints)
                parents = node.parents
                if vertices_with_out_degrees is not None:
                    count_out_degree_deltas(node.node_number, node.out_degree)
                if node.in_degree == 0:
                    no_parent_nodes.append(gate)
                    continue

                assert node.in_degree == 2

                and_gates.append([
                    gate,
                    negate_node_number_in_graph(*parents[0]),
                    negate_node_number_in_graph(*parents[1])
                ])

        for no_parent_node in no_parent_nodes:
            assert no_parent_node in self._inputs

        return and_gates, no_parent_nodes, list(constraints)

    # the function translates graph to aag format (updates self._aag characteristics)
    def normalize(self, vertices_with_out_degrees: Dict[int, int] = None) -> Tuple[
        List[int], List[int], List[List[int]], List[Tuple[int, int]]
    ]:
        self._normalize_graph()
        and_gates, inputs, constraints = self._retrieve_data_from_graph(vertices_with_out_degrees)
        outputs = []
        for _output, negation_mark in self._outputs:
            outputs.append(_output + negation_mark)
        return inputs, outputs, and_gates, constraints


__all__ = [
    'Normalize',
]
