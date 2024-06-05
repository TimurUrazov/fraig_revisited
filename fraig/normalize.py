from .graph import AndInverterGraph, Gate
from typing import Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from algorithm import OutputHandler


class Normalize:
    def __init__(
            self,
            and_inverter_graph: AndInverterGraph,
            output_handler: 'OutputHandler'
    ):
        self._graph = and_inverter_graph
        self._output_handler = output_handler

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

            for negation_mark in range(1, -1, -1):
                for child_num, child_neg in children_buckets[node_parents_negation_mark ^ negation_mark]:
                    graph[initial_node_number].add_child(child_num, child_neg ^ negation_mark)
                    graph[child_num].add_parent(initial_node_number, child_neg ^ negation_mark)

            for negation_mark in range(1, -1, -1):
                for literal in literal_buckets[node_parents_negation_mark ^ negation_mark]:
                    visited_vertices.add(literal)
                    graph.update_inputs_or_outputs(literal, initial_node_number, negation_mark)
                    graph.replace_constraints(literal, initial_node_number, negation_mark)
                    if literal != initial_node_number:
                        del graph[literal]

            for node_n, _ in node.parents:
                normalize_graph_impl(node_n)

        # We go layer by layer not to turn into situation in which we got one parent vertex because
        # lower layers we processed later
        for layer in graph.get_layers():
            for vertex in layer:
                normalize_graph_impl(vertex)

    # The function translates graph to aag format (updates self._aag characteristics)
    def normalize(self):
        self._normalize_graph()
        self._graph.update_outputs(self._output_handler.handle(self._graph.outputs))


__all__ = [
    'Normalize'
]
