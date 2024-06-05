from .propagate import Propagation
from .graph import AndInverterGraph


class Sweep:
    def __init__(self, and_inverter_graph: AndInverterGraph):
        self._graph = and_inverter_graph

    # This function detects vertices:
    # which are zeroes according to edges (which lead to one parent)

    def _propagate_zeros(self):
        vertices_and_values_to_assign = []
        for vertex in self._graph:
            node = self._graph[vertex]
            if node.in_degree == 0:
                continue
            assert node.in_degree == 2
            if node.parents[1][0] == node.parents[0][0]:
                assert node.parents[1][1] != node.parents[0][1]
                vertices_and_values_to_assign.append((node.node_number, 0))

        return Propagation(
            and_inverter_graph=self._graph
        ).propagate(tuple(vertices_and_values_to_assign))

    # This function removes extra vertices from graph
    def sweep(self):
        return self._propagate_zeros()


__all__ = [
    'Sweep'
]
