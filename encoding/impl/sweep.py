from .propagate import Propagation
from .aag import AAG


class Sweep:
    def __init__(self, aag: AAG):
        self._aag = aag
        self._graph = aag.get_graph()
        inputs = aag.get_data().inputs()
        self._inputs = inputs
        self._inputs_to_indices = {input_: {index} for index, input_ in enumerate(inputs)}
        # to replace 'other' output to equivalent
        self._outputs = [(output - output % 2, output % 2) for output in aag.get_data().outputs()]
        self._outputs_to_indices = {}
        for index, output in enumerate([output[0] for output in self._outputs]):
            if output in self._outputs_to_indices:
                self._outputs_to_indices[output].add(index)
            else:
                self._outputs_to_indices[output] = {index}

    # This function detects which vertices are zeroes according to edges (which lead to one parent)
    def _propagate_zeros(self) -> AAG:
        values_to_assign_zero = []
        for vertex in self._graph:
            node = self._graph[vertex]
            if node.in_degree == 0:
                continue
            assert node.in_degree == 2
            if node.parents[1][0] == node.parents[0][0]:
                assert node.parents[1][1] != node.parents[0][1]
                values_to_assign_zero.append(node.node_number)

        return Propagation(
            aag=self._aag,
            constraints=set(),
            conflicts_limit=100
        ).propagate_several(list(zip(values_to_assign_zero, [0] * len(values_to_assign_zero))))

    # This function removes extra vertices from graph
    def sweep(self) -> AAG:
        return self._propagate_zeros()


__all__ = [
    'Sweep'
]
