from encoding import Encoding, AAG, Propagation
from copy import copy
from aig_substitutions.exceptions import ConflictAssignment
import operator


class BranchingHeuristics:
    def branch(self, encoding: Encoding) -> int:
        raise NotImplementedError


class DegreeBranchingHeuristics(BranchingHeuristics):
    def branch(self, aag: AAG) -> int:
        pass


class LookaheadBranchingHeuristics:
    def branch(self, aag: AAG):
        vertices = aag.get_data().vertices()
        vertex_metrics_zero = {}
        vertex_metrics_one = {}
        for vertex in vertices:
            for assignment_value in [0, 1]:
                prop = Propagation(
                    aag=copy(aag),
                    constraints=set(),
                    conflicts_limit=100
                )
                try:
                    prop.propagate(vertex, assignment_value)
                except ConflictAssignment:
                    pass
                except Exception as e:
                    print("vertex", vertex)
                    print("source", aag.get_data().source())
                    print("constraints", aag.get_data().constraints())
                    raise e
                out_degree_deltas_sum = prop.get_out_degree_deltas_sum()
                if assignment_value == 0:
                    vertex_metrics_zero[vertex] = out_degree_deltas_sum
                else:
                    vertex_metrics_one[vertex] = out_degree_deltas_sum

        s = []

        max_vertex = -1
        max_metrics = -1
        max_metrics_addition = -1
        for vertex in vertices:
            s.append((
                vertex_metrics_zero[vertex] * vertex_metrics_one[vertex],
                vertex_metrics_zero[vertex] + vertex_metrics_one[vertex],
                vertex,
                vertex_metrics_zero[vertex],
                vertex_metrics_one[vertex]
            ))
            metrics = vertex_metrics_zero[vertex] * vertex_metrics_one[vertex]
            if metrics > max_metrics:
                max_vertex = vertex
                max_metrics = metrics
                if max_metrics_addition == -1:
                    max_metrics_addition = vertex_metrics_zero[vertex] + vertex_metrics_one[vertex]
            elif max_metrics == metrics:
                metrics_add = vertex_metrics_zero[vertex] + vertex_metrics_one[vertex]
                if max_metrics_addition < metrics_add:
                    max_metrics_addition = metrics_add
                    max_metrics = metrics

        return max_vertex
