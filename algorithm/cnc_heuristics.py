from fraig import Fraig
import operator
from .cnc_tree import CncTree
from typing import List, Tuple, Optional
from encoding import AAG
from .output_handler import OutputHandler
from exception import ConflictAssignmentException, UnsatException
import random


class BranchingHeuristic:
    _slug = 'BranchingHeuristic'

    def __init__(self, chunk_size: Optional[int] = None):
        self._chunk_size = chunk_size

    def description(self) -> dict:
        lines = {'heuristics': self._slug}
        if self._chunk_size:
            lines['chunk_size'] = self._chunk_size
        return lines

    def branching_choice(
            self,
            aag: AAG,
            vertices_for_choice: List[int],
            include_output: bool,
            size_of_thread_branching_cache: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        raise NotImplementedError


class DegreeBranchingHeuristic(BranchingHeuristic):
    _slug = 'DegreeBranchingHeuristic'

    def __init__(self, chunk_size: Optional[int] = None):
        super().__init__(chunk_size)

    def description(self) -> dict:
        return super().description()

    def branching_choice(
            self,
            aag: AAG,
            vertices_for_choice: List[int],
            include_output: bool,  # we do not need this argument
            size_of_thread_branching_cache: Optional[int] = None,  # we do not need this argument
    ) -> List[Tuple[int, int]]:
        functionally_reduced_aig = Fraig(
            aag=aag,
            normalization_output_handler=OutputHandler(False)
        )
        vertices_for_choice = set(vertices_for_choice)
        out_degrees = {
            k: v for k, v in functionally_reduced_aig.graph_info().out_degrees().items() if k in vertices_for_choice
        }
        if self._chunk_size:
            return list(reversed(sorted(out_degrees.items(), key=operator.itemgetter(1))))[:self._chunk_size]

        return list(reversed(sorted(out_degrees.items(), key=operator.itemgetter(1))))


class LookAheadBranchingHeuristic(BranchingHeuristic):
    pass


class IntegralDegreeLookAheadBranchingHeuristic(LookAheadBranchingHeuristic):
    _slug = 'IntegralDegreeLookAheadBranchingHeuristic'

    def __init__(self, chunk_size: Optional[int] = None):
        super().__init__(chunk_size)

    def description(self) -> dict:
        return super().description()

    @staticmethod
    def _count_degrees_integral_delta(out_degrees_before_propagation, out_degrees_after_propagation) -> int:
        integral_value = 0
        for vertex_before_propagation, value_before_propagation in out_degrees_before_propagation.items():
            value_after_propagation = 0
            if vertex_before_propagation in out_degrees_after_propagation:
                value_after_propagation = out_degrees_after_propagation[vertex_before_propagation]
            # Degree of vertex can either grow or get lower
            integral_value += abs(value_before_propagation - value_after_propagation)
        return integral_value

    def branching_choice(
            self,
            aag: AAG,
            vertices_for_choice: List[int],
            include_output: bool,
            size_of_thread_branching_cache: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        vertex_metrics_zero = {}
        vertex_metrics_one = {}
        if self._chunk_size:
            vertices_for_choice = random.sample(
                vertices_for_choice, min(self._chunk_size, len(vertices_for_choice))
            )
        for vertex_to_assign in vertices_for_choice:
            for value_to_assign in range(2):
                fraig_before_propagation = Fraig(
                    aag=aag, normalization_output_handler=OutputHandler(include_output)
                )
                out_degrees_before_propagation = fraig_before_propagation.graph_info().out_degrees()
                try:
                    out_degrees_after_propagation = fraig_before_propagation.propagate_single(
                        vertex_to_assign, value_to_assign
                    ).sweep().graph_info().out_degrees()
                except ConflictAssignmentException:
                    out_degrees_after_propagation = {}
                except UnsatException:
                    out_degrees_after_propagation = {}
                except Exception as e:
                    print("vertex_to_assign", vertex_to_assign)
                    print("source", aag.get_data().source())
                    print("constraints", aag.get_data().constraints)
                    raise e

                vertex_metrics = (vertex_metrics_zero if value_to_assign == 0 else vertex_metrics_one)
                vertex_metrics[
                    vertex_to_assign
                ] = IntegralDegreeLookAheadBranchingHeuristic._count_degrees_integral_delta(
                    out_degrees_before_propagation,
                    out_degrees_after_propagation
                )

        metrics = []

        for vertex_to_assign in vertices_for_choice:
            metrics.append((
                vertex_metrics_zero[vertex_to_assign] * vertex_metrics_one[vertex_to_assign],
                vertex_metrics_zero[vertex_to_assign] + vertex_metrics_one[vertex_to_assign],
                vertex_to_assign
            ))

        vertice_and_metric_to_propagate = list(reversed([
            (metric[2], metric[0]) for metric in sorted(metrics, key=operator.itemgetter(0, 1))
        ]))

        if size_of_thread_branching_cache:
            vertice_and_metric_to_propagate = vertice_and_metric_to_propagate[:size_of_thread_branching_cache]

        return vertice_and_metric_to_propagate


class HaltingHeuristic:
    _slug = 'HaltingHeuristic'

    def __init__(self, max_depth: int = None, reduction_conflicts: int = None):
        self._max_depth = max_depth
        self._reduction_conflicts = reduction_conflicts

    def description(self) -> dict:
        lines = {'heuristics': self._slug}
        if self._max_depth:
            lines['max_depth'] = self._max_depth
        return lines

    @property
    def max_depth(self):
        return self._max_depth

    def ensure_halt(self, cnc_tree: CncTree, cnc_tree_vertex_num: int, aag: AAG) -> bool:
        return self._max_depth and cnc_tree[cnc_tree_vertex_num].depth >= self._max_depth

    def halt_transformation(
            self,
            aag: AAG,
            cnc_tree: CncTree,
            fraig_normalization_output_handler: OutputHandler,
    ) -> AAG:
        return Fraig(
            aag=aag, normalization_output_handler=fraig_normalization_output_handler
        ).reduce(reduction_conflicts=self._reduction_conflicts).sweep().to_aag()


class OutDegreeHaltingHeuristic(HaltingHeuristic):
    _slug = 'OutDegreeHaltingHeuristic'

    def __init__(self, out_degree_upper_bound: int, max_depth: int = None, reduction_conflicts: int = None):
        super().__init__(max_depth, reduction_conflicts)
        self._out_degree_upper_bound = out_degree_upper_bound

    def description(self) -> dict:
        lines = super().description()
        lines['out_degree_upper_bound'] = self._out_degree_upper_bound
        return lines

    def ensure_halt(self, cnc_tree: CncTree, cnc_tree_vertex_num: int, aag: AAG) -> bool:
        if super().ensure_halt(cnc_tree, cnc_tree_vertex_num, aag):
            return True
        out_degrees = Fraig(
            aag=aag,
            normalization_output_handler=OutputHandler(False)
        ).graph_info().out_degrees()
        return max(out_degrees.items(), key=operator.itemgetter(1))[1] < self._out_degree_upper_bound


class LeafSizeHaltingHeuristic(HaltingHeuristic):
    _slug = 'LeafSizeHaltingHeuristic'

    def __init__(self, leafs_size_lower_bound: int, max_depth: int = None, reduction_conflicts: int = None):
        super().__init__(max_depth, reduction_conflicts)
        self._leafs_size_lower_bound = leafs_size_lower_bound

    def description(self) -> dict:
        lines = super().description()
        lines['leafs_size_lower_bound'] = self._leafs_size_lower_bound
        return lines

    def ensure_halt(self, cnc_tree: CncTree, cnc_tree_vertex_num: int, aag: AAG) -> bool:
        depth_halt = super().ensure_halt(cnc_tree, cnc_tree_vertex_num, aag)
        return depth_halt or aag.get_data().n_of_vertices < self._leafs_size_lower_bound


class LeafsPercentageHaltingHeuristic(HaltingHeuristic):
    _slug = 'LeafsPercentageHaltingHeuristic'

    def __init__(self, leafs_percentage_upper_bound: int, max_depth: int = None, reduction_conflicts: int = None):
        super().__init__(max_depth, reduction_conflicts)
        self._leafs_percentage_upper_bound = leafs_percentage_upper_bound

    def description(self) -> dict:
        lines = super().description()
        lines['leafs_percentage_upper_bound'] = self._leafs_percentage_upper_bound
        return lines

    def ensure_halt(self, cnc_tree: CncTree, cnc_tree_vertex_num: int, aag: AAG) -> bool:
        if super().ensure_halt(cnc_tree, cnc_tree_vertex_num, aag):
            return True
        return cnc_tree.solved_cubes_leafs_num // cnc_tree.leafs_num > self._leafs_percentage_upper_bound


class LeafsSizeAndPercentageHaltingHeuristic(HaltingHeuristic):
    _slug = 'LeafsSizeAndPercentageHaltingHeuristic'

    def __init__(self, leafs_percentage_upper_bound: int, max_size: int, max_depth: int = None, reduction_conflicts: int = None):
        super().__init__(max_depth, reduction_conflicts)
        self._leafs_percentage_upper_bound = leafs_percentage_upper_bound
        self._leafs_size_lower_bound = max_size

    def description(self) -> dict:
        lines = super().description()
        lines['leafs_percentage_upper_bound'] = self._leafs_percentage_upper_bound
        lines['leafs_size_lower_bound'] = self._leafs_size_lower_bound
        return lines

    def ensure_halt(self, cnc_tree: CncTree, cnc_tree_vertex_num: int, aag: AAG) -> bool:
        if (super().ensure_halt(cnc_tree, cnc_tree_vertex_num, aag)
                or aag.get_data().n_of_vertices < self._leafs_size_lower_bound):
            return True
        return cnc_tree.solved_cubes_leafs_num * 100 // cnc_tree.leafs_num > self._leafs_percentage_upper_bound


class TransformationHeuristic:
    _slug = 'TransformationHeuristic'

    def single_transformation(
            self,
            cnc_tree: CncTree,
            graph_num: int,
            aag: AAG,
            vertex_to_assign: int,
            value_to_assign: int,
            fraig_normalization_output_handler: OutputHandler
    ) -> Fraig:
        raise NotImplementedError

    def description(self) -> dict:
        return {
            'transformation_heuristics': self._slug
        }

    def transformation(
            self,
            cnc_tree: CncTree,
            graph_num: int,
            aag: AAG,
            vertices_and_values_to_assign: List[Tuple[int, int]],
            fraig_normalization_output_handler: OutputHandler,
            reduce: bool = False
    ) -> Fraig:
        raise NotImplementedError


class PropagationTransformationHeuristic(TransformationHeuristic):
    _slug = 'PropagationTransformationHeuristic'

    def single_transformation(
            self,
            cnc_tree: CncTree,
            graph_num: int,
            aag: AAG,
            vertex_to_assign: int,
            value_to_assign: int,
            fraig_normalization_output_handler: OutputHandler
    ) -> Fraig:
        return Fraig(
            aag=aag, normalization_output_handler=fraig_normalization_output_handler
        ).propagate_single(vertex_to_assign, value_to_assign).sweep()

    def transformation(
            self,
            cnc_tree: CncTree,
            graph_num: int,
            aag: AAG,
            vertices_and_values_to_assign: List[Tuple[int, int]],
            fraig_normalization_output_handler: OutputHandler,
            reduce: bool = False
    ) -> Fraig:
        return Fraig(
            aag=aag, normalization_output_handler=fraig_normalization_output_handler
        ).propagate(tuple(vertices_and_values_to_assign)).sweep()

    def description(self) -> dict:
        return super().description()


class FraigTransformationHeuristic(TransformationHeuristic):
    def __init__(
            self,
            fraig_number_of_simulation_samples: int = 64 * 127,
            fraig_reduction_conflicts: int = 100,
            fraig_include_complementary_in_reduction: bool = False
    ):
        self._fraig_number_of_simulation_samples = fraig_number_of_simulation_samples
        self._fraig_reduction_conflicts = fraig_reduction_conflicts
        self._fraig_include_complementary_in_reduction = fraig_include_complementary_in_reduction

    def description(self) -> dict:
        lines = super().description()
        lines['fraig_number_of_simulation_samples'] = self._fraig_number_of_simulation_samples
        lines['fraig_reduction_conflicts'] = self._fraig_reduction_conflicts
        lines['fraig_include_complementary_in_reduction'] = self._fraig_include_complementary_in_reduction
        return lines

    def single_transformation(
            self,
            cnc_tree: CncTree,
            graph_num: int,
            aag: AAG,
            vertex_to_assign: int,
            value_to_assign: int,
            fraig_normalization_output_handler: OutputHandler
    ) -> Fraig:
        return Fraig(
            aag=aag,
            normalization_output_handler=fraig_normalization_output_handler,
            number_of_simulation_samples=self._fraig_number_of_simulation_samples,
            reduction_conflicts=self._fraig_reduction_conflicts,
            include_complementary_in_reduction=self._fraig_include_complementary_in_reduction
        ).propagate_single(
            vertex_to_assign, value_to_assign
        ).sweep().reduce().sweep()

    def transformation(
            self,
            cnc_tree: CncTree,
            graph_num: int,
            aag: AAG,
            vertices_and_values_to_assign: List[Tuple[int, int]],
            fraig_normalization_output_handler: OutputHandler,
            reduce: bool = False
    ) -> Fraig:
        fraig = Fraig(
            aag=aag,
            normalization_output_handler=fraig_normalization_output_handler,
            number_of_simulation_samples=self._fraig_number_of_simulation_samples,
            reduction_conflicts=self._fraig_reduction_conflicts,
            include_complementary_in_reduction=self._fraig_include_complementary_in_reduction
        ).propagate(
            tuple(vertices_and_values_to_assign)
        ).sweep()
        if reduce:
            fraig = fraig.reduce(reduction_conflicts=self._fraig_reduction_conflicts).sweep()
        return fraig


class FraigSizeLevelReduceTransformationHeuristic(FraigTransformationHeuristic):
    def __init__(
            self,
            levels_size: List[int],
            fraig_number_of_simulation_samples: int = 64 * 127,
            fraig_reduction_conflicts: int = 100,
            fraig_include_complementary_in_reduction: bool = False
    ):
        super().__init__(
            fraig_number_of_simulation_samples,
            fraig_reduction_conflicts,
            fraig_include_complementary_in_reduction
        )
        self._levels_size = levels_size
        self._pointer = 0

    def description(self) -> dict:
        lines = super().description()
        lines['levels_size'] = self._levels_size
        return lines

    def transformation(
            self,
            aag: AAG,
            cnc_tree: CncTree,
            graph_num: int,
            vertices_and_values_to_assign: List[Tuple[int, int]],
            fraig_normalization_output_handler: OutputHandler,
            reduce: bool = False
    ) -> Fraig:
        fraig = Fraig(
            aag=aag,
            normalization_output_handler=fraig_normalization_output_handler,
            number_of_simulation_samples=self._fraig_number_of_simulation_samples,
            reduction_conflicts=self._fraig_reduction_conflicts,
            include_complementary_in_reduction=self._fraig_include_complementary_in_reduction
        ).propagate(tuple(vertices_and_values_to_assign)).sweep()
        if (self._pointer < len(self._levels_size)
                and fraig.to_aag().get_data().n_of_vertices < self._levels_size[self._pointer]):
            self._pointer += 1
            fraig = fraig.reduce().sweep()
        return fraig


class FraigDepthLevelReduceTransformationHeuristic(FraigTransformationHeuristic):
    def __init__(
            self,
            levels_depth: List[int],
            fraig_number_of_simulation_samples: int = 64 * 127,
            fraig_reduction_conflicts: int = 100,
            fraig_include_complementary_in_reduction: bool = False
    ):
        super().__init__(
            fraig_number_of_simulation_samples,
            fraig_reduction_conflicts,
            fraig_include_complementary_in_reduction
        )
        self._levels_depth = levels_depth
        self._pointer = 0

    def description(self) -> dict:
        lines = super().description()
        lines['levels_size'] = self._levels_depth
        return lines

    def transformation(
            self,
            aag: AAG,
            cnc_tree: CncTree,
            graph_num: int,
            vertices_and_values_to_assign: List[Tuple[int, int]],
            fraig_normalization_output_handler: OutputHandler,
            reduce: bool = False
    ) -> Fraig:
        fraig = Fraig(
            aag=aag,
            normalization_output_handler=fraig_normalization_output_handler,
            number_of_simulation_samples=self._fraig_number_of_simulation_samples,
            reduction_conflicts=self._fraig_reduction_conflicts,
            include_complementary_in_reduction=self._fraig_include_complementary_in_reduction
        ).propagate(tuple(vertices_and_values_to_assign)).sweep()
        if (self._pointer < len(self._levels_depth)
                and cnc_tree[graph_num].depth >= self._levels_depth[self._pointer]):
            self._pointer += 1
            fraig = fraig.reduce(reduction_conflicts=self._fraig_reduction_conflicts).sweep()
        return fraig
