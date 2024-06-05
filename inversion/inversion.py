from .inversion_algorithm import InversionAlgorithm
from encoding import AAG
from util import WorkPath, Solvers, string_to_bits
from algorithm import BranchingHeuristic, HaltingHeuristic, TransformationHeuristic


class Inversion(InversionAlgorithm):
    def __init__(
            self,
            aag: AAG,
            output_bits_str: str,
            work_path: WorkPath,
            branching_heuristic: BranchingHeuristic,
            halt_heuristic: HaltingHeuristic,
            transformation_heuristics: TransformationHeuristic,
            executor_workers: int,
            worker_solver: Solvers
    ):
        outputs = aag.get_data().outputs()
        outputs_to_propagate = [(
            outputs[index] - outputs[index] % 2, output_bit ^ (outputs[index] % 2)
        ) for index, output_bit in enumerate(string_to_bits(output_bits_str))]

        super().__init__(
            aag=aag,
            output_bits_str=output_bits_str,
            work_path=work_path,
            branching_heuristic=branching_heuristic,
            halt_heuristic=halt_heuristic,
            transformation_heuristics=transformation_heuristics,
            executor_workers=executor_workers,
            worker_solver=worker_solver,
            vertices_to_propagate=outputs_to_propagate
        )


__all__ = [
    'Inversion'
]
