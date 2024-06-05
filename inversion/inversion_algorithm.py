from typing import Tuple, List, Optional
from algorithm import CubeAndConquerProcess, CubeAndConquerEstimateMPI
from encoding import AAG
from util import WorkPath, Logger, Solvers, string_to_bits
from algorithm import BranchingHeuristic, HaltingHeuristic, TransformationHeuristic, OutputHandler, CncReport
import os
from fraig import Fraig, AndInverterGraph
from time import time


class InversionAlgorithm:
    def __init__(
            self,
            aag: AAG,
            output_bits_str: str,
            work_path: WorkPath,
            branching_heuristic: BranchingHeuristic,
            halt_heuristic: HaltingHeuristic,
            transformation_heuristics: TransformationHeuristic,
            executor_workers: int,
            worker_solver: Solvers,
            vertices_to_propagate: List[Tuple[int, int]],
            task_type: Optional[str] = None
    ):
        self._output_bits = string_to_bits(output_bits_str)
        self._check_bits = {
            input_: value for input_, value in zip(aag.get_data().outputs(), self._output_bits)
        }
        self._inputs = aag.get_data().inputs()

        logger = Logger(out_path=work_path)

        self._work_path = work_path

        with logger as logger:
            description = {
                'cpu number': os.cpu_count(),
                'executor workers': executor_workers,
                'branching heuristic': branching_heuristic.description(),
                'transformation heuristics': transformation_heuristics.description(),
                'halt heuristic': halt_heuristic.description(),
                'inversion description': os.path.basename(aag.filepath)
            }
            if task_type:
                description.update({
                    'task type': task_type
                })
            logger.format(description, 'description.jsonl')

        start_cube_time = time()

        self._start_cube_time = start_cube_time

        self._initial_aag = aag

        fraig = Fraig(
            aag=aag,
            normalization_output_handler=OutputHandler(False)
        ).propagate(tuple(vertices_to_propagate)).reduce(
            reduction_conflicts=5000,
        )

        self._fraig = fraig
        self._transformation_heuristics = transformation_heuristics
        self._halt_heuristic = halt_heuristic
        self._branching_heuristic = branching_heuristic
        self._worker_solver = worker_solver
        self._logger = logger

        self._cnc_process = CubeAndConquerProcess(
            fraig=fraig,
            start_cube_time=start_cube_time,
            work_path=work_path,
            branching_heuristic=branching_heuristic,
            halt_heuristic=halt_heuristic,
            transformation_heuristics=transformation_heuristics,
            executor_workers=executor_workers,
            worker_solver=worker_solver,
            logger=logger,
            append_output=False
        )

    def _check_satisfying_assignment(self, cnc_report: CncReport):
        if cnc_report.satisfying_assignment:
            satisfying_assignment = cnc_report.satisfying_assignment
            aig = AndInverterGraph(aag=self._initial_aag)

            outputs_generated = aig.generate_outputs_given_inputs(
                {k: v for k, v in zip(self._inputs, string_to_bits(satisfying_assignment))}
            )

            assert outputs_generated == self._check_bits

    def cnc(self) -> CncReport:
        cnc_report = self._cnc_process.cnc()
        self._check_satisfying_assignment(cnc_report)
        return cnc_report

    def cnc_estimate(
            self,
            use_input_decomposition: bool,
            decomposition_limit: int,
            length_decompose: str,
            shuffle_inputs: bool
    ) -> CncReport:
        cnc_report = CubeAndConquerEstimateMPI(
            fraig=self._fraig,
            start_cube_time=self._start_cube_time,
            work_path=self._work_path,
            branching_heuristic=self._branching_heuristic,
            halt_heuristic=self._halt_heuristic,
            transformation_heuristics=self._transformation_heuristics,
            worker_solver=self._worker_solver,
            logger=self._logger,
            append_output=False,
            use_input_decomposition=use_input_decomposition,
            decomposition_limit=decomposition_limit,
            length_decompose=length_decompose,
            shuffle_inputs=shuffle_inputs
        ).cube_and_conquer()
        self._check_satisfying_assignment(cnc_report)
        return cnc_report


__all__ = [
    'InversionAlgorithm'
]
