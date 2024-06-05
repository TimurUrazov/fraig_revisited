import time

from util import WorkPath, Solvers, Logger, Report, MultiprocessingLog, solve_cnf_source, Status, analyze_result
from scheme import AAGLEC
from exception import SatException
from algorithm import (BranchingHeuristic, HaltingHeuristic, TransformationHeuristic, OutputHandler,
                       CubeAndConquerProcess, CubeAndConquerEstimateMPI, CncReport, ProcessStatistics,
                       ProcessesCommonStatistics)
from fraig import Fraig
from typing import Dict, Optional, List
from mpi4py import MPI
from encoding import CNF
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_EXCEPTION


class LogicalEquivalenceChecking:
    def __init__(
            self,
            left_scheme_file: str,
            right_scheme_file: str,
            work_path: WorkPath,
            branching_heuristic: BranchingHeuristic,
            halt_heuristic: HaltingHeuristic,
            transformation_heuristics: TransformationHeuristic,
            worker_solver: Solvers
    ):

        self._miter_scheme = AAGLEC(
            left_scheme_from_file=left_scheme_file,
            right_scheme_from_file=right_scheme_file
        )

        self._work_path = work_path

        self._branching_heuristic = branching_heuristic
        self._halt_heuristic = halt_heuristic
        self._transformation_heuristics = transformation_heuristics

        self._worker_solver = worker_solver

        self._logger = Logger(out_path=self._work_path)

        self.logs = {
            'branching_heuristic': self._branching_heuristic.description(),
            'transformation_heuristics': self._transformation_heuristics.description(),
            'halt_heuristic': self._halt_heuristic.description(),
            'lec_description': self._miter_scheme.description(),
            'worker_solver': self._worker_solver.name,
        }

    def _log(self, logs: Dict):
        self.logs.update(logs)
        with self._logger as logger:
            logger.format(self.logs, 'description.jsonl')

    def solve_using_processes(self, number_of_executor_workers: int) -> CncReport:
        self._log({
            'num_of_executor_processes': number_of_executor_workers,
            'calculation/estimate': 'calculation',
        })
        output_handler = OutputHandler(True)

        start_time = time.time()
        a = self._miter_scheme.get_scheme_till_xor()

        fraig = Fraig(
            aag=a,
            normalization_output_handler=output_handler
        )

        return CubeAndConquerProcess(
            fraig=fraig,
            start_cube_time=start_time,
            work_path=self._work_path,
            branching_heuristic=self._branching_heuristic,
            halt_heuristic=self._halt_heuristic,
            transformation_heuristics=self._transformation_heuristics,
            executor_workers=number_of_executor_workers,
            worker_solver=self._worker_solver,
            logger=self._logger,
            append_output=output_handler.append_outputs
        ).cnc()

    def estimate_using_mpi(
            self,
            use_input_decomposition: bool,
            decomposition_limit: int,
            length_decompose: str,
            shuffle_inputs: bool,
            depth: Optional[int] = None
    ) -> Report:
        if MPI.COMM_WORLD.Get_rank() == 0:
            log = {
                'calculation/estimate': 'estimate',
                'num_of_executor_processes': MPI.COMM_WORLD.size,
                'decomposition_limit': decomposition_limit,
                'length_decompose': length_decompose,
                'shuffle_inputs': shuffle_inputs,
            }
            if depth:
                log.update({
                    'depth': depth,
                })
            self._log(log)

        halt_heuristic = self._halt_heuristic

        output_handler = OutputHandler(True)

        start_time = time.time()

        fraig = Fraig(
            aag=self._miter_scheme.get_scheme_till_xor(),
            normalization_output_handler=output_handler
        )

        if depth:
            halt_heuristic = HaltingHeuristic(
                max_depth=depth,
                reduction_conflicts=300
            )

        return CubeAndConquerEstimateMPI(
            fraig=fraig,
            start_cube_time=start_time,
            work_path=self._work_path,
            branching_heuristic=self._branching_heuristic,
            halt_heuristic=halt_heuristic,
            transformation_heuristics=self._transformation_heuristics,
            worker_solver=self._worker_solver,
            logger=self._logger,
            append_output=output_handler.append_outputs,
            use_input_decomposition=use_input_decomposition,
            decomposition_limit=decomposition_limit,
            length_decompose=length_decompose,
            shuffle_inputs=shuffle_inputs
        ).cube_and_conquer()

    def _init_pool_processes(
            self,
            logger_,
            worker_solver_,
            worker_stats_queue_,
            worker_stats_,
            cnf_
    ):
        global logger
        global worker_solver
        global worker_stats
        global cnf

        logger = logger_
        worker_solver = worker_solver_
        worker_stats = worker_stats_[worker_stats_queue_.get()]
        cnf = cnf_

    def _worker_fn(self, cube: List[int]):
        worker_start_time = time.time()
        report = analyze_result(solve_cnf_source(
            worker_solver, cnf.get_data().source(supplements=(cube, []))
        ), print_output=False)
        logger.emit({
            'status': report.status.name,
            'conflicts': report.conflicts,
            'solvetime': report.process_time
        })
        if report.status == Status.UNSAT:
            worker_stats.update_unsat_statistics(report.conflicts, report.process_time)
            worker_stats.update_time(time.time() - worker_start_time)
        if report.status == Status.SAT:
            worker_stats.update_sat_statistics(report.conflicts, report.process_time)
            worker_stats.update_time(time.time() - worker_start_time)

    def solve_march_cubes(self, path_to_cnf_file: str, path_to_cubes_file: str, number_of_executor_workers: int):
        self._log({
            'num_of_executor_processes': number_of_executor_workers,
            'calculation/estimate': 'calculation',
            'path_to_cnf_file': path_to_cnf_file,
            'path_to_cubes_file': path_to_cubes_file
        })
        start_algorithm_time = time.time()
        multiprocessing_log = MultiprocessingLog(self._logger, 'cube.jsonl')

        worker_stats_queue = multiprocessing.Queue()

        worker_stats = [ProcessStatistics() for _ in range(number_of_executor_workers - 1)]

        [worker_stats_queue.put(i) for i in range(number_of_executor_workers - 1)]

        initial_cnf = CNF(from_file=path_to_cnf_file)

        executor = ProcessPoolExecutor(
            max_workers=number_of_executor_workers - 1,
            initializer=self._init_pool_processes,
            initargs=(
                multiprocessing_log,
                self._worker_solver,
                worker_stats_queue,
                worker_stats,
                initial_cnf
            )
        )

        cubes = []

        with open(path_to_cubes_file, 'r') as file_handle:
            lines = file_handle.readlines()
            for line in lines:
                cubes.append(list(map(int, line.split('a')[1].split()[:-1])))

        futures = []
        with executor:
            for cube in cubes:
                futures.append(executor.submit(self._worker_fn, cube=cube))

        report = CncReport(Status.UNSAT)

        done, _ = wait(futures, return_when=FIRST_EXCEPTION)
        for completed_on_error in done:
            try:
                completed_on_error.result()
            except SatException as sat_exception:
                report = CncReport(Status.SAT, sat_exception.satisfying_assignment)
        executor.shutdown(wait=False, cancel_futures=True)

        processes_common_statistics = ProcessesCommonStatistics(worker_stats)

        self._logger.format({
            'sat solutions': processes_common_statistics.sat_solutions,
            'sat conflicts': processes_common_statistics.sat_conflicts,
            'sat solvetime': processes_common_statistics.sat_cpu_time,
            'unsat solvetime': processes_common_statistics.unsat_cpu_time,
            'unsat solutions': processes_common_statistics.unsat_solutions,
            'unsat conflicts': processes_common_statistics.unsat_conflicts,
            'cpu time': processes_common_statistics.cpu_time,
            'average cpu time': processes_common_statistics.average_cpu_time,
            'min cpu time': processes_common_statistics.min_cpu_time,
            'max cpu time': processes_common_statistics.max_cpu_time,
            'cpu time standard deviation': processes_common_statistics.cpu_time_standard_deviation,
            'conflicts': processes_common_statistics.conflicts,
            'average conflicts': processes_common_statistics.average_conflicts,
            'min conflicts': processes_common_statistics.min_conflicts,
            'max conflicts': processes_common_statistics.max_conflicts,
            'conflicts standard deviation': processes_common_statistics.conflicts_standard_deviation,
            'algorithm time': time.time() - start_algorithm_time,
            'number of errors': processes_common_statistics.errors,
            'number of refutations': processes_common_statistics.refutations,
            'number of controversial assignments': processes_common_statistics.controversial_assignments,
            'number of constraint violations': processes_common_statistics.constraint_violations,
            'sat status': report.status.name + ' ' + report.satisfying_assignment if report.status == Status.SAT else report.status.name
        }, 'cube_statistics.jsonl')


__all__ = [
    'LogicalEquivalenceChecking'
]
