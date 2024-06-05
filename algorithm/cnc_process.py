from typing import List, Tuple
from encoding import AAG
from fraig import Fraig
from .output_handler import OutputHandler
from .cnc_heuristics import BranchingHeuristic, HaltingHeuristic, TransformationHeuristic
from .cnc_tree import CncTree, VertexType
from util import Logger, WorkPath, MultiprocessingLog, Status, solve_cnf_lec_result, SatParser, Solvers
from exception import (UnsatException, SatException, ConflictAssignmentException, ControversialAssignmentException,
                       ConstraintViolationAssignmentException)
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_EXCEPTION
from multiprocessing import Event
import multiprocessing
from time import time
from .statistics import ProcessStatistics, ProcessesCommonStatistics
from .cnc_report import CncReport


def init_pool_processes(
        logger_,
        initial_current_inputs_to_initial_,
        initial_initial_inputs_to_assignments_,
        event_object_,
        halt_heuristic_,
        worker_solver_,
        worker_stats_queue_,
        worker_stats_
):
    global event_object
    global initial_current_inputs_to_initial
    global initial_initial_inputs_to_assignments
    global logger
    global halt_heuristic
    global worker_solver
    global worker_stats

    event_object = event_object_
    initial_current_inputs_to_initial = initial_current_inputs_to_initial_
    initial_initial_inputs_to_assignments = initial_initial_inputs_to_assignments_
    logger = logger_
    halt_heuristic = halt_heuristic_
    worker_solver = worker_solver_
    worker_stats = worker_stats_[worker_stats_queue_.get()]


def worker_fn_process(
        cube_filename: str,
        aag: AAG,
        initial_input_to_assignment,
        current_input_to_initial,
        append_outputs: bool
):
    worker_start_time = time()
    try:
        handler = OutputHandler(append_outputs)
        aag: AAG = halt_heuristic.halt_transformation(aag, None, handler)

        new_cnf, aag_to_cnf_inputs_vars = aag.to_cnf_with_mapping()

        result = solve_cnf_lec_result(new_cnf, worker_solver, handler.append_outputs)

        satisfying_assignment = SatParser(result)

        solvetime, conflicts, satisfying_assignment_cnf_dict, answer = satisfying_assignment.parse_complete()

        logger.emit({
            'status': answer.value,
            'conflicts': conflicts,
            'solvetime': solvetime,
            'append output': append_outputs,
            'aag size': aag.get_data().n_of_vertices,
            'solved before': worker_stats.solved_tasks,
            'conflicts or errors before': worker_stats.conflicts_or_errors,
            'filename': cube_filename
        })

        if answer == Status.UNSAT:
            worker_stats.update_unsat_statistics(conflicts, solvetime)
            worker_stats.update_time(time() - worker_start_time)
            return

        worker_stats.update_sat_statistics(conflicts, solvetime)

        satisfying_assignment_aag_input_dict = {}

        for aag_var, cnf_var in aag_to_cnf_inputs_vars.items():
            satisfying_assignment_aag_input_dict[
                initial_current_inputs_to_initial[current_input_to_initial[aag_var]]
            ] = satisfying_assignment_cnf_dict[cnf_var]

        for aag_var, value in initial_input_to_assignment.items():
            satisfying_assignment_aag_input_dict[initial_current_inputs_to_initial[aag_var]] = value
        for aag_var, value in initial_initial_inputs_to_assignments.items():
            satisfying_assignment_aag_input_dict[aag_var] = value

        sorted_list = sorted(satisfying_assignment_aag_input_dict.items())

        sorted_input_list = [v for _, v in sorted_list]

        sorted_input_in_str = ''.join(list(map(str, sorted_input_list)))

        logger.emit({
            'comment': 'satisfying assignment',
            'satisfying_assignment_aag_input_dict': satisfying_assignment_aag_input_dict,
            'sorted input in str': sorted_input_in_str
        })

        worker_stats.update_time(time() - worker_start_time)

        raise SatException(sorted_input_in_str)
    except SatException as sat_exception:
        raise sat_exception
    except UnsatException:
        delta_time = time() - worker_start_time
        worker_stats.update_unsat_statistics(1, delta_time)
        worker_stats.update_time(delta_time)
    except ControversialAssignmentException:
        delta_time = time() - worker_start_time
        worker_stats.increase_refutations(delta_time)
        worker_stats.update_time(delta_time)
        worker_stats.increase_controversial_assignment()
    except ConstraintViolationAssignmentException:
        delta_time = time() - worker_start_time
        worker_stats.increase_refutations(delta_time)
        worker_stats.update_time(delta_time)
        worker_stats.increase_constraint_violation()
    except ConflictAssignmentException:
        delta_time = time() - worker_start_time
        worker_stats.increase_refutations(delta_time)
        worker_stats.update_time(delta_time)


class CubeAndConquerProcess:
    def __init__(
            self,
            fraig: Fraig,
            start_cube_time: float,
            work_path: WorkPath,
            branching_heuristic: BranchingHeuristic,
            halt_heuristic: HaltingHeuristic,
            transformation_heuristics: TransformationHeuristic,
            executor_workers: int,
            worker_solver: Solvers,
            logger: Logger,
            append_output: bool
    ):
        self._fraig = fraig

        self._start_cube_time = start_cube_time

        self._logger = logger

        self._branching_heuristic = branching_heuristic
        self._transformation_heuristics = transformation_heuristics
        self._halt_heuristic = halt_heuristic

        self._cnc_tree = CncTree()

        self._cubes_path: WorkPath = work_path.to_path('cubes')

        self._executor_workers = executor_workers

        self._worker_solver = worker_solver

        self._number_of_sat_solutions = 0
        self._number_of_unsat_solutions = 0
        self._number_of_errors = 0

        self._number_of_constraint_violation_errors = 0
        self._number_of_controversial_assignment_errors = 0
        self._number_of_refutations = 0

        self._append_output = append_output

    def cnc(self) -> CncReport:
        event_object_ = Event()

        fraig = self._fraig

        initial_current_inputs_to_initial_ = fraig.graph_info().current_input_to_initial
        initial_initial_inputs_to_assignments_ = fraig.graph_info().initial_input_to_assignment

        aag_ = fraig.to_aag()

        multiprocessing_log = MultiprocessingLog(self._logger, 'cube.jsonl')

        worker_stats_queue = multiprocessing.Queue()

        worker_stats = [ProcessStatistics() for _ in range(self._executor_workers - 1)]

        [worker_stats_queue.put(i) for i in range(self._executor_workers - 1)]

        executor = ProcessPoolExecutor(
            max_workers=self._executor_workers - 1,
            initializer=init_pool_processes,
            initargs=(
                multiprocessing_log,
                initial_current_inputs_to_initial_,
                initial_initial_inputs_to_assignments_,
                event_object_,
                self._halt_heuristic,
                self._worker_solver,
                worker_stats_queue,
                worker_stats
            )
        )

        futures = []

        def dfs(graph_number: int, parent_number: int, stack: List[Tuple[int, int]], append_output: bool):
            value_to_assign = -1
            if graph_number != 0:
                value_to_assign = stack[-1][1]
            try:
                output_handler = OutputHandler(append_output)
                fraig = self._transformation_heuristics.transformation(
                    cnc_tree=self._cnc_tree,
                    aag=aag_,
                    graph_num=graph_number,
                    vertices_and_values_to_assign=stack,
                    fraig_normalization_output_handler=output_handler
                )
                multiprocessing_log.emit({
                    'comment': 'after transformation',
                })
                if graph_number != 0:
                    self._cnc_tree.add_leaf(
                        parent_number, value_to_assign, VertexType.LEAF, output_handler.append_outputs
                    )

                initial_input_to_assignment = fraig.graph_info().initial_input_to_assignment
                current_input_to_initial = fraig.graph_info().current_input_to_initial

                aag = fraig.to_aag()

                if self._halt_heuristic.ensure_halt(self._cnc_tree, graph_number, aag):
                    assert output_handler.append_outputs == self._cnc_tree[graph_number].append_output
                    multiprocessing_log.emit({
                        'comment': 'gave cube to executor',
                        'depth': self._cnc_tree[graph_number].depth,
                    })
                    self._cnc_tree.make_leaf_cube(graph_number)

                    cube_filename = self._cubes_path.to_file(str(graph_number))

                    futures.append(executor.submit(
                        worker_fn_process,
                        cube_filename,
                        aag,
                        initial_input_to_assignment,
                        current_input_to_initial,
                        output_handler.append_outputs
                    ))

                    return

                vertices_and_measures = self._branching_heuristic.branching_choice(
                    aag=aag,
                    vertices_for_choice=aag.get_data().vertices(),
                    include_output=output_handler.append_outputs
                )

                vertices_to_assign = [vertex_and_measure[0] for vertex_and_measure in reversed(
                    sorted(vertices_and_measures, key=lambda v_and_m: v_and_m[1])
                )]

                vertex_to_assign = vertices_to_assign[0]

                if not event_object_.is_set():
                    for value_to_assign in range(1, -1, -1):
                        stack.append((vertex_to_assign, value_to_assign))
                        child_graph_num = CncTree.count_child_num(graph_number, value_to_assign)
                        dfs(child_graph_num, graph_number, stack, output_handler.append_outputs)
                        stack.pop()
            except ControversialAssignmentException:
                self._number_of_refutations += 1
                self._number_of_controversial_assignment_errors += 1
                if graph_number != 0:
                    self._cnc_tree.add_leaf(
                        parent_number, value_to_assign, VertexType.REFUTATION_LEAF, append_output
                    )
            except ConstraintViolationAssignmentException:
                self._number_of_refutations += 1
                self._number_of_constraint_violation_errors += 1
                if graph_number != 0:
                    self._cnc_tree.add_leaf(
                        parent_number, value_to_assign, VertexType.REFUTATION_LEAF, append_output
                    )
            except ConflictAssignmentException:
                self._number_of_refutations += 1
                if graph_number != 0:
                    self._cnc_tree.add_leaf(
                        parent_number, value_to_assign, VertexType.REFUTATION_LEAF, append_output
                    )
            except UnsatException:
                self._number_of_unsat_solutions += 1
                if graph_number != 0:
                    self._cnc_tree.add_leaf(
                        parent_number, value_to_assign, VertexType.SOLUTION_LEAF, append_output
                    )
            except SatException:
                self._number_of_sat_solutions += 1
                if graph_number != 0:
                    self._cnc_tree.add_leaf(
                        parent_number, value_to_assign, VertexType.SOLUTION_LEAF, append_output
                    )
            except Exception:
                self._number_of_errors += 1

        dfs(0, -1, [], self._append_output)
        cube_time = time() - self._start_cube_time

        report = CncReport(Status.UNSAT)

        done, _ = wait(futures, return_when=FIRST_EXCEPTION)
        for completed_on_error in done:
            try:
                completed_on_error.result()
            except SatException as sat_exception:
                report = CncReport(Status.SAT, sat_exception.satisfying_assignment)
        executor.shutdown(wait=False, cancel_futures=True)

        processes_common_statistics = ProcessesCommonStatistics(worker_stats)

        refutation_leafs_num = self._cnc_tree.refutation_leafs_num + processes_common_statistics.refutations
        cube_leafs_num = self._cnc_tree.cube_leafs_num
        solution_leafs_num = self._cnc_tree.solution_leafs_num
        leafs_num = self._cnc_tree.leafs_num

        self._logger.format({
            'sat solutions': self._number_of_sat_solutions + processes_common_statistics.sat_solutions,
            'sat conflicts': processes_common_statistics.sat_conflicts,
            'sat solvetime': processes_common_statistics.sat_cpu_time,
            'unsat solvetime': processes_common_statistics.unsat_cpu_time,
            'unsat solutions': self._number_of_unsat_solutions + processes_common_statistics.unsat_solutions,
            'unsat conflicts': self._number_of_unsat_solutions + processes_common_statistics.unsat_conflicts,
            'refutation_leafs_num': refutation_leafs_num,
            'cube_leafs_num': cube_leafs_num,
            'solution_leafs_num': solution_leafs_num,
            'leafs_num': leafs_num,
            'tree_size': self._cnc_tree.tree_size,
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
            'cube time': cube_time,
            'algorithm time': time() - self._start_cube_time,
            'number of errors': self._number_of_errors + processes_common_statistics.errors,
            'number of refutations': self._number_of_refutations + processes_common_statistics.refutations,
            'number of controversial assignments': self._number_of_controversial_assignment_errors + processes_common_statistics.controversial_assignments,
            'number of constraint violations': self._number_of_constraint_violation_errors + processes_common_statistics.constraint_violations,
            'sat status': report.status.name + ' ' + report.satisfying_assignment if report.status == Status.SAT else report.status.name
        }, 'cube_statistics.jsonl')

        return report


__all__ = [
    'CubeAndConquerProcess'
]
