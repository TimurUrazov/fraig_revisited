from encoding import AAG
from typing import Tuple, List
from .cnc_heuristics import BranchingHeuristic, HaltingHeuristic, TransformationHeuristic
from util import WorkPath, SatOracleReport, Status, Logger, Solvers, SatParser, MultithreadingLog, solve_cnf_lec_result
from .cnc_tree import CncTree, VertexType
from fraig import Fraig
from .output_handler import OutputHandler
from multiprocessing import Event
from threading import Thread
from exception import (UnsatException, SatException, ConflictAssignmentException, ControversialAssignmentException,
                       ConstraintViolationAssignmentException)
from enum import Enum
from mpi4py import MPI
from queue import Queue
from time import time
from .statistics import ProcessStatistics, ProcessesCommonStatistics, EstimateCounter
from statistics import mean, variance
from random import randint
from .input_decompose import cnf_input_decompose
from util import Report


class TagsMPI(Enum):
    [
        SAT,
        READY,
        START,
        EXIT,
        DONE,
        CONTINUE
    ] = range(6)


eps = 0.1
delta = 0.05


def waiter_fn(
        event_object: Event,
        comm,
        queue: Queue,
        depth,
        estimate_counter: EstimateCounter,
        multithreading_log: MultithreadingLog,
        worker_stats: List[ProcessStatistics]
):
    status = MPI.Status()
    size_of_comm = comm.size
    num_workers = size_of_comm - 1
    closed_workers = 0
    while closed_workers < num_workers:
        comm.recv(source=MPI.ANY_SOURCE, tag=TagsMPI.READY.value, status=status)
        tag = status.Get_tag()
        source = status.Get_source()
        if tag == TagsMPI.READY.value:
            task = queue.get(block=True)
            task = ([task],)
            multithreading_log.emit({
                'comment': 'sending task',
                'destination': source
            })
            comm.send(task, dest=source, tag=TagsMPI.START.value)
            closed_workers += 1

    sent_tasks = closed_workers

    closed_workers = 0

    exps = []

    got_exps_from_num = 0

    expectation = None

    exit_estimate = False

    initial_num_of_exps = 1

    while closed_workers < num_workers:
        data, = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source = status.Get_source()
        tag = status.Get_tag()

        if tag == TagsMPI.DONE.value:
            multithreading_log.emit({
                'comment': 'got estimation',
                'from worker': source
            })
            got_exps_from_num += 1

            cpu_times = []
            for answer, cpu_time, conflicts, time in data:
                if cpu_time is not None:
                    cpu_times.append(cpu_time)
                if answer == Status.UNSAT.value:
                    worker_stats[source - 1].update_unsat_statistics(conflicts, cpu_time)
                    worker_stats[source - 1].update_time(time)
                elif answer == Status.SAT.value:
                    worker_stats[source - 1].update_sat_statistics(conflicts, cpu_time)
                    worker_stats[source - 1].update_time(time)
                    exit_estimate = True
                elif answer == Status.CONSTRAINT_VIOLATION.value:
                    worker_stats[source - 1].increase_refutations(time)
                    worker_stats[source - 1].update_time(time)
                    worker_stats[source - 1].increase_constraint_violation()
                elif answer == Status.CONTROVERSIAL_ASSIGNMENT.value:
                    worker_stats[source - 1].increase_refutations(time)
                    worker_stats[source - 1].update_time(time)
                    worker_stats[source - 1].increase_controversial_assignment()
                elif answer == Status.CONFLICT_ASSIGNMENT.value:
                    worker_stats[source - 1].increase_refutations(time)
                    worker_stats[source - 1].update_time(time)
                elif answer == Status.ERROR.value:
                    worker_stats[source - 1].increase_error()
            exps.extend(cpu_times)
            if got_exps_from_num == sent_tasks:
                if exit_estimate:
                    for i in range(num_workers):
                        comm.send((None, ), dest=i + 1, tag=TagsMPI.EXIT.value)

                expectation = mean(exps)
                variance_ = variance(exps)

                num_of_estimates = len(exps)

                estimate_counter.update_num_of_estimates(num_of_estimates)

                denom = eps * eps * num_of_estimates * expectation * expectation

                multithreading_log.emit({
                    'comment': 'statistics',
                    'variance': variance_,
                    'expectation': expectation,
                    'num of estimates': num_of_estimates,
                    'pr':variance_ / denom if denom > 0 else 'div by zero',
                    'current estimate': expectation * (2 ** depth)
                })

                if not exit_estimate:
                    if variance_ == 0 or denom == 0:
                        boundary = num_of_estimates * 2
                    else:
                        boundary = variance_ / eps * eps * delta * expectation * expectation

                    multithreading_log.emit({
                        'comment': 'statistics',
                        'boundary': boundary,
                    })

                    if num_of_estimates > boundary:
                        exit_estimate = True
                        for i in range(num_workers):
                            comm.send((None,), dest=i + 1, tag=TagsMPI.EXIT.value)
                    else:
                        full_left_work = int(boundary - num_of_estimates) + 1
                        if full_left_work > num_of_estimates:
                            task_n = num_of_estimates
                            initial_num_of_exps *= 2
                        else:
                            task_n = max(full_left_work, num_workers)

                        multithreading_log.emit({
                            'comment': 'ask workers to continue estimating',
                            'tasks left num': task_n
                        })

                        tasks_to_distribute = []
                        for i in range(task_n):
                            queue_task = queue.get(block=True)

                            multithreading_log.emit({
                                'comment': 'took from queue task',
                                'number of task': i,
                                # 'task': queue_task
                            })
                            if queue_task == -1:
                                multithreading_log.emit({'comment': 'estimate will be exited'})
                                exit_estimate = True
                                for i in range(num_workers):
                                    comm.send((None,), dest=i + 1, tag=TagsMPI.EXIT.value)
                                break
                            tasks_to_distribute.append(queue_task)
                        if not exit_estimate:
                            tasks_to_distribute_num = len(tasks_to_distribute)
                            worker_tasks = max(tasks_to_distribute_num // num_workers, 1)
                            threads_num = min(num_workers, tasks_to_distribute_num)
                            sent_tasks = threads_num
                            multithreading_log.emit({'sent tasks num': sent_tasks})
                            overflown = tasks_to_distribute_num - threads_num * worker_tasks
                            offset = 0
                            chunks = []
                            for i in range(threads_num):
                                chunks.append(tasks_to_distribute[offset: offset + (
                                    1 if i < overflown else 0
                                ) + worker_tasks])
                                offset += (1 if i < overflown else 0) + worker_tasks
                            for i in range(threads_num):
                                task = (chunks[i],)
                                comm.send(task, dest=i + 1, tag=TagsMPI.CONTINUE.value)

                got_exps_from_num = 0
        elif tag == TagsMPI.EXIT.value:
            multithreading_log.emit({
                'comment': 'worker exited estimation',
                'number of worker': source
            })
            closed_workers += 1
        elif tag == TagsMPI.SAT.value:
            multithreading_log.emit({
                'comment': 'worker found satisfiable assignment',
                'number of worker': source,
                'satisfiable assignment': data
            })
            estimate_counter.add_sat_solution(data)
            estimate_counter.sat_status = True
            exit_estimate = True

    estimate_counter.mean = expectation
    estimate_counter.estimate = expectation * (2 ** depth)
    event_object.set()


class CubeAndConquerEstimateMPI:
    def __init__(
            self,
            fraig: Fraig,
            start_cube_time: float,
            work_path: WorkPath,
            branching_heuristic: BranchingHeuristic,
            halt_heuristic: HaltingHeuristic,
            transformation_heuristics: TransformationHeuristic,
            worker_solver: Solvers,
            logger: Logger,
            append_output: bool,
            use_input_decomposition: bool,
            decomposition_limit: int,
            length_decompose: str,
            shuffle_inputs: bool
    ):
        self._fraig = fraig

        self._start_cube_time = start_cube_time

        self._logger = logger

        self._multithreading_logger = MultithreadingLog(
            self._logger,
            "_".join(["process", str(MPI.COMM_WORLD.Get_rank())]) + ".log"
        )

        self._branching_heuristic = branching_heuristic
        self._transformation_heuristics = transformation_heuristics
        self._halt_heuristic = halt_heuristic

        self._cnc_tree = CncTree()

        self._cubes_path: WorkPath = work_path.to_path('cubes')

        self._worker_solver = worker_solver

        self._number_of_sat_solutions = 0
        self._number_of_unsat_solutions = 0
        self._number_of_errors = 0

        self._number_of_constraint_violation_errors = 0
        self._number_of_controversial_assignment_errors = 0
        self._number_of_refutations = 0

        self._append_output = append_output

        self._use_input_decomposition = use_input_decomposition

        self._decomposition_limit = decomposition_limit

        self._length_decompose = length_decompose

        self._shuffle_inputs = shuffle_inputs

    def worker_fn_process(
            self,
            cube_filename: str,
            aag: AAG,
            append_outputs: bool,
            initial_input_to_assignment,
            current_input_to_initial,
            initial_current_inputs_to_initial,
            initial_initial_inputs_to_assignments,
            comm
    ):
        worker_start_time = MPI.Wtime()
        try:
            output_handler = OutputHandler(append_outputs)
            aag = self._halt_heuristic.halt_transformation(aag, None, output_handler)

            cnf, aag_to_cnf_inputs_vars = aag.to_cnf_with_mapping()

            self._multithreading_logger.emit({
                'append_output': output_handler.append_outputs
            })

            if self._use_input_decomposition and len(cnf.get_data().get_inputs()) > 1:
                outputs = cnf.get_data().get_outputs()
                constraints = [outputs] if output_handler.append_outputs and len(outputs) > 0 else []
                cnf_source = cnf.get_data().source(supplements=([], constraints))

                cpu_time, conflicts, satisfying_assignment_cnf_dict, answer = cnf_input_decompose(
                    self._shuffle_inputs,
                    cnf_source,
                    self._length_decompose,
                    self._decomposition_limit,
                    self._worker_solver,
                    self._multithreading_logger
                )

            else:
                result = solve_cnf_lec_result(
                    cnf, self._worker_solver, include_outputs=output_handler.append_outputs
                )

                satisfying_assignment = SatParser(result)

                cpu_time, conflicts, satisfying_assignment_cnf_dict, answer = satisfying_assignment.parse_complete()

                self._multithreading_logger.emit({
                    'solved task status': answer.value,
                    'solved task conflicts': conflicts,
                    'solved task cpu time': cpu_time,
                })

            if answer == Status.UNSAT or answer == 'UNSAT':
                return 'UNSAT', cpu_time, conflicts, MPI.Wtime() - worker_start_time

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

            self._multithreading_logger.emit({
                'found satisfying assignment': sorted_input_in_str,
                'solved task conflicts': conflicts,
                'solved task cpu time': cpu_time,
            })

            comm.send((sorted_input_in_str,), dest=0, tag=TagsMPI.SAT.value)
            return 'SAT', cpu_time, conflicts, MPI.Wtime() - worker_start_time
        except UnsatException:
            return Status.UNSAT.value, .0, 1, MPI.Wtime() - worker_start_time
        except ControversialAssignmentException:
            return Status.CONTROVERSIAL_ASSIGNMENT.value, None, None, MPI.Wtime() - worker_start_time
        except ConstraintViolationAssignmentException:
            return Status.CONSTRAINT_VIOLATION.value, None, None, MPI.Wtime() - worker_start_time
        except ConflictAssignmentException:
            return Status.CONFLICT_ASSIGNMENT.value, None, None, MPI.Wtime() - worker_start_time

    def cube_and_conquer(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        status = MPI.Status()

        if rank == 0:
            estimate_counter = EstimateCounter()
            event_object = Event()
            start_time = MPI.Wtime()
            queue = Queue(maxsize=1000)
            worker_stats = [ProcessStatistics() for _ in range(comm.size - 1)]
            t = Thread(target=waiter_fn, args=(
                event_object,
                comm,
                queue,
                self._halt_heuristic.max_depth,
                estimate_counter,
                self._multithreading_logger,
                worker_stats
            ))
            t.start()

            global_sat_oracle_report = SatOracleReport()

            global_sat_oracle_report.update_report(self._fraig.sat_oracle_report())

            initial_current_inputs_to_initial = self._fraig.graph_info().current_input_to_initial
            initial_initial_inputs_to_assignments = self._fraig.graph_info().initial_input_to_assignment
        else:
            initial_current_inputs_to_initial = None
            initial_initial_inputs_to_assignments = None

        initial_current_inputs_to_initial = comm.bcast(initial_current_inputs_to_initial, root=0)
        initial_initial_inputs_to_assignments = comm.bcast(initial_initial_inputs_to_assignments, root=0)

        if rank == 0:
            self._number_of_sat_solutions = 0
            self._number_of_unsat_solutions = 0
            self._number_of_errors = 0

            aag_ = self._fraig.to_aag()

            def get_conflicts_and_time(status: Status, sat_oracle_report_local: SatOracleReport):
                if status in sat_oracle_report_local:
                    return sat_oracle_report_local[status].conflicts, sat_oracle_report_local[status].process_time
                else:
                    return 0, 0.0

            def dfs(
                    graph_number: int,
                    parent_number: int,
                    stack: List[Tuple[int, int]],
                    append_output: bool,
                    rightmost: bool
            ):
                if event_object.is_set():
                    return
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

                    if graph_number != 0:
                        self._cnc_tree.add_leaf(
                            parent_number, value_to_assign, VertexType.LEAF, output_handler.append_outputs
                        )

                    self._multithreading_logger.emit({
                        'After transformation depth is': self._cnc_tree[graph_number].depth
                    })

                    global_sat_oracle_report.update_report(fraig.sat_oracle_report())

                    initial_input_to_assignment = fraig.graph_info().initial_input_to_assignment
                    current_input_to_initial = fraig.graph_info().current_input_to_initial

                    aag = fraig.to_aag()

                    if self._halt_heuristic.ensure_halt(self._cnc_tree, graph_number, aag):
                        assert output_handler.append_outputs == self._cnc_tree[graph_number].append_output
                        estimate_counter.last_task_num = estimate_counter.last_task_num + 1
                        if rightmost:
                            estimate_counter.cube_max_num = estimate_counter.last_task_num
                            self._multithreading_logger.emit({
                                'cube_max_num is': estimate_counter.cube_max_num
                            })
                        self._multithreading_logger.emit({
                            'Gave cube to executor on depth': self._cnc_tree[graph_number].depth
                        })
                        self._cnc_tree.make_leaf_cube(graph_number)

                        cube_filename = self._cubes_path.to_file(str(graph_number))

                        queue_task = (
                            cube_filename,
                            aag,
                            initial_input_to_assignment,
                            current_input_to_initial,
                            output_handler.append_outputs
                        )

                        queue.put(queue_task, block=True)
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

                    if not event_object.is_set():
                        for value_to_assign in (range(2) if randint(0, 1) == 1 else range(1, -1, -1)):
                            stack.append((vertex_to_assign, value_to_assign))
                            child_graph_num = CncTree.count_child_num(graph_number, value_to_assign)
                            dfs(child_graph_num, graph_number, stack, output_handler.append_outputs, rightmost and (value_to_assign == 1))
                            stack.pop()
                except ConflictAssignmentException:
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

            dfs(0, -1, [], self._append_output, True)
            cube_time = time() - self._start_cube_time
            queue.put(-1)
            event_object.wait()
            t.join()

            refutation_leafs_num = self._cnc_tree.refutation_leafs_num
            cube_leafs_num = self._cnc_tree.cube_leafs_num
            solution_leafs_num = self._cnc_tree.solution_leafs_num
            leafs_num = self._cnc_tree.leafs_num

            sat_oracle_unsat_conflicts_summary, sat_oracle_unsat_cpu_time_summary = get_conflicts_and_time(
                Status.UNSAT, global_sat_oracle_report
            )
            sat_oracle_sat_conflicts_summary, sat_oracle_sat_cpu_time_summary = get_conflicts_and_time(
                Status.SAT, global_sat_oracle_report
            )
            sat_oracle_indet_conflicts_summary, sat_oracle_indet_cpu_time_summary = get_conflicts_and_time(
                Status.INDET, global_sat_oracle_report
            )

            sat_oracle_conflicts_summary = (sat_oracle_unsat_conflicts_summary +
                                            sat_oracle_sat_conflicts_summary + sat_oracle_indet_conflicts_summary)

            sat_oracle_cpu_time_summary = (sat_oracle_unsat_cpu_time_summary +
                                            sat_oracle_sat_cpu_time_summary + sat_oracle_indet_cpu_time_summary)

            is_sat = estimate_counter.sat_status

            processes_common_statistics = ProcessesCommonStatistics(worker_statistics=worker_stats)

            log_info = {
                'is real result or estimate?': ('real result' if estimate_counter.all_tasks_done else 'estimate'),
                'estimated hardness is': estimate_counter.estimate,
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
                'number_of_unsat_solutions_on_prop': self._number_of_unsat_solutions,
                'number_of_sat_solutions_on_prop': self._number_of_sat_solutions,
                'number_of_errors_on_prop': self._number_of_errors,
                'tree_size': self._cnc_tree.tree_size,
                'sat_oracle_unsat_conflicts_summary': sat_oracle_unsat_conflicts_summary,
                'sat_oracle_sat_conflicts_summary': sat_oracle_sat_conflicts_summary,
                'sat_oracle_indet_conflicts_summary': sat_oracle_indet_conflicts_summary,
                'sat_oracle_unsat_cpu_time_summary': sat_oracle_unsat_cpu_time_summary,
                'sat_oracle_sat_cpu_time_summary': sat_oracle_sat_cpu_time_summary,
                'sat_oracle_indet_cpu_time_summary': sat_oracle_indet_cpu_time_summary,
                'number_of_sat_oracle_unsat_statuses': global_sat_oracle_report.number_of_unsat_statuses,
                'number_of_sat_oracle_sat_statuses': global_sat_oracle_report.number_of_sat_statuses,
                'number_of_sat_oracle_indet_statuses': global_sat_oracle_report.number_of_indet_statuses,
                'sat_oracle_conflicts_summary': sat_oracle_conflicts_summary,
                'sat_oracle_cpu_time_summary': sat_oracle_cpu_time_summary,
                'sat_status': "SAT" if is_sat else "UNSAT",
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
                'estimation took': MPI.Wtime() - start_time
            }

            if is_sat:
                satisfying_assignments = estimate_counter.satisfying_assignments
                self._multithreading_logger.emit({
                    'Satisfying assignments are': satisfying_assignments
                })

            self._multithreading_logger.emit(log_info)
            with self._logger as logger:
                logger.format(log_info, 'cube_statistics.jsonl')
            return Report(
                process_time=estimate_counter.estimate, status=Status.SAT if is_sat else Status.UNSAT, conflicts=0
            )
        else:
            comm.send((None,), dest=0, tag=TagsMPI.READY.value)
            while True:
                data, = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                tag = status.Get_tag()
                if tag == TagsMPI.START.value or tag == TagsMPI.CONTINUE.value:
                    answers = []
                    self._multithreading_logger.emit({
                        'comment': 'got task',
                        'data': len(data)
                    })
                    for (
                        cube_filename,
                        aag,
                        initial_input_to_assignment,
                        current_input_to_initial,
                        append_outputs
                    ) in data:
                        answer = self.worker_fn_process(
                            cube_filename,
                            aag,
                            append_outputs,
                            initial_input_to_assignment,
                            current_input_to_initial,
                            initial_current_inputs_to_initial,
                            initial_initial_inputs_to_assignments,
                            comm
                        )
                        answers.append(answer)
                    self._multithreading_logger.emit({
                        'comment': 'done'
                    })
                    comm.send((answers, ), dest=0, tag=TagsMPI.DONE.value)
                elif tag == TagsMPI.EXIT.value:
                    self._multithreading_logger.emit({
                        'comment': 'exiting executing estimation'
                    })
                    comm.send((None,), dest=0, tag=TagsMPI.EXIT.value)
                    break


__all__ = [
    'CubeAndConquerEstimateMPI'
]
