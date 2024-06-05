import math
from typing import List
from multiprocessing import Value
import threading


class ProcessSharedCounter:
    def __init__(self):
        self._unsat_conflicts: Value = Value('i', 0)
        self._sat_conflicts: Value = Value('i', 0)
        self._unsat_cpu_time: Value = Value('f', 0)
        self._sat_cpu_time: Value = Value('f', 0)
        self._unsat_solutions: Value = Value('i', 0)
        self._sat_solutions: Value = Value('i', 0)
        self._refutations: Value = Value('i', 0)
        self._sat_status: Value = Value('b', False)

    def update_unsat_statistics(self, conflicts: int, cpu_time: float):
        with self._unsat_solutions.get_lock():
            if self._sat_status.value:
                return
            self._unsat_solutions.value += 1
            self._unsat_conflicts.value += conflicts
            self._unsat_cpu_time.value += cpu_time

    def update_sat_statistics(self, conflicts: int, cpu_time: float):
        with self._sat_solutions.get_lock():
            if self._sat_status.value:
                return
            self._sat_status.value = True
            self._sat_solutions.value += 1
            self._sat_conflicts.value += conflicts
            self._sat_cpu_time.value += cpu_time

    def increase_unsat_cpu_time(self, cpu_time: float):
        with self._unsat_cpu_time.get_lock():
            if self._sat_status.value:
                return
            self._unsat_cpu_time.value += cpu_time

    def increase_sat_cpu_time(self, cpu_time: float):
        with self._sat_cpu_time.get_lock():
            if self._sat_status.value:
                return
            self._sat_cpu_time.value += cpu_time

    def increase_refutations(self):
        with self._refutations.get_lock():
            if self._sat_status.value:
                return
            self._refutations.value += 1

    @property
    def unsat_conflicts(self) -> int:
        return self._unsat_conflicts.value

    @property
    def sat_conflicts(self) -> int:
        return self._sat_conflicts.value

    @property
    def sat_cpu_time(self) -> float:
        return self._sat_cpu_time.value

    @property
    def unsat_cpu_time(self) -> float:
        return self._unsat_cpu_time.value

    @property
    def unsat_solutions(self) -> int:
        return self._unsat_solutions.value

    @property
    def sat_solutions(self) -> int:
        return self._sat_solutions.value

    @property
    def refutations(self) -> int:
        return self._refutations.value


class ThreadSharedCounter:
    def __init__(self):
        self._lock = threading.Lock()
        self._unsat_conflicts = 0
        self._sat_conflicts = 0
        self._unsat_cpu_time = 0
        self._sat_cpu_time = 0
        self._unsat_solutions = 0
        self._sat_solutions = 0
        self._refutations = 0

    def increase_unsat_statistics(self, conflicts: int, cpu_time: float):
        with self._lock:
            self._unsat_solutions += 1
            self._unsat_conflicts += conflicts
            self._unsat_cpu_time += cpu_time

    def update_sat_statistics(self, conflicts: int, cpu_time: float):
        with self._lock:
            self._sat_solutions += 1
            self._sat_conflicts += conflicts
            self._sat_cpu_time += cpu_time

    def increase_refutations(self):
        with self._lock:
            self._refutations += 1

    @property
    def unsat_conflicts(self) -> int:
        return self._unsat_conflicts

    @property
    def sat_conflicts(self) -> int:
        return self._sat_conflicts

    @property
    def sat_cpu_time(self) -> int:
        return self._sat_cpu_time

    @property
    def unsat_cpu_time(self) -> int:
        return self._unsat_cpu_time

    @property
    def unsat_solutions(self) -> int:
        return self._unsat_solutions

    @property
    def sat_solutions(self) -> int:
        return self._sat_solutions

    @property
    def refutations(self) -> int:
        return self._refutations


class ProcessStatistics:
    def __init__(self):
        self._unsat_conflicts: Value = Value('i', 0)
        self._sat_conflicts: Value = Value('i', 0)
        self._unsat_cpu_time: Value = Value('f', 0)
        self._sat_cpu_time: Value = Value('f', 0)
        self._unsat_solutions: Value = Value('i', 0)
        self._sat_solutions: Value = Value('i', 0)
        self._refutations: Value = Value('i', 0)
        self._time: Value = Value('f', 0)
        self._refutation_time: Value = Value('f', 0)
        self._constraint_violation: Value = Value('i', 0)
        self._controversial_assignment: Value = Value('i', 0)
        self._errors: Value = Value('i', 0)
        self._num_of_solved_tasks: Value = Value('i', 0)
        self._num_of_conflicts: Value = Value('i', 0)

    def _increase_task_counter(self):
        self._num_of_solved_tasks.value += 1

    def _increase_conflict_or_error_counter(self):
        self._num_of_conflicts.value += 1

    def update_unsat_statistics(self, conflicts: int, cpu_time: float):
        self._increase_task_counter()
        self._unsat_solutions.value += 1
        self._unsat_conflicts.value += conflicts
        self._unsat_cpu_time.value += cpu_time

    def update_sat_statistics(self, conflicts: int, cpu_time: float):
        self._increase_task_counter()
        self._sat_solutions.value += 1
        self._sat_conflicts.value += conflicts
        self._sat_cpu_time.value += cpu_time

    def update_time(self, time: float):
        self._time.value += time

    def increase_refutations(self, refutation_time: float):
        self._increase_conflict_or_error_counter()
        self._refutations.value += 1
        self._refutation_time.value += refutation_time

    def increase_controversial_assignment(self):
        self._increase_conflict_or_error_counter()
        self._controversial_assignment.value += 1

    def increase_constraint_violation(self):
        self._increase_conflict_or_error_counter()
        self._constraint_violation.value += 1

    def increase_error(self):
        self._increase_conflict_or_error_counter()
        self._errors.value += 1

    @property
    def unsat_conflicts(self) -> int:
        return self._unsat_conflicts.value

    @property
    def sat_conflicts(self) -> int:
        return self._sat_conflicts.value

    @property
    def sat_cpu_time(self) -> float:
        return self._sat_cpu_time.value

    @property
    def unsat_cpu_time(self) -> float:
        return self._unsat_cpu_time.value

    @property
    def unsat_solutions(self) -> int:
        return self._unsat_solutions.value

    @property
    def sat_solutions(self) -> int:
        return self._sat_solutions.value

    @property
    def refutations(self) -> int:
        return self._refutations.value

    @property
    def cpu_time(self) -> float:
        return self._unsat_cpu_time.value + self._sat_cpu_time.value

    @property
    def time(self) -> float:
        return self._time.value

    @property
    def refutation_time(self) -> float:
        return self._refutation_time.value

    @property
    def errors(self) -> int:
        return self._errors.value

    @property
    def constraint_violations(self) -> int:
        return self._constraint_violation.value

    @property
    def controversial_assignments(self) -> int:
        return self._controversial_assignment.value

    @property
    def conflicts(self) -> int:
        return self._unsat_conflicts.value + self._sat_conflicts.value + self._refutations.value

    @property
    def conflicts_or_errors(self) -> int:
        return self._num_of_conflicts.value

    @property
    def solved_tasks(self) -> int:
        return self._num_of_solved_tasks.value


class ProcessesCommonStatistics:
    def __init__(self, worker_statistics: List[ProcessStatistics]):
        self._cpu_time = 0
        self._conflicts = 0
        self._max_cpu_time = -1
        self._min_cpu_time = 1e18
        self._max_conflicts = -1
        self._min_conflicts = 1e18
        self._sat_solutions = 0
        self._unsat_solutions = 0
        self._refutations = 0
        self._sat_cpu_time = 0
        self._unsat_cpu_time = 0
        self._sat_conflicts = 0
        self._unsat_conflicts = 0
        self._errors = 0
        self._constraint_violations = 0
        self._controversial_assignments = 0

        for worker_statistic in worker_statistics:
            self._sat_solutions += worker_statistic.sat_solutions
            self._unsat_solutions += worker_statistic.unsat_solutions
            self._cpu_time += worker_statistic.cpu_time
            self._conflicts += worker_statistic.conflicts
            self._refutations += worker_statistic.refutations
            self._sat_cpu_time += worker_statistic.sat_cpu_time
            self._unsat_cpu_time += worker_statistic.unsat_cpu_time
            self._sat_conflicts += worker_statistic.sat_conflicts
            self._unsat_conflicts += worker_statistic.unsat_conflicts
            self._max_cpu_time = max(worker_statistic.cpu_time, self._max_cpu_time)
            self._min_cpu_time = min(worker_statistic.cpu_time, self._min_cpu_time)
            self._max_conflicts = max(worker_statistic.conflicts, self._max_conflicts)
            self._min_conflicts = min(worker_statistic.conflicts, self._min_conflicts)
            self._errors += worker_statistic.errors
            self._constraint_violations += worker_statistic.constraint_violations
            self._controversial_assignments += worker_statistic.controversial_assignments

        number_of_worker_statistics = len(worker_statistics)
        self._average_cpu_time = self._cpu_time / number_of_worker_statistics
        self._average_conflicts = self._conflicts / number_of_worker_statistics

        self._cpu_time_standard_deviation = 0
        self._conflicts_standard_deviation = 0

        for worker_statistic in worker_statistics:
            cpu_time_delta = self._average_cpu_time - worker_statistic.cpu_time
            conflicts_delta = self._average_conflicts - worker_statistic.conflicts
            self._cpu_time_standard_deviation += cpu_time_delta * cpu_time_delta
            self._conflicts_standard_deviation += conflicts_delta * conflicts_delta

        self._cpu_time_standard_deviation /= (number_of_worker_statistics - 1)
        self._cpu_time_standard_deviation = math.sqrt(self._cpu_time_standard_deviation)
        self._conflicts_standard_deviation /= (number_of_worker_statistics - 1)
        self._conflicts_standard_deviation = math.sqrt(self._conflicts_standard_deviation)

    @property
    def cpu_time(self) -> float:
        return self._cpu_time

    @property
    def average_cpu_time(self) -> float:
        return self._average_cpu_time

    @property
    def max_cpu_time(self) -> float:
        return self._max_cpu_time

    @property
    def min_cpu_time(self) -> float:
        return self._min_cpu_time

    @property
    def conflicts(self) -> int:
        return self._conflicts

    @property
    def average_conflicts(self) -> float:
        return self._average_conflicts

    @property
    def max_conflicts(self) -> int:
        return self._max_conflicts

    @property
    def min_conflicts(self) -> int:
        return self._min_conflicts

    @property
    def conflicts_standard_deviation(self) -> float:
        return self._conflicts_standard_deviation

    @property
    def cpu_time_standard_deviation(self) -> float:
        return self._cpu_time_standard_deviation

    @property
    def sat_solutions(self) -> int:
        return self._sat_solutions

    @property
    def unsat_solutions(self) -> int:
        return self._unsat_solutions

    @property
    def refutations(self) -> int:
        return self._refutations

    @property
    def sat_cpu_time(self) -> float:
        return self._sat_cpu_time

    @property
    def unsat_cpu_time(self) -> float:
        return self._unsat_cpu_time

    @property
    def sat_conflicts(self) -> float:
        return self._sat_conflicts

    @property
    def unsat_conflicts(self) -> float:
        return self._unsat_conflicts

    @property
    def constraint_violations(self) -> float:
        return self._constraint_violations

    @property
    def controversial_assignments(self) -> float:
        return self._controversial_assignments

    @property
    def errors(self) -> float:
        return self._errors


class EstimateCounter:
    def __init__(self):
        self._sat_status = False
        self._satisfying_assignments = []
        self._estimate = None
        self._cube_max_num = None
        self._all_tasks_done = False
        self._mean = None
        self._last_task_num = 0

    @property
    def estimate(self):
        return self._estimate

    @estimate.setter
    def estimate(self, value: float):
        self._estimate = value

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, value: float):
        self._mean = value

    @property
    def all_tasks_done(self) -> bool:
        return self._all_tasks_done

    def add_sat_solution(self, sat_solution: str):
        self._satisfying_assignments.append(sat_solution)

    def update_num_of_estimates(self, num_of_estimates: int):
        if num_of_estimates == self._cube_max_num:
            self._all_tasks_done = True

    @property
    def sat_status(self) -> bool:
        return self._sat_status

    @sat_status.setter
    def sat_status(self, value):
        self._sat_status = value

    @property
    def satisfying_assignments(self):
        return self._satisfying_assignments

    @property
    def cube_max_num(self) -> int:
        return self._cube_max_num

    @cube_max_num.setter
    def cube_max_num(self, value):
        self._cube_max_num = value

    @property
    def last_task_num(self) -> int:
        return self._last_task_num

    @last_task_num.setter
    def last_task_num(self, value):
        self._last_task_num = value
