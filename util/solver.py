import subprocess
from .config import CONFIG
from enum import Enum
from typing import TYPE_CHECKING, Optional, List
from collections import defaultdict

if TYPE_CHECKING:
    from encoding import AAG, CNF


class Status(Enum):
    [
        SAT,
        UNSAT,
        INDET,
        CONSTRAINT_VIOLATION,
        CONTROVERSIAL_ASSIGNMENT,
        CONFLICT_ASSIGNMENT,
        ERROR
    ] = [
        'SAT',
        'UNSAT',
        'INDET',
        'CONSTRAINT_VIOLATION',
        'CONTROVERSIAL_ASSIGNMENT',
        'CONFLICT_ASSIGNMENT',
        'ERROR'
    ]


class Report:
    def __init__(self, status: Optional[Status], conflicts: int, process_time: float):
        self._status = status
        self._conflicts = conflicts
        self._process_time = process_time

    @property
    def process_time(self) -> Optional[float]:
        return self._process_time

    def update_report(self, report: 'Report'):
        self._process_time += report.process_time
        self._conflicts += report.conflicts

    @property
    def conflicts(self) -> Optional[int]:
        return self._conflicts

    @property
    def status(self) -> Optional[Status]:
        return self._status


class SatOracleReport(dict):
    def __init__(self):
        super().__init__()
        self._status_counter = defaultdict(int)

    def __getitem__(self, item) -> Report:
        return super().__getitem__(item)

    def __setitem__(self, key: Status, value: Report):
        super().__setitem__(key, value)

    def update_report_with_status(self, report: Report, status_count: int = 1):
        self._status_counter[report.status] += status_count
        if report.status in self:
            self[report.status].update_report(report)
        else:
            self[report.status] = report

    def update_report(self, report: 'SatOracleReport'):
        for status in report:
            self.update_report_with_status(report[status], report._status_counter[status])

    @property
    def number_of_unsat_statuses(self) -> int:
        return self._status_counter[Status.UNSAT]

    @property
    def number_of_sat_statuses(self) -> int:
        return self._status_counter[Status.SAT]

    @property
    def number_of_indet_statuses(self) -> int:
        return self._status_counter[Status.INDET]


def solve_cnf(cnf_str: str, args):
    solver = subprocess.run(
        args,
        capture_output=True,
        text=True,
        input=cnf_str
    )
    result = solver.stdout.split("\n")
    errors = solver.stderr
    if len(errors) > 0:
        print("exception:", errors)
    return result


def solve_cnf_with_kissat(config_path, cnf_str, conflicts_limit=None, time_limit_in_seconds=None):
    kissat_args = [config_path]
    if conflicts_limit is not None:
        kissat_args.append("--conflicts={}".format(int(conflicts_limit)))
    if time_limit_in_seconds is not None:
        kissat_args.append("--time={}".format(int(time_limit_in_seconds)))
    return solve_cnf(cnf_str, kissat_args)


def solve_cnf_with_kissat_2023(cnf_str, conflicts_limit=None, time_limit_in_seconds=None):
    return solve_cnf_with_kissat(CONFIG.path_to_kissat_2023(), cnf_str, conflicts_limit, time_limit_in_seconds)


def solve_cnf_with_kissat_2022(cnf_str, conflicts_limit=None, time_limit_in_seconds=None):
    return solve_cnf_with_kissat(CONFIG.path_to_kissat_2022(), cnf_str, conflicts_limit, time_limit_in_seconds)


def solve_cnf_with_rokk_lrb(cnf_str: str):
    return solve_cnf(cnf_str, [CONFIG.path_to_rokk_lrb()])


def solve_cnf_with_cadical(cnf_str: str):
    return solve_cnf(cnf_str, [CONFIG.path_to_cadical()])


def solve_cnf_with_rokk(cnf_str: str):
    return solve_cnf(cnf_str, [CONFIG.path_to_rokk()])


class Solvers(Enum):
    [
        KISSAT_2023,
        ROKK_LRB,
        KISSAT_2022,
        CADICAL,
        ROKK,
    ] = range(5)


solver_handlers = {
    0: solve_cnf_with_kissat_2023,
    1: solve_cnf_with_rokk_lrb,
    2: solve_cnf_with_kissat_2022,
    3: solve_cnf_with_cadical,
    4: solve_cnf_with_rokk
}


def solve_cnf_source(solver: Solvers, cnf_str: str) -> List[str]:
    return solver_handlers[solver.value](
        cnf_str
    )


def solve_cnf_lec_result(aag_lec_cnf: 'CNF', solver: Solvers, include_outputs: bool = True) -> List[str]:
    outputs = aag_lec_cnf.get_data().get_outputs()
    constraints = [outputs] if include_outputs and len(outputs) > 0 else []
    result = solver_handlers[solver.value](
        aag_lec_cnf.get_data().source(supplements=([], constraints))
    )
    return result


def solve_aag_lec(aag: 'AAG', solver: Solvers, include_outputs: bool = True) -> Report:
    return analyze_result(
        solve_cnf_lec_result(aag.to_cnf_with_constraints(), solver, include_outputs), print_output=False
    )


def solve_cnf_lec(aag_lec_cnf: 'CNF', solver: Solvers, include_outputs: bool = True) -> Report:
    return analyze_result(solve_cnf_lec_result(aag_lec_cnf, solver, include_outputs), print_output=False)


def process_result(result):
    solvetime = None
    conflicts = None
    answer = Status.INDET
    satisfying_assignment = []
    for line in result:
        if "c process-time" in line or "c CPU time" in line or "c total process time since initialization" in line:
            solvetime = float(line.split()[-2])
        elif "c conflicts:" in line:
            conflicts = int(line.split()[-4])
        elif "c conflicts " in line:
            conflicts = int(line.split()[3])
        elif line.startswith("v "):
            satisfying_assignment.extend([(abs(number), int(number > 0)) for number in map(int, line.split()[1:])])
        elif len(line) > 0 and line[0] == "s":
            if "UNSAT" in line:
                answer = Status.UNSAT
            elif "SAT" in line:
                answer = Status.SAT
    return solvetime, conflicts, dict(satisfying_assignment), answer


class SatParser:
    def __init__(self, result):
        self._result = result

    def parse_complete(self):
        return process_result(self._result)


def analyze_result(result, comment=None, check_solvetime=0, check_conflicts=0, print_output=True) -> Report:
    if result is None:
        answer = "Simplified and solved after fraiging without using sat solver"
        solvetime = 0
        conflicts = 0
    else:
        solvetime, conflicts, _, answer = process_result(result)

    if print_output:
        if comment is not None:
            print(comment)

        print("answer:", answer, "solvetime:", str(solvetime + check_solvetime), "conflicts:",
              str(conflicts + check_conflicts))

    return Report(process_time=solvetime, status=answer, conflicts=conflicts)


__all__ = [
    'solve_cnf_with_kissat_2022',
    'solve_cnf_with_kissat_2023',
    'solve_cnf_with_rokk_lrb',
    'analyze_result',
    'solve_aag_lec',
    'solve_cnf_lec',
    'Status',
    'Solvers',
    'Report',
    'SatOracleReport',
    'SatParser',
    'solve_cnf_lec_result',
    'solve_cnf_source'
]
