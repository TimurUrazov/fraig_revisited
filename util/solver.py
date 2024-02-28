import subprocess
from .config import CONFIG


def solve_cnf_with_kissat(cnf_str, conflicts_limit=None, time_limit_in_seconds=None):
    kissat_args = [CONFIG.path_to_kissat()]
    if conflicts_limit is not None:
        kissat_args.append("--conflicts={}".format(int(conflicts_limit)))
    if time_limit_in_seconds is not None:
        kissat_args.append("--time={}".format(int(time_limit_in_seconds)))
    solver = subprocess.run(
        kissat_args,
        capture_output=True,
        text=True,
        input=cnf_str
    )
    result = solver.stdout.split("\n")
    errors = solver.stderr
    if len(errors) > 0:
        print("errors:", errors)
    return result


def solve_cnf_with_rokk(cnf_str: str):
    kissat_args = [CONFIG.path_to_rokk()]
    solver = subprocess.run(
        kissat_args,
        capture_output=True,
        text=True,
        input=cnf_str
    )
    result = solver.stdout.split("\n")
    errors = solver.stderr
    if len(errors) > 0:
        print("errors:", errors)
    return result


def analyze_result(result, comment=None, check_solvetime=0, check_conflicts=0, print_output=True):
    answer = None
    solvetime = None
    conflicts = None

    if result is None:
        answer = "Simplified and solved after fraiging without using sat solver"
        solvetime = 0
        conflicts = 0
    else:
        for line in result:
            if "c process-time" in line or "c CPU time" in line:
                solvetime = float(line.split()[-2])
            elif "c conflicts:" in line:
                conflicts = int(line.split()[-4])
            elif "c conflicts " in line:
                conflicts = int(line.split()[3])
            elif len(line) > 0 and line[0] == "s":
                if "UNSAT" in line:
                    answer = "UNSAT"
                elif "SAT" in line:
                    answer = "SAT"

    if print_output:
        if comment is not None:
            print(comment)

        print("answer:", answer, "solvetime:", str(solvetime + check_solvetime), "conflicts:",
              str(conflicts + check_conflicts))

    return answer, solvetime, conflicts


__all__ = [
    'solve_cnf_with_kissat',
    'solve_cnf_with_rokk',
    'analyze_result'
]
