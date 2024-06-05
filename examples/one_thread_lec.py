from util import WorkPath, Solvers, solve_aag_lec, Logger
from scheme import AAGLEC


def solve_one_thread_cnf(sol: Solvers, first_scheme_filename: str, second_scheme_filename: str, logger: Logger):
    root_path = WorkPath('aiger')
    left_scheme_file = root_path.to_file(first_scheme_filename)
    right_scheme_file = root_path.to_file(second_scheme_filename)
    aag_lec_cnf = AAGLEC(
        left_scheme_from_file=left_scheme_file,
        right_scheme_from_file=right_scheme_file
    ).get_scheme_till_xor()
    solution = solve_aag_lec(aag_lec_cnf, sol)
    with logger as logger:
        logger.format({
            'solver': sol.name,
            'status': solution.status.name,
            'process time': solution.process_time,
            'conflicts': solution.conflicts
        }, first_scheme_filename + '_' + second_scheme_filename + '.jsonl')


def solve_one_thread_cnf_kvd(sol: Solvers, logger: Logger):
    solve_one_thread_cnf(sol, 'karatsuba12x12.aag', 'dadda12x12.aag', logger)


def solve_one_thread_cnf_kvc(sol: Solvers, logger: Logger):
    solve_one_thread_cnf(sol, 'karatsuba12x12.aag', 'column12x12.aag', logger)


def solve_one_thread_cnf_kvw(sol: Solvers, logger: Logger):
    solve_one_thread_cnf(sol, 'karatsuba12x12.aag', 'wallace12x12.aag', logger)


def solve_one_thread_cnf_dvw(sol: Solvers, logger: Logger):
    solve_one_thread_cnf(sol, 'dadda12x12.aag', 'wallace12x12.aag', logger)


def solve_one_thread_cnf_dvc(sol: Solvers, logger: Logger):
    solve_one_thread_cnf(sol, 'dadda12x12.aag', 'column12x12.aag', logger)


def solve_one_thread_cnf_wvc(sol: Solvers, logger: Logger):
    solve_one_thread_cnf(sol, 'wallace12x12.aag', 'column12x12.aag', logger)


def solve_one_thread_cnf_kvd_kissat_2022(logger: Logger):
    return solve_one_thread_cnf_kvd(Solvers.KISSAT_2022, logger)


def solve_one_thread_cnf_kvc_cadical(logger: Logger):
    return solve_one_thread_cnf_wvc(Solvers.CADICAL, logger)
