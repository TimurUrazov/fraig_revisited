from algorithm import (IntegralDegreeLookAheadBranchingHeuristic, LeafSizeHaltingHeuristic,
                       FraigTransformationHeuristic)
from lec import LogicalEquivalenceChecking
from util import WorkPath, date_now, Solvers


def logical_equivalence_with_integral_degree(
        solver: Solvers,
        number_of_workers: int,
        left_scheme_file: str,
        right_scheme_file: str
):
    LogicalEquivalenceChecking(
        left_scheme_file=left_scheme_file,
        right_scheme_file=right_scheme_file,
        branching_heuristic=IntegralDegreeLookAheadBranchingHeuristic(),
        halt_heuristic=LeafSizeHaltingHeuristic(
            leafs_size_lower_bound=2300,
            max_depth=8,
            reduction_conflicts=300
        ),
        transformation_heuristics=FraigTransformationHeuristic(),
        work_path=WorkPath('out').to_path(date_now()),
        worker_solver=solver
    ).solve_using_processes(number_of_executor_workers=number_of_workers)


def solve_cnc_lec_c_v_w_12_rokk_d8():
    root_path = WorkPath('aiger')

    c_v_w_12 = (root_path.to_file('column12x12.aag'), root_path.to_file('wallace12x12.aag'))

    logical_equivalence_with_integral_degree(Solvers.ROKK_LRB, 8, *c_v_w_12)
