from algorithm import DegreeBranchingHeuristic, LeafSizeHaltingHeuristic, FraigTransformationHeuristic
from lec import LogicalEquivalenceChecking
from util import WorkPath, date_now, Solvers


if __name__ == '__main__':
    root_path = WorkPath('aiger')

    LogicalEquivalenceChecking(
        left_scheme_file=root_path.to_file('dadda12x12.aag'),
        right_scheme_file=root_path.to_file('karatsuba12x12.aag'),
        branching_heuristic=DegreeBranchingHeuristic(),
        halt_heuristic=LeafSizeHaltingHeuristic(
            leafs_size_lower_bound=4450,
            max_depth=8,
            reduction_conflicts=300
        ),
        transformation_heuristics=FraigTransformationHeuristic(),
        work_path=WorkPath('out').to_path(date_now()),
        worker_solver=Solvers.ROKK_LRB
    ).solve_using_processes(
        number_of_executor_workers=16
    )
