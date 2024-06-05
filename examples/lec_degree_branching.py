from algorithm import DegreeBranchingHeuristic, LeafSizeHaltingHeuristic, HaltingHeuristic, FraigTransformationHeuristic
from lec import LogicalEquivalenceChecking
from util import WorkPath, date_now, Solvers


def parallel_cnc_lec_with_different_depth(
        solver: Solvers,
        number_of_workers: int,
        left_scheme_file: str,
        right_scheme_file: str
):
    for depth in [5, 8, 10]:
        LogicalEquivalenceChecking(
            left_scheme_file=left_scheme_file,
            right_scheme_file=right_scheme_file,
            branching_heuristic=DegreeBranchingHeuristic(),
            halt_heuristic=LeafSizeHaltingHeuristic(
                leafs_size_lower_bound=2300,
                max_depth=depth,
                reduction_conflicts=300
            ),
            transformation_heuristics=FraigTransformationHeuristic(),
            work_path=WorkPath('out').to_path(date_now()),
            worker_solver=solver
        ).solve_using_processes(
            number_of_executor_workers=number_of_workers
        )


def cnc_estimate_mpi_2_xor_input_decomposition():
    root_path = WorkPath('aiger')

    LogicalEquivalenceChecking(
        left_scheme_file=root_path.to_file('dadda12x12.aag'),
        right_scheme_file=root_path.to_file('wallace12x12.aag'),
        branching_heuristic=DegreeBranchingHeuristic(),
        halt_heuristic=HaltingHeuristic(
            max_depth=10,
            reduction_conflicts=300
        ),
        transformation_heuristics=FraigTransformationHeuristic(),
        work_path=WorkPath('out').to_path(date_now()),
        worker_solver=Solvers.ROKK_LRB
    ).estimate_using_mpi(
        use_input_decomposition=True,
        decomposition_limit=1000,
        length_decompose='2',
        shuffle_inputs=False
    )


def parallel_cnc_lec_with_different_schemes():
    executor_workers = 8
    root_path = WorkPath('aiger')

    d_v_c_12 = (root_path.to_file('dadda12x12.aag'), root_path.to_file('column12x12.aag'))

    d_v_k_12 = (root_path.to_file('dadda12x12.aag'), root_path.to_file('karatsuba12x12.aag'))

    d_v_w_12 = (root_path.to_file('dadda12x12.aag'), root_path.to_file('wallace12x12.aag'))

    c_v_k_12 = (root_path.to_file('karatsuba12x12.aag'),root_path.to_file('column12x12.aag'))

    c_v_w_12 = (root_path.to_file('wallace12x12.aag'), root_path.to_file('column12x12.aag'))

    k_v_w_12 = (root_path.to_file('karatsuba12x12.aag'), root_path.to_file('wallace12x12.aag'))

    schemes = [d_v_c_12, d_v_k_12, d_v_w_12, c_v_k_12, c_v_w_12, k_v_w_12]

    for left_scheme, right_scheme in schemes:
        parallel_cnc_lec_with_different_depth(Solvers.ROKK_LRB, executor_workers, left_scheme, right_scheme)


def parallel_cnc_lec_with_different_solvers():
    executor_workers = 8
    root_path = WorkPath('aiger')

    c_v_w_12 = (root_path.to_file('wallace12x12.aag'), root_path.to_file('column12x12.aag'))

    for solver in Solvers:
        parallel_cnc_lec_with_different_depth(solver, executor_workers, *c_v_w_12)


def parallel_cnc_lec_with_different_number_of_threads():
    thread_nums = [8, 16, 32, 64]

    root_path = WorkPath('aiger')

    d_v_k_12 = (root_path.to_file('dadda12x12.aag'), root_path.to_file('karatsuba12x12.aag'))

    for thread_num in thread_nums:
        parallel_cnc_lec_with_different_depth(Solvers.ROKK_LRB, thread_num, *d_v_k_12)


def parallel_cnc_lec_with_leaf_size(
        solver: Solvers,
        number_of_workers: int,
        depth: int,
        size_limit: int,
        left_scheme_file: str,
        right_scheme_file: str
):
    LogicalEquivalenceChecking(
        left_scheme_file=left_scheme_file,
        right_scheme_file=right_scheme_file,
        branching_heuristic=DegreeBranchingHeuristic(),
        halt_heuristic=LeafSizeHaltingHeuristic(
            leafs_size_lower_bound=size_limit,
            max_depth=depth,
            reduction_conflicts=300
        ),
        transformation_heuristics=FraigTransformationHeuristic(),
        work_path=WorkPath('out').to_path(date_now()),
        worker_solver=solver
    ).solve_using_processes(
        number_of_executor_workers=number_of_workers
    )


def parallel_cnc_lec(
        solver: Solvers,
        number_of_workers: int,
        depth: int,
        left_scheme_file: str,
        right_scheme_file: str
):
    LogicalEquivalenceChecking(
        left_scheme_file=left_scheme_file,
        right_scheme_file=right_scheme_file,
        branching_heuristic=DegreeBranchingHeuristic(),
        halt_heuristic=HaltingHeuristic(
            max_depth=depth,
            reduction_conflicts=300
        ),
        transformation_heuristics=FraigTransformationHeuristic(),
        work_path=WorkPath('out').to_path(date_now()),
        worker_solver=solver
    ).solve_using_processes(
        number_of_executor_workers=number_of_workers
    )


def solve_cnc_lec_c_v_k_12_rokk_d8():
    root_path = WorkPath('aiger')

    c_v_k_12 = (root_path.to_file('column12x12.aag'), root_path.to_file('karatsuba12x12.aag'))

    parallel_cnc_lec(Solvers.ROKK_LRB, 8, 8, *c_v_k_12)


def solve_cnc_lec_c_v_k_12_n4600_rokk_d8():
    root_path = WorkPath('aiger')

    c_v_k_12 = (root_path.to_file('column12x12.aag'), root_path.to_file('karatsuba12x12.aag'))

    parallel_cnc_lec_with_leaf_size(Solvers.ROKK_LRB, 8, 8, 4600, *c_v_k_12)


def solve_cnc_lec_c_v_k_12_n4600_rokk():
    root_path = WorkPath('aiger')

    c_v_k_12 = (root_path.to_file('column12x12.aag'), root_path.to_file('karatsuba12x12.aag'))

    parallel_cnc_lec_with_leaf_size(Solvers.ROKK_LRB, 8, None, 4600, *c_v_k_12)


def solve_cnc_lec_c_v_k_12_n4600_rokk_d16():
    root_path = WorkPath('aiger')

    c_v_k_12 = (root_path.to_file('column12x12.aag'), root_path.to_file('karatsuba12x12.aag'))

    parallel_cnc_lec_with_leaf_size(Solvers.ROKK_LRB, 8, 16, 4600, *c_v_k_12)


def solve_cnc_lec_c_v_w_12_n2850_rokk_d8():
    root_path = WorkPath('aiger')

    c_v_w_12 = (root_path.to_file('column12x12.aag'), root_path.to_file('wallace12x12.aag'))

    parallel_cnc_lec_with_leaf_size(Solvers.ROKK_LRB, 8, 8, 2850, *c_v_w_12)


def solve_cnc_lec_c_v_w_12_rokk_d8():
    root_path = WorkPath('aiger')

    c_v_w_12 = (root_path.to_file('column12x12.aag'), root_path.to_file('wallace12x12.aag'))

    parallel_cnc_lec(Solvers.ROKK_LRB, 8, 8, *c_v_w_12)


def solve_cnc_lec_c_v_w_12_n2850_rokk():
    root_path = WorkPath('aiger')

    c_v_w_12 = (root_path.to_file('column12x12.aag'), root_path.to_file('wallace12x12.aag'))

    parallel_cnc_lec_with_leaf_size(Solvers.ROKK_LRB, 8, None, 2850, *c_v_w_12)


def solve_cnc_lec_d_v_k_12_n4450_rokk_d8():
    root_path = WorkPath('aiger')

    d_v_k_12 = (root_path.to_file('dadda12x12.aag'), root_path.to_file('karatsuba12x12.aag'))

    parallel_cnc_lec_with_leaf_size(Solvers.ROKK_LRB, 8, 8, 4450, *d_v_k_12)


def solve_cnc_lec_d_v_k_12_n4450_rokk_d12():
    root_path = WorkPath('aiger')

    d_v_k_12 = (root_path.to_file('dadda12x12.aag'), root_path.to_file('karatsuba12x12.aag'))

    parallel_cnc_lec_with_leaf_size(Solvers.ROKK_LRB, 8, 12, 4450, *d_v_k_12)


def solve_cnc_lec_d_v_k_12_rokk_d12():
    root_path = WorkPath('aiger')

    d_v_k_12 = (root_path.to_file('dadda12x12.aag'), root_path.to_file('karatsuba12x12.aag'))

    parallel_cnc_lec(Solvers.ROKK_LRB, 8, 12, *d_v_k_12)


def solve_cnc_lec_d_v_k_12_rokk_d8():
    root_path = WorkPath('aiger')

    d_v_k_12 = (root_path.to_file('dadda12x12.aag'), root_path.to_file('karatsuba12x12.aag'))

    parallel_cnc_lec(Solvers.ROKK_LRB, 8, 8, *d_v_k_12)


def solve_cnc_lec_d_v_k_12_n4450_rokk():
    root_path = WorkPath('aiger')

    d_v_k_12 = (root_path.to_file('dadda12x12.aag'), root_path.to_file('karatsuba12x12.aag'))

    parallel_cnc_lec_with_leaf_size(Solvers.ROKK_LRB, 8, None, 4450, *d_v_k_12)


def solve_cnc_lec_d_v_k_12_n4450_rokk_d16():
    root_path = WorkPath('aiger')

    d_v_k_12 = (root_path.to_file('dadda12x12.aag'), root_path.to_file('karatsuba12x12.aag'))

    parallel_cnc_lec_with_leaf_size(Solvers.ROKK_LRB, 8, 16, 4450, *d_v_k_12)


def solve_cnc_lec_d_v_k_12_rokk_d16():
    root_path = WorkPath('aiger')

    d_v_k_12 = (root_path.to_file('dadda12x12.aag'), root_path.to_file('karatsuba12x12.aag'))

    parallel_cnc_lec(Solvers.ROKK_LRB, 8, 16, *d_v_k_12)


def solve_cnc_lec_w_v_k_12_n4450_rokk_d8():
    root_path = WorkPath('aiger')

    w_v_k_12 = (root_path.to_file('wallace12x12.aag'), root_path.to_file('karatsuba12x12.aag'))

    parallel_cnc_lec_with_leaf_size(Solvers.ROKK_LRB, 8, 8, 4450, *w_v_k_12)


def solve_cnc_lec_w_v_k_12_n4450_rokk_d12():
    root_path = WorkPath('aiger')

    w_v_k_12 = (root_path.to_file('wallace12x12.aag'), root_path.to_file('karatsuba12x12.aag'))

    parallel_cnc_lec_with_leaf_size(Solvers.ROKK_LRB, 8, 12, 4450, *w_v_k_12)


def solve_cnc_lec_w_v_k_12_rokk_d12():
    root_path = WorkPath('aiger')

    w_v_k_12 = (root_path.to_file('wallace12x12.aag'), root_path.to_file('karatsuba12x12.aag'))

    parallel_cnc_lec(Solvers.ROKK_LRB, 8, 12, *w_v_k_12)


def solve_cnc_lec_w_v_k_12_rokk_d8():
    root_path = WorkPath('aiger')

    w_v_k_12 = (root_path.to_file('wallace12x12.aag'), root_path.to_file('karatsuba12x12.aag'))

    parallel_cnc_lec(Solvers.ROKK_LRB, 8, 8, *w_v_k_12)


def solve_cnc_lec_w_v_k_12_n4450_rokk():
    root_path = WorkPath('aiger')

    w_v_k_12 = (root_path.to_file('wallace12x12.aag'), root_path.to_file('karatsuba12x12.aag'))

    parallel_cnc_lec_with_leaf_size(Solvers.ROKK_LRB, 8, None, 4450, *w_v_k_12)


def solve_cnc_lec_w_v_k_12_n4450_rokk_d16():
    root_path = WorkPath('aiger')

    w_v_k_12 = (root_path.to_file('wallace12x12.aag'), root_path.to_file('karatsuba12x12.aag'))

    parallel_cnc_lec_with_leaf_size(Solvers.ROKK_LRB, 8, 16, 4450, *w_v_k_12)


def solve_cnc_lec_w_v_k_12_rokk_d16():
    root_path = WorkPath('aiger')

    w_v_k_12 = (root_path.to_file('wallace12x12.aag'), root_path.to_file('karatsuba12x12.aag'))

    parallel_cnc_lec(Solvers.ROKK_LRB, 8, 16, *w_v_k_12)


def solve_dvw_14_with_kissat_2022():
    root_path = WorkPath('aiger')
    LogicalEquivalenceChecking(
        left_scheme_file=root_path.to_file('karatsuba14x14.aag'),
        right_scheme_file=root_path.to_file('column14x14.aag'),
        branching_heuristic=DegreeBranchingHeuristic(),
        halt_heuristic=LeafSizeHaltingHeuristic(
            leafs_size_lower_bound=7300,
            max_depth=16,
            reduction_conflicts=300
        ),
        transformation_heuristics=FraigTransformationHeuristic(),
        work_path=WorkPath('out').to_path(date_now()),
        worker_solver=Solvers.KISSAT_2022
    ).solve_using_processes(number_of_executor_workers=24)
