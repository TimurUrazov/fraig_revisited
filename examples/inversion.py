from util import WorkPath, date_now, Solvers
from encoding import AAG
from algorithm import DegreeBranchingHeuristic, LeafSizeHaltingHeuristic, FraigTransformationHeuristic, HaltingHeuristic
from inversion import Inversion


def simple_inversion_geffe():
    root_path = WorkPath('aiger')

    geffe64 = AAG(from_file=root_path.to_file('geffe64.aag'))

    Inversion(
        aag=geffe64,
        output_bits_str='0000011100100011101010110011010110001010010101110111101011010001',
        branching_heuristic=DegreeBranchingHeuristic(),
        halt_heuristic=LeafSizeHaltingHeuristic(
            leafs_size_lower_bound=1000,
            max_depth=4,
            reduction_conflicts=300
        ),
        transformation_heuristics=FraigTransformationHeuristic(),
        work_path=WorkPath('out').to_path(date_now()),
        executor_workers=8,
        worker_solver=Solvers.KISSAT_2023
    ).cnc()


def inversion_a5():
    root_path = WorkPath('aiger')

    a5_1 = AAG(from_file=root_path.to_file('A5_1_64.aag'))

    Inversion(
        aag=a5_1,
        output_bits_str='0111110000101010100101001110000110100010001010000110001100000010',
        branching_heuristic=DegreeBranchingHeuristic(),
        halt_heuristic=HaltingHeuristic(
            max_depth=30,
            reduction_conflicts=300
        ),
        transformation_heuristics=FraigTransformationHeuristic(),
        work_path=WorkPath('out').to_path(date_now()),
        executor_workers=8,
        worker_solver=Solvers.KISSAT_2023
    ).cnc()
