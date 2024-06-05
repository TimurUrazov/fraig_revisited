from algorithm import DegreeBranchingHeuristic, LeafSizeHaltingHeuristic, FraigTransformationHeuristic
from lec import LogicalEquivalenceChecking
from util import WorkPath, date_now, Solvers


def solve_cubes_created_with_march(solver: Solvers, cnf_file: str, cubes_file: str, executor_workers: int):
    root_path = WorkPath('aiger')
    path_to_cnf = root_path.to_file(cnf_file, 'CnC_cubes_for_LEC_mults')
    path_to_cubes = root_path.to_file(cubes_file, 'CnC_cubes_for_LEC_mults')
    LogicalEquivalenceChecking(
        left_scheme_file=root_path.to_file('dadda12x12.aag'),
        right_scheme_file=root_path.to_file('wallace12x12.aag'),
        branching_heuristic=DegreeBranchingHeuristic(),
        halt_heuristic=LeafSizeHaltingHeuristic(
            max_depth=8,
            leafs_size_lower_bound=2650,
            reduction_conflicts=300
        ),
        transformation_heuristics=FraigTransformationHeuristic(),
        work_path=WorkPath('out').to_path(date_now()),
        worker_solver=solver
    ).solve_march_cubes(
        path_to_cnf_file=path_to_cnf,
        path_to_cubes_file=path_to_cubes,
        number_of_executor_workers=executor_workers
    )

    
def solve_c_v_k_12_with_default_cnc_cubes_and_kissat():
    solve_cubes_created_with_march(
        Solvers.KISSAT_2022, 
        'lec_CvK_12.cnf', 
        'lec_CvK_12_cubes_default.txt', 
        8
    )


def solve_c_v_k_12_with_d8_cnc_cubes_and_kissat():
    solve_cubes_created_with_march(
        Solvers.KISSAT_2022, 
        'lec_CvK_12.cnf', 
        'lec_CvK_12_cubes_d8.txt', 
        8
    )


def solve_c_v_k_12_with_n4600_cnc_cubes_and_kissat():
    solve_cubes_created_with_march(
        Solvers.KISSAT_2022, 
        'lec_CvK_12.cnf', 
        'lec_CvK_12_cubes_n4600.txt', 
        8
    )
    
    
def solve_c_v_w_12_with_default_cnc_cubes_and_kissat():
    solve_cubes_created_with_march(
        Solvers.KISSAT_2022, 
        'lec_CvW_12.cnf', 
        'lec_CvW_12_cubes_default.txt', 
        8
    )
    
    
def solve_c_v_w_12_with_d8_cnc_cubes_and_kissat():
    solve_cubes_created_with_march(
        Solvers.KISSAT_2022, 
        'lec_CvW_12.cnf', 
        'lec_CvW_12_cubes_d8.txt', 
        8
    )
    
    
def solve_c_v_w_12_with_n2850_cnc_cubes_and_kissat():
    solve_cubes_created_with_march(
        Solvers.KISSAT_2022, 
        'lec_CvW_12.cnf', 
        'lec_CvW_12_cubes_n2850.txt', 
        8
    )


def solve_d_v_c_12_with_default_cnc_cubes_and_kissat():
    solve_cubes_created_with_march(
        Solvers.KISSAT_2022, 
        'lec_DvC_12.cnf', 
        'lec_DvC_12_cubes_default.txt', 
        8
    )


def solve_d_v_c_12_with_d8_cnc_cubes_and_kissat():
    solve_cubes_created_with_march(
        Solvers.KISSAT_2022, 
        'lec_DvC_12.cnf', 
        'lec_DvC_12_cubes_d8.txt',  
        8
    )


def solve_d_v_c_12_with_n2800_cnc_cubes_and_kissat():
    solve_cubes_created_with_march(
        Solvers.KISSAT_2022, 
        'lec_DvC_12.cnf', 
        'lec_DvC_12_cubes_n2800.txt',  
        8
    )
    
    
def solve_d_v_k_12_with_default_cnc_cubes_and_kissat():
    solve_cubes_created_with_march(
        Solvers.KISSAT_2022, 
        'lec_DvK_12.cnf', 
        'lec_DvK_12_cubes_default.txt',  
        8
    )


def solve_d_v_k_12_with_d8_cnc_cubes_and_kissat():
    solve_cubes_created_with_march(
        Solvers.KISSAT_2022, 
        'lec_DvK_12.cnf', 
        'lec_DvK_12_cubes_d8.txt',  
        8
    )
    
    
def solve_d_v_k_12_with_n4450_cnc_cubes_and_kissat():
    solve_cubes_created_with_march(
        Solvers.KISSAT_2022, 
        'lec_DvK_12.cnf', 
        'lec_DvK_12_cubes_n4450.txt',  
        8
    )


def solve_d_v_w_12_with_default_cnc_cubes_and_kissat():
    solve_cubes_created_with_march(
        Solvers.KISSAT_2022, 
        'lec_DvW_12.cnf', 
        'lec_DvW_12_cubes_default.txt',  
        8
    )
    
    
def solve_d_v_w_12_with_d8_cnc_cubes_and_kissat():
    solve_cubes_created_with_march(
        Solvers.KISSAT_2022, 
        'lec_DvW_12.cnf', 
        'lec_DvW_12_cubes_d8.txt',  
        8
    )
    

def solve_d_v_w_12_with_n2650_cnc_cubes_and_kissat():
    solve_cubes_created_with_march(
        Solvers.KISSAT_2022, 
        'lec_DvW_12.cnf', 
        'lec_DvW_12_cubes_n2650.txt',  
        8
    )


def solve_k_v_w_12_with_default_cnc_cubes_and_kissat():
    solve_cubes_created_with_march(
        Solvers.KISSAT_2022, 
        'lec_KvW_12.cnf', 
        'lec_KvW_12_cubes_default.txt',  
        8
    )


def solve_k_v_w_12_with_d8_cnc_cubes_and_kissat():
    solve_cubes_created_with_march(
        Solvers.KISSAT_2022, 
        'lec_KvW_12.cnf', 
        'lec_KvW_12_cubes_d8.txt',  
        8
    )

    
def solve_k_v_w_12_with_n4550_cnc_cubes_and_kissat():
    solve_cubes_created_with_march(
        Solvers.KISSAT_2022, 
        'lec_KvW_12.cnf', 
        'lec_KvW_12_cubes_n4550.txt',  
        8
    )


def solve_c_v_k_12_with_default_cnc_cubes_and_rokk_lrb():
    solve_cubes_created_with_march(
        Solvers.ROKK_LRB,
        'lec_CvK_12.cnf',
        'lec_CvK_12_cubes_default.txt',
        8
    )


def solve_c_v_k_12_with_d8_cnc_cubes_and_rokk_lrb():
    solve_cubes_created_with_march(
        Solvers.ROKK_LRB,
        'lec_CvK_12.cnf',
        'lec_CvK_12_cubes_d8.txt',
        8
    )


def solve_c_v_k_12_with_n4600_cnc_cubes_and_rokk_lrb():
    solve_cubes_created_with_march(
        Solvers.ROKK_LRB,
        'lec_CvK_12.cnf',
        'lec_CvK_12_cubes_n4600.txt',
        8
    )


def solve_c_v_w_12_with_default_cnc_cubes_and_rokk_lrb():
    solve_cubes_created_with_march(
        Solvers.ROKK_LRB,
        'lec_CvW_12.cnf',
        'lec_CvW_12_cubes_default.txt',
        8
    )


def solve_c_v_w_12_with_d8_cnc_cubes_and_rokk_lrb():
    solve_cubes_created_with_march(
        Solvers.ROKK_LRB,
        'lec_CvW_12.cnf',
        'lec_CvW_12_cubes_d8.txt',
        8
    )


def solve_c_v_w_12_with_n2850_cnc_cubes_and_rokk_lrb():
    solve_cubes_created_with_march(
        Solvers.ROKK_LRB,
        'lec_CvW_12.cnf',
        'lec_CvW_12_cubes_n2850.txt',
        8
    )


def solve_d_v_c_12_with_default_cnc_cubes_and_rokk_lrb():
    solve_cubes_created_with_march(
        Solvers.ROKK_LRB,
        'lec_DvC_12.cnf',
        'lec_DvC_12_cubes_default.txt',
        8
    )


def solve_d_v_c_12_with_d8_cnc_cubes_and_rokk_lrb():
    solve_cubes_created_with_march(
        Solvers.ROKK_LRB,
        'lec_DvC_12.cnf',
        'lec_DvC_12_cubes_d8.txt',
        8
    )


def solve_d_v_c_12_with_n2800_cnc_cubes_and_rokk_lrb():
    solve_cubes_created_with_march(
        Solvers.ROKK_LRB,
        'lec_DvC_12.cnf',
        'lec_DvC_12_cubes_n2800.txt',
        8
    )


def solve_d_v_k_12_with_default_cnc_cubes_and_rokk_lrb():
    solve_cubes_created_with_march(
        Solvers.ROKK_LRB,
        'lec_DvK_12.cnf',
        'lec_DvK_12_cubes_default.txt',
        8
    )


def solve_d_v_k_12_with_d8_cnc_cubes_and_rokk_lrb():
    solve_cubes_created_with_march(
        Solvers.ROKK_LRB,
        'lec_DvK_12.cnf',
        'lec_DvK_12_cubes_d8.txt',
        8
    )


def solve_d_v_k_12_with_n4450_cnc_cubes_and_rokk_lrb():
    solve_cubes_created_with_march(
        Solvers.ROKK_LRB,
        'lec_DvK_12.cnf',
        'lec_DvK_12_cubes_n4450.txt',
        8
    )


def solve_d_v_w_12_with_default_cnc_cubes_and_rokk_lrb():
    solve_cubes_created_with_march(
        Solvers.ROKK_LRB,
        'lec_DvW_12.cnf',
        'lec_DvW_12_cubes_default.txt',
        8
    )


def solve_d_v_w_12_with_d8_cnc_cubes_and_rokk_lrb():
    solve_cubes_created_with_march(
        Solvers.ROKK_LRB,
        'lec_DvW_12.cnf',
        'lec_DvW_12_cubes_d8.txt',
        8
    )


def solve_d_v_w_12_with_n2650_cnc_cubes_and_rokk_lrb():
    solve_cubes_created_with_march(
        Solvers.ROKK_LRB,
        'lec_DvW_12.cnf',
        'lec_DvW_12_cubes_n2650.txt',
        8
    )


def solve_k_v_w_12_with_default_cnc_cubes_and_rokk_lrb():
    solve_cubes_created_with_march(
        Solvers.ROKK_LRB,
        'lec_KvW_12.cnf',
        'lec_KvW_12_cubes_default.txt',
        8
    )


def solve_k_v_w_12_with_d8_cnc_cubes_and_rokk_lrb():
    solve_cubes_created_with_march(
        Solvers.ROKK_LRB,
        'lec_KvW_12.cnf',
        'lec_KvW_12_cubes_d8.txt',
        8
    )


def solve_k_v_w_12_with_n4550_cnc_cubes_and_rokk_lrb():
    solve_cubes_created_with_march(
        Solvers.ROKK_LRB,
        'lec_KvW_12.cnf',
        'lec_KvW_12_cubes_n4550.txt',
        8
    )
