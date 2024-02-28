from algorithm import CubeAndConquer
from util import WorkPath
from scheme import AAGLEC
from datetime import datetime


def date_now() -> str:
    return datetime.today().strftime("%Y.%m.%d-%H:%M:%S")


if __name__ == '__main__':
    root_path = WorkPath('aiger')
    dadda = AAGLEC(
        left_scheme_from_file=root_path.to_file('dadda3x3.aag'),
        right_scheme_from_file=root_path.to_file('karatsuba3x3.aag')
    ).get_scheme_till_xor()

    graph_size_limit = 40
    reduce_limit = 70

    cnc = CubeAndConquer(dadda, graph_size_limit, reduce_limit)
    work_path = WorkPath('out').to_path(date_now() + '-' + str(graph_size_limit) + '-' + str(reduce_limit))
    cnc.cube_and_conquer(work_path=work_path, max_workers=16)
