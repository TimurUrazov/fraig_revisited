from encoding import AAG, CNF, Propagation, Sweep, Reducer
from typing import List, Dict, Any
from copy import copy
from .heuristics import LookaheadBranchingHeuristics
from aig_substitutions import ConflictAssignment
from mpi4py.futures import MPIPoolExecutor
from mpi4py import MPI
from concurrent.futures import as_completed, ThreadPoolExecutor
from util import WorkPath, write_aag, write_cnf, solve_cnf_with_kissat, analyze_result, solve_cnf_with_rokk
from typing import Tuple
import json
from datetime import datetime


class CubeTreeVertex:
    def __init__(self, is_cutoff_leaf: bool, aag: AAG = None):
        self._left_child = None
        self._right_child = None
        self._is_cutoff_leaf = is_cutoff_leaf
        self._parent = None
        self._aag = aag
        self._branching_value = None
        self._edge_vertex = None
        self._constr = None
        self._is_leaf = True

    def add_child(self, branching_value: int, child_num: int):
        self._is_leaf = False
        if branching_value == 0:
            self._left_child = child_num
        else:
            self._right_child = child_num

    def add_parent(self, parent_num: int, branching_value: int, edge_vertex: int, c):
        self._parent = parent_num
        self._branching_value = branching_value
        self._edge_vertex = edge_vertex
        self._constr = c

    def get_aag(self):
        return self._aag

    @property
    def parent(self) -> int:
        return self._parent

    @property
    def is_leaf(self) -> bool:
        return self._is_leaf

    @property
    def branching_value(self) -> int:
        return self._branching_value

    @property
    def edge_vertex(self) -> int:
        return self._edge_vertex

    def get_constraints(self):
        return self._constr


class CubeTree:
    def __init__(self):
        self._cutoff_leaf_num = 0
        self._tree_size = 0
        self._tree = {self._tree_size: CubeTreeVertex(False)}
        self._leaf_size = 1

    def add_leaf(self, current_vertex: int, parent_num: int, branching_value: int, is_cutoff_leaf: bool, vertex: int, aag: AAG = None, c = None):
        self._tree_size += 1
        if self._tree[parent_num].is_leaf:
            self._leaf_size -= 1
        self._tree[parent_num].add_child(branching_value, current_vertex)
        self._tree[current_vertex] = CubeTreeVertex(is_cutoff_leaf, aag)
        self._tree[current_vertex].add_parent(parent_num, branching_value, vertex, c)
        self._leaf_size += 1
        if is_cutoff_leaf:
            self._cutoff_leaf_num += 1

    def cutoff_leaf_percentage(self) -> float:
        return self._cutoff_leaf_num * 100 / self._leaf_size

    def get_tree_vertex(self, tree_vertex_num: int) -> CubeTreeVertex:
        return self._tree[tree_vertex_num]


class CubeAndConquer:
    def __init__(self, aag_lec: AAG, graph_size_limit: int, reduce_limit: int):
        self._cube_tree = CubeTree()
        self._aag = aag_lec
        self._path_to_graphs = None
        self._path_to_cubes = None
        self._path_to_error = None
        self._graph_size_limit = graph_size_limit
        self._path_to_logfile = None
        self._reduce_limit = reduce_limit

    # This function makes decision about which vertex to assign and makes propagation
    # The function status, branching value or nothing and graph or nothing
    # Statuses:
    # 0 - graph
    # 1 - conflict (cutoff leaf)
    # 2 - solved task
    # 3 - cube
    # 4 - error
    def _single_propagation(self, vertex: int) -> List[Tuple[int, int, int]]:
        path_to_aag = self._path_to_graphs.to_file(str(vertex)) + ".aag"
        graph = AAG(from_file=path_to_aag)

        info = []

        try:
            branching_vertex = LookaheadBranchingHeuristics().branch(copy(graph))
        except Exception as e:
            aag_data = graph.get_data()
            aag_data.add_comment("lookahead exception", str(e))
            write_aag(
                aag_data.source(),
                self._path_to_error.to_file(str(vertex))
            )
            info.append((4, 0, None))
            return info

        if branching_vertex != -1:
            for v_to_assign in [0, 1]:
                child_number = 2 * vertex + 1 + v_to_assign
                try:
                    propagated = Propagation(
                        aag=copy(graph),
                        constraints=set(),
                        conflicts_limit=100
                    ).propagate(node_number=branching_vertex, value_to_assign=v_to_assign)

                    sweep = Sweep(propagated).sweep()

                    data = sweep.get_data()
                    if data.n_of_and_gates + data.n_of_inputs < self._reduce_limit:
                        Reducer(aag=sweep, constraints=set()).reduce(conflict_limit=1500)

                        propagated = Sweep(sweep).sweep()

                        data = propagated.get_data()

                    if data.n_of_and_gates + data.n_of_inputs < self._graph_size_limit:
                        cnf = propagated.to_cnf_with_constraints(
                            filename=self._path_to_cubes.to_file(str(child_number))
                        ).get_data().source()
                        write_cnf(
                            cnf, self._path_to_cubes.to_file(str(child_number))
                        )
                        info.append((3, branching_vertex, child_number))
                    else:
                        write_aag(
                            propagated.get_data().source(),
                            self._path_to_graphs.to_file(str(child_number))
                        )
                        info.append((0, branching_vertex, child_number))
                except ConflictAssignment:
                    info.append((1, branching_vertex, child_number))
                except Exception as e:
                    aag_data = graph.get_data()
                    aag_data.add_comment("exception", str(e))
                    aag_data.add_comment("branching vertex", str(branching_vertex))
                    aag_data.add_comment("branching value", str(v_to_assign))
                    write_aag(
                        aag_data.source(),
                        self._path_to_error.to_file(str(vertex))
                    )
                    info.append((4, 1, None))
        else:
            info.append((2, None, None))

        return info

    def _cube_solution(self, vertex: int) -> Tuple[int, int]:
        cnf = CNF(from_file=self._path_to_cubes.to_file(str(vertex)) + '.cnf')
        cnf_data = cnf.get_data()
        try:
            str_cnf = cnf_data.source(supplements=([], [cnf.get_data().get_outputs()]))
            result = solve_cnf_with_kissat(str_cnf)
            answer, _, conflicts = analyze_result(result)
            print(conflicts)
            if answer == "SAT":
                return 1, conflicts
            if answer == "UNSAT":
                return 0, conflicts
        except Exception as e:
            cnf_data.add_comment("cnf exception: ", str("Sd"))
            write_cnf(
                cnf_data.source(),
                self._path_to_error.to_file(str(vertex))
            )
            return 2, 0

    def _write(self, string: str, filepath: str):
        with open(filepath, 'a+') as handle:
            handle.write(string)
        return self

    def _format(self, obj: Dict[str, Any], filepath: str):
        return self._write(f'{json.dumps(obj)}\n', filepath)

    @staticmethod
    def date_now() -> str:
        return datetime.today().strftime("%Y.%m.%d-%H:%M:%S")

    def cube_and_conquer(self, work_path: WorkPath, max_workers: int = None):
        self._path_to_graphs = work_path.to_path('graphs')
        self._path_to_cubes = work_path.to_path('cubes')
        self._path_to_error = work_path.to_path('errors')
        self._path_to_logfile = work_path.to_path('logs').to_file(CubeAndConquer.date_now())
        initial_graph_file_path = self._path_to_graphs.to_file(str(0))
        write_aag(self._aag.get_data().source(), initial_graph_file_path)
        graph_numbers = [0]

        cutoff_leafs = 0

        number_of_sat = 0
        number_of_unsat = 0
        number_of_cnf_errors = 0
        number_of_lookahead_errors = 0
        number_of_branching_errors = 0

        number_of_sat_conflict = 0
        number_of_unsat_conflict = 0

        cube_numbers = []

        propagations = 0

        max_workers = max(1, max_workers - 1)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while len(graph_numbers) > 0:
                futures = []

                for graph_number in graph_numbers:
                    futures.append(
                        executor.submit(
                            self._single_propagation, vertex=graph_number
                        )
                    )

                current_graphs_numbers = []

                for future in as_completed(futures):
                    propagations += 1
                    if propagations % 1000 == 0:
                        self._format({
                            'propagations': propagations,
                        }, self._path_to_logfile)
                    info = future.result()
                    for status, branching_vertex, graph_number in info:
                        if status == 0:
                            self._cube_tree.add_leaf(
                                graph_number, (graph_number - 1) // 2, 1 - graph_number % 2, False, branching_vertex
                            )
                            current_graphs_numbers.append(graph_number)
                        elif status == 1:
                            self._cube_tree.add_leaf(
                                graph_number, (graph_number - 1) // 2, 1 - graph_number % 2, True, branching_vertex
                            )
                            cutoff_leafs += 1
                        elif status == 2:
                            pass
                        elif status == 3:
                            self._cube_tree.add_leaf(
                                graph_number, (graph_number - 1) // 2, 1 - graph_number % 2, False, branching_vertex
                            )
                            cube_numbers.append(graph_number)
                        else:
                            if branching_vertex == 0:
                                number_of_lookahead_errors += 1
                            else:
                                number_of_branching_errors += 1
                graph_numbers = current_graphs_numbers

            futures = []

            for cube_number in cube_numbers:
                futures.append(
                    executor.submit(
                        self._cube_solution, vertex=cube_number
                    )
                )

            cnfs = 0

            for future in as_completed(futures):
                status, conflict = future.result()
                cnfs += 1
                if cnfs % 1000 == 0:
                    self._format({
                        'cnfs': cnfs,
                    }, self._path_to_logfile)
                if status == 1:
                    number_of_sat += 1
                    number_of_sat_conflict += conflict
                elif status == 0:
                    number_of_unsat += 1
                    number_of_unsat_conflict += conflict
                else:
                    number_of_cnf_errors += 1

        self._format({
            'sats': number_of_sat,
            'unsats': number_of_unsat,
            'number_of_sat_conflict': number_of_sat_conflict,
            'number_of_unsat_conflict': number_of_unsat_conflict,
            'cutoff leafs': cutoff_leafs,
            'number of cubes': cnfs,
            'number of propagations': propagations
        }, self._path_to_logfile)
