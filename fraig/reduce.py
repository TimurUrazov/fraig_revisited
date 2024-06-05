from typing import Dict, Set, Tuple, List, Mapping
from encoding import CNF, Constraints
from util import solve_cnf_with_kissat_2023, analyze_result, Status
from random import getrandbits
from collections import defaultdict
from .graph import AndInverterGraph
from util import SatOracleReport


# This class makes random simulation
class Simulator:
    def __init__(self, graph):
        self._graph = graph
        self._random_numbers = {}

    # Here we implement random simulation
    # number_of_samples is the number of random inputs
    def simulate(self, number_of_samples: int, include_complementary: bool = True) -> Dict[int, List[int]]:
        sim_vec_to_class = defaultdict(list)
        ones = 2 ** number_of_samples - 1

        def simulate_node(node_number: int):
            if node_number in self._random_numbers:
                return

            node = self._graph[node_number]

            if node.in_degree == 0:
                current_simulation_vector = getrandbits(number_of_samples)
                sim_vec_to_class[current_simulation_vector].append(node_number)
                self._random_numbers[node_number] = current_simulation_vector
                return

            assert node.in_degree == 2

            parents = node.parents

            parent_hashes = []

            for parent_num, negation in parents:
                if parent_num not in self._random_numbers:
                    simulate_node(parent_num)
                parent_hashes.append(self._random_numbers[parent_num] ^ (ones if negation == 1 else 0))

            current_simulation_vector = parent_hashes[0] & parent_hashes[1]

            # Nodes can be equal up to negation
            sim_vec_to_class[current_simulation_vector].append(node_number)
            if include_complementary:
                sim_vec_to_class[current_simulation_vector ^ ones].append(node_number)

            self._random_numbers[node_number] = current_simulation_vector

        for node_num in self._graph:
            simulate_node(node_num)

        if include_complementary:
            sim_vecs_of_classes_to_del = set()

            for current_simulation in sim_vec_to_class:
                complementary_sim_vec = current_simulation ^ ones
                if complementary_sim_vec not in sim_vecs_of_classes_to_del and complementary_sim_vec in sim_vec_to_class:
                    sim_vecs_of_classes_to_del.add(current_simulation)

            for sim_vec_of_class_to_del in sim_vecs_of_classes_to_del:
                del sim_vec_to_class[sim_vec_of_class_to_del]

            # To reduce consequent solver calls and decrease time we can check whether node is parent or child
            # in hash equivalence class

        return sim_vec_to_class


# Here are equivalence classes for nodes which simulation vector hashes are proved to be equal
class EquivalenceClasses:
    def __init__(self):
        self._max_class_num = 0
        self._class_of_vertex_to_num_of_class = {}
        self._num_to_class = defaultdict(set)
        self._vertex_num_to_eq_mark_in_class = {}

    # Returns mapping from equivalence class number to vertices related to it
    def get_equivalence_classes(self) -> Mapping[int, Set[int]]:
        return self._num_to_class.copy()

    # Returns equivalence class marks which shows if vertices are equal or are equal up to negation
    def get_equivalence_marks(self) -> Mapping[int, bool]:
        return self._vertex_num_to_eq_mark_in_class.copy()

    # If one of nodes doesn't belong to any equivalence class, then we assign it to existent class with id
    # of class other node belongs to or create new with self._max_class_num id + 1.
    # self._vertex_num_to_eq_mark_in_class maps vertex to mark within a class:
    # Nodes have different marks within a class. If vertices are equal then their marks are the same.
    # If they are equal up to negation, then their marks are different.
    # self._class_of_vertex_to_num_of_class maps vertex to its number of class
    # self._num_to_class maps number of class to list of vertices within a class
    def add_pair_to_class(self, first: int, second: int, are_equivalent: bool):

        def fill_fields(first_node, second_node, are_equivalent_nodes):
            self._class_of_vertex_to_num_of_class[first_node] = self._class_of_vertex_to_num_of_class[second_node]
            self._num_to_class[self._class_of_vertex_to_num_of_class[second_node]].add(first_node)
            if are_equivalent_nodes:
                self._vertex_num_to_eq_mark_in_class[first_node] = self._vertex_num_to_eq_mark_in_class[second_node]
            else:
                self._vertex_num_to_eq_mark_in_class[first_node] = not self._vertex_num_to_eq_mark_in_class[second_node]

        if first in self._class_of_vertex_to_num_of_class:
            fill_fields(second, first, are_equivalent)
        elif second in self._class_of_vertex_to_num_of_class:
            fill_fields(first, second, are_equivalent)
        else:
            self._max_class_num += 1
            self._class_of_vertex_to_num_of_class[first] = self._max_class_num
            self._class_of_vertex_to_num_of_class[second] = self._max_class_num
            self._num_to_class[self._max_class_num].add(first)
            self._num_to_class[self._max_class_num].add(second)
            self._vertex_num_to_eq_mark_in_class[first] = True
            self._vertex_num_to_eq_mark_in_class[second] = are_equivalent

    # Check if vertices are already in equivalence class
    def belongs(self, first: int, second: int) -> bool:
        return first in self._class_of_vertex_to_num_of_class and second in self._class_of_vertex_to_num_of_class


# This class reduces aag
class Reducer:
    def __init__(
            self,
            and_inverter_graph: AndInverterGraph,
            num_of_simulate_samples: int = 64 * 127,
            include_complementary: bool = True,
            conflict_limit: int = 10000
    ):
        self._graph = and_inverter_graph
        self._num_of_simulate_samples = num_of_simulate_samples
        self._include_complementary = include_complementary
        self._conflict_limit = conflict_limit

    # The function builds constraints for node and its parents according to Tseitin transformation
    def _build_subgraph_constraints(self) -> Mapping[int, Constraints]:
        def negate_var(node_num: int, is_negated: int):
            if is_negated == 1:
                return -node_num
            return node_num

        constraints_and_inputs = {}

        for node_number in self._graph:
            node = self._graph[node_number]

            parents = node.parents

            assert node.in_degree % 2 == 0

            if node.in_degree == 0:
                constraints_and_inputs[node_number] = []
                continue

            var_output = node_number
            var_input1 = negate_var(*parents[0])
            var_input2 = negate_var(*parents[1])

            constraints_and_inputs[node_number] = [
                [var_output, -var_input1, -var_input2],
                [-var_output, var_input1],
                [-var_output, var_input2]
            ]

        return constraints_and_inputs

    # The function gets nodes from one cone and another and create constraints expressing xor of such cones
    @staticmethod
    def _create_xors_by_two_outputs(
            current_max_node_number: int, node_num_1: int, node_num_2: int, constraints: Constraints
    ) -> int:
        current_max_node_number += 1
        new_constraints = [[
            current_max_node_number,
            -node_num_1,
            -node_num_2
        ]]

        current_max_node_number += 1
        new_constraints.append([
            current_max_node_number,
            node_num_1,
            node_num_2
        ])

        current_max_node_number += 1
        new_constraints.append([
            current_max_node_number,
            -(current_max_node_number - 1),
            -(current_max_node_number - 2)
        ])

        for new_constraint in new_constraints:
            var_output = new_constraint[0]
            var_input1 = new_constraint[1]
            var_input2 = new_constraint[2]
            constraints.extend([
                [var_output, -var_input1, -var_input2],
                [-var_output, var_input1],
                [-var_output, var_input2]
            ])

        return current_max_node_number

    # The function just assign literals within constraints with minimal possible number (node_num_1
    # and node_num_2) are numbers of nodes in which cones end
    @staticmethod
    def _renumber_constraints(constraints: Constraints, node_num_1: int, node_num_2: int) -> Tuple[
        Constraints, int, int, int
    ]:
        new_constraints = []
        old_to_new_literal = {}
        new_literal_num = 0

        for constraint in constraints:
            new_constraint = []
            for literal in constraint:
                if abs(literal) not in old_to_new_literal:
                    new_literal_num += 1
                    old_to_new_literal[abs(literal)] = new_literal_num
                new_constraint.append(old_to_new_literal[abs(literal)] * (literal // abs(literal)))
            new_constraints.append(new_constraint)

        return new_constraints, new_literal_num, old_to_new_literal[node_num_1], old_to_new_literal[node_num_2]

    # The function checks if node_num_1 and node_num_2 are equal or equal up to negation
    # we encode xor of two cones and:
    # 1) check if nodes are functionally equivalent: fix xor as 1 and if this is unsatisfiable then
    # nodes are functionally equivalent
    # 2) check if nodes are not functionally equivalent: fix xor as 0 and if this is unsatisfiable
    # then nodes are functionally inequivalent (equal up to negation)
    def _find_equivalent_nodes(
            self,
            node_num_1: int,
            node_num_2: int,
            subgraph_constraints: Mapping[int, Constraints],
            equivalence_class: EquivalenceClasses,
            conflict_limit: int,
            sat_oracle_report: SatOracleReport
    ):
        constraints = []
        visited = set()

        def dfs(node_number: int):
            if node_number in visited:
                return
            visited.add(node_number)

            constraints.extend(subgraph_constraints[node_number])

            for node_num, _ in self._graph[node_number].parents:
                dfs(node_num)

        dfs(node_num_1)
        dfs(node_num_2)

        new_constraints, current_max_node_number, new_node_1_number, new_node_2_number = Reducer._renumber_constraints(
            constraints, node_num_1, node_num_2
        )

        max_node_number = Reducer._create_xors_by_two_outputs(
            current_max_node_number,
            new_node_1_number,
            new_node_2_number,
            new_constraints
        )

        # Check if nodes are functionally equivalent: fix xor as 1 and this is unsatisfiable
        new_constraints.append([max_node_number])

        result = solve_cnf_with_kissat_2023(
            cnf_str=CNF(from_clauses=new_constraints).get_data().source(),
            conflicts_limit=conflict_limit
        )

        report = analyze_result(result, 'checking if nodes are functionally equivalent', print_output=False)
        sat_oracle_report.update_report_with_status(report)

        if report.status == Status.UNSAT:
            equivalence_class.add_pair_to_class(node_num_1, node_num_2, True)

        if not self._include_complementary:
            return
        # Check if nodes are not functionally equivalent: fix xor as 0 and this is unsatisfiable
        new_constraints[-1] = [-max_node_number]

        result = solve_cnf_with_kissat_2023(
            cnf_str=CNF(from_clauses=new_constraints).get_data().source(),
            conflicts_limit=conflict_limit
        )
        report = analyze_result(result, 'checking if nodes are functionally inequivalent', print_output=False)
        sat_oracle_report.update_report_with_status(report)

        if report.status == Status.UNSAT:
            equivalence_class.add_pair_to_class(node_num_1, node_num_2, False)

    # The function merges equivalent nodes in graph deleting redundant nodes
    # conflict_limit is parameter passed to kissat
    def _merge_equivalent(self) -> SatOracleReport:
        sat_oracle_report = SatOracleReport()
        depth, _ = self._graph.get_depths()
        subgraph_constraints = self._build_subgraph_constraints()
        equivalence_classes = EquivalenceClasses()
        hash_equivalence_classes = Simulator(self._graph).simulate(self._num_of_simulate_samples).values()
        for hash_equivalence_class in hash_equivalence_classes:
            hash_equivalence_class_len = len(hash_equivalence_class)
            # check all pairs with equal hash for equivalence
            for node1_num in range(hash_equivalence_class_len):
                for node2_num in range(node1_num + 1, hash_equivalence_class_len):
                    node1 = hash_equivalence_class[node1_num]
                    node2 = hash_equivalence_class[node2_num]
                    if node1 in self._graph.inputs or node2 in self._graph.inputs:
                        continue
                    if node1 != node2 and not equivalence_classes.belongs(node1, node2):
                        self._find_equivalent_nodes(
                            node1,
                            node2,
                            subgraph_constraints,
                            equivalence_classes,
                            self._conflict_limit,
                            sat_oracle_report
                        )

        equivalence_marks = equivalence_classes.get_equivalence_marks()

        # Since we have to delete all nodes of equivalent class, lets found a leader --
        # the vertex with minimal depth (not to get cycles)
        for _, equivalence_class in equivalence_classes.get_equivalence_classes().items():
            equivalence_class = list(equivalence_class)
            leader = equivalence_class[0]
            for other_node_num in equivalence_class[1:]:
                # if vertices are equivalent then their outputs should be merged
                if depth[other_node_num] < depth[leader]:
                    leader = other_node_num
            other_nodes = []
            for class_node in equivalence_class:
                if leader != class_node:
                    other_nodes.append(class_node)
            # here we negate edges if needed and change pointers deleting redundant nodes
            for other_node_num in other_nodes:
                negation = int(equivalence_marks[leader] != equivalence_marks[other_node_num])
                if not self._include_complementary:
                    assert negation == 0
                node_parents = self._graph[other_node_num].parents
                for parent_n, parent_neg in node_parents:
                    self._graph[parent_n].remove_child(other_node_num, parent_neg)
                for child_num, child_neg in self._graph[other_node_num].children:
                    self._graph[leader].add_child(child_num, child_neg ^ negation)
                    self._graph[child_num].remove_parent(other_node_num, child_neg)
                    self._graph[child_num].add_parent(leader, child_neg ^ negation)
                self._graph.update_inputs_or_outputs(other_node_num, leader, negation)
                self._graph.replace_constraints(other_node_num, leader, negation)
                del self._graph[other_node_num]

        return sat_oracle_report

    # The function reduces graph by merging equivalent nodes
    def reduce(self) -> SatOracleReport:
        return self._merge_equivalent()


__all__ = [
    'Reducer'
]
