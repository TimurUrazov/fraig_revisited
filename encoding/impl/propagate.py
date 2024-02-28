from typing import Mapping, Set, Tuple, List, Dict, Union, Callable
from random import getrandbits
from .aag import AAG, Constraints
from .cnf import CNF
from util import solve_cnf_with_kissat, analyze_result
from aig_substitutions.exceptions import NonExistentAssignment, ControversialAssignment, ConstraintViolationAssignment
from .normalize import Normalize
from copy import copy


# there are 3 types of incorrect assignments:
# 1) Controversial assignment: we are trying to assign value to vertex which has been already assigned with 1 - value
# 2) Constraint violation
# 3) Non-existent assignment: the assignment which can't be made because vertex always takes other value
# (it's literally non-existent, because it's for self-checking)
class Propagation:
    def __init__(self, aag: AAG, constraints: Set[Tuple[int]], conflicts_limit: int):
        self._aag = aag
        self._graph = aag.get_graph()
        self._constraints = constraints
        self._random_numbers = None
        self._number_of_samples = 64
        self._ones = 2**self._number_of_samples - 1
        self._node_constraints = None
        self._assigned_ahead = {}
        self._conflicts_limit = conflicts_limit
        inputs = aag.get_data().inputs()
        self._inputs = inputs
        self._inputs_to_indices = {input_: {index} for index, input_ in enumerate(inputs)}
        # to replace 'other' output to equivalent
        self._outputs = [(output - output % 2, output % 2) for output in aag.get_data().outputs()]
        self._outputs_to_indices = {}
        for index, output in enumerate([output[0] for output in self._outputs]):
            if output in self._outputs_to_indices:
                self._outputs_to_indices[output].add(index)
            else:
                self._outputs_to_indices[output] = {index}
        self._out_degree_deltas = None

    def _get_out_degrees(self):
        if self._out_degree_deltas is None:
            self._out_degree_deltas = {}

            for node_num in self._graph:
                self._out_degree_deltas[node_num] = self._graph[node_num].out_degree

        return self._out_degree_deltas

    def _simulate(self) -> Mapping[int, int]:
        if self._random_numbers is None:
            self._random_numbers = {}

            def simulate_node(node_number: int):
                if node_number in self._random_numbers:
                    return

                node = self._graph[node_number]

                if node.in_degree == 0:
                    current_simulation_vector = getrandbits(self._number_of_samples)
                    self._random_numbers[node_number] = current_simulation_vector
                    return

                assert node.in_degree == 2

                parents = node.parents

                parent_hashes = []

                for parent_num, negation in parents:
                    if parent_num not in self._random_numbers:
                        simulate_node(parent_num)
                    parent_hashes.append(self._random_numbers[parent_num] ^ (self._ones if negation == 1 else 0))

                current_simulation_vector = parent_hashes[0] & parent_hashes[1]

                self._random_numbers[node_number] = current_simulation_vector

            for node_num in self._graph:
                simulate_node(node_num)

        return self._random_numbers

    # The function returns clauses defining And-clauses
    def _nodes_constraints(self, node_n) -> Constraints:
        if self._node_constraints is None:
            self._node_constraints = {}

            def negate_var(node_num: int, is_negated: int):
                if is_negated == 1:
                    return -node_num
                return node_num

            for node_number in self._graph:
                node = self._graph[node_number]

                parents = node.parents

                assert node.in_degree % 2 == 0

                if node.in_degree == 0:
                    self._node_constraints[node_number] = []
                    continue

                var_output = node_number
                var_input1 = negate_var(*parents[0])
                var_input2 = negate_var(*parents[1])

                self._node_constraints[node_number] = [
                    [var_output, -var_input1, -var_input2],
                    [-var_output, var_input1],
                    [-var_output, var_input2]
                ]

        return self._node_constraints[node_n]

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

    # the function just assign literals within constraints with minimal possible number (node_num_1
    # and node_num_2) are numbers of nodes in which cones end
    @staticmethod
    def _renumber_constraints(constraints: Constraints, node_num: int) -> Tuple[Constraints, int]:
        new_constraints = []
        old_to_new_literal = {}
        new_literal_num = 1

        old_to_new_literal[node_num] = new_literal_num

        for constraint in constraints:
            new_constraint = []
            for literal in constraint:
                if abs(literal) not in old_to_new_literal:
                    new_literal_num += 1
                    old_to_new_literal[abs(literal)] = new_literal_num
                new_constraint.append(old_to_new_literal[abs(literal)] * (literal // abs(literal)))
            new_constraints.append(new_constraint)

        return new_constraints, old_to_new_literal[node_num]

    # The function builds constraints for the cone which ends in vertex
    def _build_subgraph_constraints(self, vertex: int, graph_copy) -> Constraints:
        constraints = []
        visited = set()

        def dfs(node_number: int):
            if node_number in visited:
                return
            visited.add(node_number)

            constraints.extend(self._nodes_constraints(node_number))

            for node_num, _ in graph_copy[node_number].parents:
                if node_num in visited:
                    return
                dfs(node_num)

        dfs(vertex)

        return constraints

    # Function checks if assignment is possible
    def _ensure_assignment(self, node_number: int, value_to_assign: int, graph_copy):
        simulation_vectors = self._simulate()
        # if we want to assign zero then we check if there is any zero in simulation vector or:
        if simulation_vectors[node_number] ^ (self._ones if value_to_assign == 0 else 0) == 0:
            constraints = self._build_subgraph_constraints(node_number, graph_copy)
            new_constraints, assumption_value = Propagation._renumber_constraints(constraints, node_number)
            new_constraints.append([-assumption_value if value_to_assign == 0 else assumption_value])

            result = solve_cnf_with_kissat(
                cnf_str=CNF(from_clauses=new_constraints).get_data().source(),
                conflicts_limit=self._conflicts_limit
            )

            answer, _, _ = analyze_result(
                result,
                f'checking if {node_number} can be assigned with {value_to_assign}',
                print_output=True
            )

            if answer == 'UNSAT':
                raise NonExistentAssignment(node_number, value_to_assign)

    # I see two possible ways to assign node:
    # 1) Trying to escape solver invocation (but eventually having to call it):
    # checking whether we violate constraints or assignments already made and only after that
    # applying the assignment and subsequent assignments (calling solver if simulation vector shows
    # value can't be assigned).
    # 2) Call solver every time simulation vector shows value can't be assigned without 'looking ahead'.
    # I consider it reasonable to reduce solver calls. So I chose the first way.

    # The function assigns node which is assigned with value 'in advance'
    def _assign_node(self, node_number: int, value_to_assign: int, graph_copy):
        self._ensure_assignment(node_number, value_to_assign, graph_copy)
        self._graph[node_number].set_assignment(value_to_assign)

    # The function checks if assignment is controversial and returns True if value was not already assigned
    def _check_controversial_assignment(self, node_number: int, value_to_assign: int) -> bool:
        already_assigned = node_number in self._assigned_ahead
        if already_assigned:
            if self._assigned_ahead[node_number] ^ value_to_assign == 1:
                raise ControversialAssignment(node_number, value_to_assign)
        return not already_assigned

    # The function remove constraint in both nodes
    def _remove_constraint(self, first_node_num: int, second_node_num: int, first_neg_mark: int, sec_neg_mark: int):
        first_node = self._graph[first_node_num]
        second_node = self._graph[second_node_num]
        first_node.remove_constraint(first_neg_mark, second_node_num, sec_neg_mark)
        if first_node_num != second_node_num or first_neg_mark != sec_neg_mark:
            second_node.remove_constraint(sec_neg_mark, first_node_num, sec_neg_mark)

    # The function adds constraint to both nodes
    def _add_constraint(
            self, first_node_num: int, second_node_num: int, first_neg_mark: int, sec_neg_mark: int
    ) -> List[Tuple[int, int]]:
        first_node = self._graph[first_node_num]
        second_node = self._graph[second_node_num]
        derived_assignment = []
        first_contradiction, first_values_to_remove = first_node.check_constraint_contradiction(
            first_neg_mark, second_node_num, sec_neg_mark
        )
        if not first_contradiction:
            for second_node_neg in first_values_to_remove:
                first_node.remove_constraint(first_neg_mark, second_node_num, second_node_neg)
                if first_node_num != second_node_num or second_node_neg != first_neg_mark:
                    second_node.remove_constraint(second_node_neg, first_node_num, first_neg_mark)
            derived_assignment.append((first_node_num, first_neg_mark ^ 1))
        else:
            second_contradiction, second_values_to_remove = second_node.check_constraint_contradiction(
                sec_neg_mark, first_node_num, first_neg_mark
            )
            if not second_contradiction:
                for first_node_neg in second_values_to_remove:
                    second_node.remove_constraint(sec_neg_mark, first_node_num, first_node_neg)
                    if first_node_num != second_node_num or first_node_neg != sec_neg_mark:
                        first_node.remove_constraint(first_node_neg, second_node_num, sec_neg_mark)
                derived_assignment.append((second_node_num, sec_neg_mark ^ 1))
            else:
                first_node.add_constraint(first_neg_mark, second_node_num, sec_neg_mark)
                second_node.add_constraint(sec_neg_mark, first_node_num, first_neg_mark)
        return derived_assignment

    # The function checks what constraints it violates and retrieves new assignments
    def _check_constraint_violation_and_derive_assignments(
            self, node_number: int, assigned_value: int
    ) -> List[Tuple[int, int]]:
        node = self._graph[node_number]
        # First, we define which values to assign according to value we assigned to current node
        derived_assignments = node.derive_values_from_constraints(assigned_value)
        # Second, we check whether already assigned values violate constraint
        for node_n, node_v in derived_assignments:
            if node_n in self._assigned_ahead:
                if self._assigned_ahead[node_n] ^ node_v == 1:
                    raise ConstraintViolationAssignment(
                        node_n, self._assigned_ahead[node_n], [node_number, node_n], assigned_value, node_v ^ 1
                    )
            self._assigned_ahead[node_n] = node_v
            self._remove_constraint(node_n, node_number, node_v ^ 1, assigned_value)
        nodes_to_remove_constraints_with = node.derive_nodes_to_remove_constraints_with(assigned_value)
        for node_n, node_v in nodes_to_remove_constraints_with:
            self._remove_constraint(node_n, node_number, node_v, assigned_value ^ 1)
        return derived_assignments

    # This function allows to propagate zero to parent if node_num is zero and parent_neg_mark ^ parent_value == 1
    def _propagate_zero_to_parent(
            self,
            node_num: int,
            first_parent: int,
            second_parent: int,
            first_parent_neg_mark: int,
            second_parent_neg_mark: int
    ) -> List[Tuple[int, int]]:
        derived_assignment = []
        if self._assigned_ahead[node_num] == 1:
            return derived_assignment
        if first_parent in self._assigned_ahead and self._assigned_ahead[first_parent] ^ first_parent_neg_mark == 1:
            derived_assignment.append((second_parent, second_parent_neg_mark))
        if second_parent in self._assigned_ahead and self._assigned_ahead[second_parent] ^ second_parent_neg_mark == 1:
            derived_assignment.append((first_parent, first_parent_neg_mark))
        return derived_assignment

    # The function propagate assignment already made and returns new assignments
    # Zeroes move to outputs, ones move to inputs. Zero means that node and edge give zero. When two parents are ones,
    # then child is one
    #
    # Before we propagate assignment we check what constraints we violate, and we add new constraints
    def _propagate_assignment(self, node_number: int, assigned_value: int) -> List[Tuple[int, int]]:
        derived_assignments = self._check_constraint_violation_and_derive_assignments(node_number, assigned_value)

        node = self._graph[node_number]
        if assigned_value == 1:
            parents = node.parents
            for parent_num, negation_mark in parents:
                child_value_to_assign = negation_mark ^ assigned_value
                if self._check_controversial_assignment(parent_num, child_value_to_assign):
                    derived_assignments.append((parent_num, child_value_to_assign))
                    self._assigned_ahead[parent_num] = child_value_to_assign
        elif node.in_degree == 2:
            parents_of_zero = node.parents
            if parents_of_zero[0][0] not in self._assigned_ahead and parents_of_zero[1][0] not in self._assigned_ahead:
                derived_assignment = self._add_constraint(
                    parents_of_zero[0][0], parents_of_zero[1][0], parents_of_zero[0][1] ^ 1, parents_of_zero[1][1] ^ 1
                )
                if len(derived_assignment) == 1 and self._check_controversial_assignment(
                        derived_assignment[0][0], derived_assignment[0][1]
                ):
                    derived_assignments.extend(derived_assignment)
                    self._assigned_ahead[derived_assignment[0][0]] = derived_assignment[0][1]
            elif parents_of_zero[0][0] not in self._assigned_ahead or parents_of_zero[1][0] not in self._assigned_ahead:
                derived_assignment = self._propagate_zero_to_parent(
                    node_number,
                    parents_of_zero[0][0],
                    parents_of_zero[1][0],
                    parents_of_zero[0][1],
                    parents_of_zero[1][1]
                )
                if len(derived_assignment) == 1 and self._check_controversial_assignment(
                        derived_assignment[0][0], derived_assignment[0][1]
                ):
                    derived_assignments.extend(derived_assignment)
                    self._assigned_ahead[derived_assignment[0][0]] = derived_assignment[0][1]
        children = node.children
        for child_num, negation_mark in children:
            child_value_to_assign = negation_mark ^ assigned_value
            if child_value_to_assign == 0:
                if self._check_controversial_assignment(child_num, child_value_to_assign):
                    derived_assignments.append((child_num, child_value_to_assign))
                    self._assigned_ahead[child_num] = child_value_to_assign
            if child_value_to_assign == 1:
                for parent_n, parent_neg in self._graph[child_num].parents:
                    if parent_n != node_number and parent_n in self._assigned_ahead:
                        if parent_neg ^ self._assigned_ahead[parent_n] == 1:
                            if self._check_controversial_assignment(child_num, child_value_to_assign):
                                derived_assignments.append((child_num, child_value_to_assign))
                                self._assigned_ahead[child_num] = child_value_to_assign
        return derived_assignments

    def _assign(self, node_number: int, value_to_assign: int):
        self._assigned_ahead[node_number] = value_to_assign
        current_assignments = self._propagate_assignment(node_number, value_to_assign)

        # First, we check if assignment is controversial
        while len(current_assignments) > 0:
            derived_assignments = []
            for n_number, v_to_assign in current_assignments:
                assert n_number in self._assigned_ahead
                # if value was assigned then we 'propagate' the assignment
                derived_assignments.extend(self._propagate_assignment(n_number, v_to_assign))
            current_assignments = derived_assignments

        # Then we apply assignments running solver (for self-checking)
        # for n_number, v_to_assign in self._assigned_ahead.items():
        #     self._assign_node(n_number, v_to_assign, graph_copy)

    def _assign_several(self, node_numbers_and_values_to_assign: List[Tuple[int, int]]):
        current_assignments = node_numbers_and_values_to_assign
        for node_number, value_to_assign in current_assignments:
            self._assigned_ahead[node_number] = value_to_assign

        # First, we check if assignment is controversial
        while len(current_assignments) > 0:
            derived_assignments = []
            for n_number, v_to_assign in current_assignments:
                assert n_number in self._assigned_ahead
                # if value was assigned then we 'propagate' the assignment
                derived_assignments.extend(self._propagate_assignment(n_number, v_to_assign))
            current_assignments = derived_assignments

    # The function deletes pointers from parents and children of node to this node
    def _delete_pointers_to_vertex(self, node_to_del_pointers_on):
        graph = self._graph
        node = graph[node_to_del_pointers_on]
        for child_node, child_negation_mark in node.children:
            if child_node in graph:
                graph[child_node].remove_parent(node_to_del_pointers_on, child_negation_mark)
        for parent_node, parent_negation_mark in node.parents:
            if parent_node in graph:
                graph[parent_node].remove_child(node_to_del_pointers_on, parent_negation_mark)

    # The function removes nodes which are already assigned (and handle dangling edges)
    def _remove_assigned_nodes(self):
        # noinspection PyTypeChecker
        def assign_element(
                elements_to_indices: Dict[int, Set[int]],
                elements: Union[List[Tuple[int, int]], List[int]],
                node: int,
                value: int,
                is_output: bool
        ):
            if node in elements_to_indices:
                for element_index in elements_to_indices[node]:
                    if is_output:
                        elem, mark = elements[element_index]
                        elements[element_index] = (value ^ mark, 0)
                    else:
                        elements[element_index] = value
                del elements_to_indices[node]

        for assigned_node in self._assigned_ahead:
            # before deleting node we assign outputs or inputs
            assign_element(
                self._outputs_to_indices, self._outputs, assigned_node, self._assigned_ahead[assigned_node], True
            )
            assign_element(
                self._inputs_to_indices, self._inputs, assigned_node, self._assigned_ahead[assigned_node], False
            )
            self._delete_pointers_to_vertex(assigned_node)
            del self._graph[assigned_node]

    # The function normalizes graph
    def _normalize(self, output_handler: Callable[[List[int]], List[int]]) -> AAG:
        inputs, outputs, and_gates, constraints = Normalize(
            graph=self._graph,
            inputs=self._inputs,
            outputs=self._outputs,
            inputs_to_indices=self._inputs_to_indices,
            outputs_to_indices=self._outputs_to_indices
        ).normalize(self._out_degree_deltas)

        return AAG(
            from_gates_and_inputs=(
                inputs,
                output_handler(outputs),
                and_gates
            ),
            constraints=constraints,
            graph=self._graph
        )

    # The function assign node and make propagation (consequent assignments)
    def propagate(
            self,
            node_number: int,
            value_to_assign: int,
            output_handler: Callable[[List[int]], List[int]] = lambda elements: list(filter(lambda x: x > 1, elements))
    ) -> AAG:
        self._get_out_degrees()
        self._assign(node_number, value_to_assign)
        self._remove_assigned_nodes()
        return self._normalize(output_handler)

    # The function assign node and make propagation (consequent assignments)
    def propagate_several(
            self,
            node_numbers_and_values_to_assign: List[Tuple[int, int]],
            output_handler: Callable[[List[int]], List[int]] = lambda elements: list(
                filter(lambda x: x > 1, elements))
    ) -> AAG:
        self._get_out_degrees()
        self._assign_several(node_numbers_and_values_to_assign)
        self._remove_assigned_nodes()
        return self._normalize(output_handler)

    # The function returns vertices whose degree has changed and delta by which it changed
    def vertices_with_degree_changed(self) -> List[Tuple[int, int]]:
        return list(self._out_degree_deltas.items())

    def get_out_degree_deltas_sum(self) -> int:
        out_degree_deltas_sum = 0
        for _, value in self._out_degree_deltas.items():
            out_degree_deltas_sum += value
        return out_degree_deltas_sum


__all__ = [
    'Propagation'
]
