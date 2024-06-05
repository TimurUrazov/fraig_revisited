from exception import ControversialAssignmentException, ConstraintViolationAssignmentException
from .graph import AndInverterGraph
from typing import Tuple, List


class Propagation:
    def __init__(self, and_inverter_graph: AndInverterGraph):
        self._graph = and_inverter_graph
        self._assigned_ahead = {}

    # The function checks if assignment is controversial and returns True if value was not already assigned
    def _check_controversial_assignment(self, node_number: int, value_to_assign: int) -> bool:
        already_assigned = node_number in self._assigned_ahead
        if already_assigned:
            if self._assigned_ahead[node_number] ^ value_to_assign == 1:
                raise ControversialAssignmentException(node_number, value_to_assign)
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
                    raise ConstraintViolationAssignmentException(
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

    def _assign(self, node_numbers_and_values_to_assign: Tuple[Tuple[int, int], ...]):
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
        for assigned_node, assigned_value in self._assigned_ahead.items():
            # before deleting node we assign outputs or inputs
            self._graph.assign_input_or_output(assigned_node, assigned_value)
            self._delete_pointers_to_vertex(assigned_node)
            del self._graph[assigned_node]

    # The function assign node and make propagation (consequent assignments)
    def propagate(self, node_numbers_and_values_to_assign: Tuple[Tuple[int, int], ...]):
        self._assign(node_numbers_and_values_to_assign)
        self._remove_assigned_nodes()


__all__ = [
    'Propagation'
]
