from typing import List, Tuple


class Gate:
    def __init__(self, number: int):
        self._node_number = number
        self._parents = {}
        self._children = {}
        self._out_degree = 0
        self._in_degree = 0
        # Node can be included in constraint negated or not (depending on index in self._constraints)
        # Elements are mapping to vertices and their negation marks in constraints
        self._constraints = ({}, {})

    @property
    def in_degree(self) -> int:
        return self._in_degree

    @property
    def out_degree(self) -> int:
        return self._out_degree

    def add_parent(self, parent: int, is_negative: int) -> bool:
        parent_appeared = False
        if parent in self._parents:
            if self._parents[parent] ^ is_negative == 1:
                self._parents[parent] = 2
                self._in_degree += 1
                parent_appeared = True
        else:
            self._parents[parent] = is_negative
            self._in_degree += 1
            parent_appeared = True
        return parent_appeared

    def add_child(self, child: int, is_negative: int) -> bool:
        child_appeared = False
        if child in self._children:
            if self._children[child] ^ is_negative == 1:
                self._children[child] = 2
                self._out_degree += 1
                child_appeared = True
        else:
            self._children[child] = is_negative
            self._out_degree += 1
            child_appeared = True
        return child_appeared

    def remove_parent(self, parent: int, is_negative: int) -> bool:
        parent_removed = False
        if parent in self._parents:
            parent_removed = True
            self._in_degree -= 1
            if self._parents[parent] == 2:
                self._parents[parent] = is_negative ^ 1
            else:
                del self._parents[parent]
        return parent_removed

    def remove_child(self, child, is_negative: int) -> bool:
        child_removed = False
        if child in self._children:
            child_removed = True
            self._out_degree -= 1
            if self._children[child] == 2:
                self._children[child] = is_negative ^ 1
            else:
                del self._children[child]
        return child_removed

    def remove_children(self):
        self._children = {}
        self._out_degree = 0

    def _retrieve_constraints(self, negation_mark):
        constraints = []
        for node_n, node_neg in self._constraints[negation_mark].items():
            if node_neg == 2:
                constraints.extend([(node_n, 0), (node_n, 1)])
            else:
                constraints.append((node_n, node_neg))
        return constraints

    @property
    def constraints(self) -> List[Tuple[int, int]]:
        def order_nodes(first_node: int, second_node: int) -> Tuple[int, int]:
            if first_node > second_node:
                return second_node, first_node
            else:
                return first_node, second_node

        constraints = []
        for negation_mark in [0, 1]:
            for node_n, node_neg in self._retrieve_constraints(negation_mark):
                constraints.append(order_nodes(self.node_number + negation_mark, node_n + node_neg))
        return constraints

    @property
    def node_number(self) -> int:
        return self._node_number

    @node_number.setter
    def node_number(self, number: int):
        self._node_number = number

    @property
    def children(self) -> List[Tuple[int, int]]:
        children = []
        for child_number, child_value in self._children.items():
            if child_value == 2:
                children.extend([(child_number, 0), (child_number, 1)])
            else:
                children.append((child_number, child_value))
        return children

    @property
    def parents(self) -> List[Tuple[int, int]]:
        parents = []
        for parent_number, parent_value in self._parents.items():
            if parent_value == 2:
                parents.extend([(parent_number, 0), (parent_number, 1)])
            else:
                parents.append((parent_number, parent_value))
        return parents

    # When current_node_negation ^ current_node_value == 0, then other_node_negation ^ other_node_value should be 1 for
    # all other nodes in constraint with x. So we can derive such values and check whether they violate constraints.
    def derive_values_from_constraints(self, assigned_value: int) -> List[Tuple[int, int]]:
        derived_values = []
        for node_num, negation_mark in self._retrieve_constraints(assigned_value):
            derived_values.append((node_num, negation_mark ^ 1))
        return derived_values

    # When current_node_negation ^ current_node_value == 1, then other_node_negation ^ other_node_value can take
    # arbitrary values
    def derive_nodes_to_remove_constraints_with(self, assigned_value: int) -> List[Tuple[int, int]]:
        nodes_to_remove_constraints_with = []
        for node_num, negation_mark in self._retrieve_constraints(assigned_value ^ 1):
            nodes_to_remove_constraints_with.append((node_num, negation_mark))
        return nodes_to_remove_constraints_with

    # The function removes constraint with current node in which it is included with such a negation mark and
    # other node
    def remove_constraint(self, negation_mark: int, other_node: int, other_node_negation_mark: int):
        if self._constraints[negation_mark][other_node] == 2:
            self._constraints[negation_mark][other_node] = other_node_negation_mark ^ 1
        else:
            del self._constraints[negation_mark][other_node]

    # When we add constraint in which one vertex coincides with another but another differs only in negation then
    # we conclude that current_vertex ^ negation_mark should be 1
    def check_constraint_contradiction(self, negation_mark: int, other_node: int, other_node_neg_mark: int) -> Tuple[
        bool, List[int]
    ]:
        bucket = self._constraints[negation_mark]
        contradiction = other_node in bucket and (
                bucket[other_node] == 2 or bucket[other_node] ^ other_node_neg_mark == 1
        )
        constraints_to_remove = []
        if other_node in bucket:
            if bucket[other_node] == 2:
                constraints_to_remove.extend([0, 1])
            else:
                constraints_to_remove.append(bucket[other_node])
        return not contradiction, constraints_to_remove

    # The function adds constraint
    def add_constraint(self, negation_mark: int, other_node: int, other_node_neg_mark: int):
        if other_node not in self._constraints[negation_mark]:
            self._constraints[negation_mark][other_node] = other_node_neg_mark
        elif self._constraints[negation_mark][other_node] ^ other_node_neg_mark == 1:
            self._constraints[negation_mark][other_node] = 2

    # The function returns negation mark of current vertex in constraint and other vertex, and it's negation mark
    # in this constraint
    def pairs_in_constraints(self) -> List[Tuple[int, int, int]]:
        pairs_in_constraints = []
        for negation_mark in range(2):
            for node_n, node_neg in self._retrieve_constraints(negation_mark):
                pairs_in_constraints.append((negation_mark, node_n, node_neg))
        return pairs_in_constraints


__all__ = [
    'Gate'
]
