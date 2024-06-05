from enum import Enum
import threading


class VertexType(Enum):
    [
        VERTEX,
        LEAF,
        REFUTATION_LEAF,
        CUBE_LEAF,
        SOLUTION_LEAF
    ] = range(5)


class CubeTreeVertex:
    def __init__(
            self,
            vertex_type: VertexType,
            depth: int,
            heuristics_dict_pointer: int,
            append_output: bool = True
    ):
        self._vertex_type = vertex_type
        self._depth = depth
        self._heuristics_dict_pointer = heuristics_dict_pointer
        self._append_output = append_output

    @property
    def heuristics_dict_pointer(self) -> int:
        return self._heuristics_dict_pointer

    @heuristics_dict_pointer.setter
    def heuristics_dict_pointer(self, heuristics_dict_pointer: int):
        self._heuristics_dict_pointer = heuristics_dict_pointer

    @property
    def append_output(self) -> bool:
        return self._append_output

    @append_output.setter
    def append_output(self, append_output: bool):
        self._append_output = append_output

    @property
    def vertex_type(self) -> VertexType:
        return self._vertex_type

    @property
    def depth(self) -> int:
        return self._depth

    @vertex_type.setter
    def vertex_type(self, vertex_type: bool):
        self._vertex_type = vertex_type


class CncTree(dict):
    def __init__(self):
        super().__init__()
        self._refutation_leafs_num = 0
        self._solution_leafs_num = 0
        self._cube_leafs_num = 0

        self._leafs_num = 1
        self._tree_size = 1
        self.lock = threading.Lock()
        self[0] = CubeTreeVertex(VertexType.LEAF, 0, -1)

    def __getitem__(self, item: int) -> CubeTreeVertex:
        return super().__getitem__(item)

    def __setitem__(self, key: int, value: CubeTreeVertex):
        with self.lock:
            super().__setitem__(key, value)

    @staticmethod
    def count_child_num(parent_num: int, branching_value: int) -> int:
        return parent_num * 2 + 1 + branching_value

    def add_leaf(self, parent_num: int, branching_value: int, vertex_type: VertexType, append_output: bool):
        self._tree_size += 1
        parent_vertex_type = self[parent_num].vertex_type
        if not (parent_vertex_type == VertexType.LEAF or parent_vertex_type == VertexType.VERTEX):
            print(parent_num, parent_vertex_type)
        assert parent_vertex_type == VertexType.LEAF or parent_vertex_type == VertexType.VERTEX
        if parent_vertex_type == VertexType.LEAF:
            self[parent_num].vertex_type = VertexType.VERTEX
            self._leafs_num -= 1
        child_num = CncTree.count_child_num(parent_num, branching_value)
        self[child_num] = CubeTreeVertex(
            vertex_type,
            self[parent_num].depth + 1,
            self[parent_num].heuristics_dict_pointer,
            self[parent_num].append_output and append_output
        )
        self._leafs_num += 1
        if vertex_type == VertexType.REFUTATION_LEAF:
            self._refutation_leafs_num += 1
        elif vertex_type == VertexType.SOLUTION_LEAF:
            self._solution_leafs_num += 1
        elif vertex_type == VertexType.CUBE_LEAF:
            self._cube_leafs_num += 1

    def make_leaf_cube(self, leaf_num: int):
        assert self[leaf_num].vertex_type == VertexType.LEAF
        self[leaf_num].vertex_type = VertexType.CUBE_LEAF
        self._cube_leafs_num += 1

    @property
    def leafs_num(self) -> int:
        return self._leafs_num

    @property
    def tree_size(self) -> int:
        return self._tree_size

    @property
    def refutation_leafs_num(self) -> int:
        return self._refutation_leafs_num

    @property
    def cube_leafs_num(self) -> int:
        return self._cube_leafs_num

    @property
    def solution_leafs_num(self) -> int:
        return self._solution_leafs_num

    @property
    def solved_cubes_leafs_num(self) -> int:
        return self._solution_leafs_num + self._refutation_leafs_num

    def retrieve_statistics(self):
        raise NotImplementedError


__all__ = [
    'CncTree',
    'VertexType'
]
