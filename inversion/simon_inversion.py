from .inversion_algorithm import InversionAlgorithm
from encoding import AAG
from fraig import AndInverterGraph
from typing import List
from util import WorkPath, Solvers
from algorithm import BranchingHeuristic, HaltingHeuristic, TransformationHeuristic


class Simon16KnownSecretKeyBitsInversion(InversionAlgorithm):
    def __init__(
            self,
            aag: AAG,
            open_text: List[int],
            secret_key: List[int],
            work_path: WorkPath,
            branching_heuristic: BranchingHeuristic,
            halt_heuristic: HaltingHeuristic,
            transformation_heuristics: TransformationHeuristic,
            executor_workers: int,
            worker_solver: Solvers,
            task_type: str
    ):
        secret_key_bits = []

        [secret_key_bits.extend(bin(block)[2:].zfill(16)) for block in secret_key]

        open_text_bits = []

        [open_text_bits.extend(bin(block)[2:].zfill(32)) for block in open_text]

        input_bits = secret_key_bits + open_text_bits

        vertices_to_propagate = list(zip([int(2 * i) for i in range(65, 129)], [int(i) for i in open_text_bits]))

        first_16_cypher_bits = list(
            zip([int(2 * i) for i in range(1, 17)], [int(i) for i in bin(secret_key[0])[2:].zfill(16)])
        )

        cypher_text = AndInverterGraph(
            aag=aag
        ).generate_outputs_given_inputs(dict(zip([int(2 * i) for i in range(1, 129)], [int(i) for i in input_bits])))

        vertices_to_propagate.extend(
            (output - output % 2, output_bit ^ (output % 2)) for output, output_bit in cypher_text.items()
        )

        vertices_to_propagate.extend(first_16_cypher_bits)

        super().__init__(
            aag=aag,
            output_bits_str=''.join(list(map(str, [v for _, v in sorted(cypher_text.items())]))),
            work_path=work_path,
            branching_heuristic=branching_heuristic,
            halt_heuristic=halt_heuristic,
            transformation_heuristics=transformation_heuristics,
            executor_workers=executor_workers,
            worker_solver=worker_solver,
            vertices_to_propagate=vertices_to_propagate,
            task_type=task_type
        )
