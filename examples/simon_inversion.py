from util import WorkPath, date_now, Solvers
from encoding import AAG
from algorithm import DegreeBranchingHeuristic, LeafSizeHaltingHeuristic, FraigTransformationHeuristic
from inversion import Simon16KnownSecretKeyBitsInversion
from typing import List


def inverse_9round_simon_16bits_known_common(open_text: List[int], secret_key: List[int], task_type: str):
    simon_9r = AAG(from_file=WorkPath('aiger').to_file(
        'simon_encrypt_encoding_keysize64_blocksize32_blocks2_rounds9.aag'
    ))
    return Simon16KnownSecretKeyBitsInversion(
        aag=simon_9r,
        open_text=open_text,
        secret_key=secret_key,
        work_path=WorkPath('out').to_path(date_now()),
        branching_heuristic=DegreeBranchingHeuristic(),
        halt_heuristic=LeafSizeHaltingHeuristic(
            leafs_size_lower_bound=1000,
            max_depth=8,
            reduction_conflicts=300
        ),
        transformation_heuristics=FraigTransformationHeuristic(),
        executor_workers=32,
        worker_solver=Solvers.ROKK_LRB,
        task_type=task_type
    )


def estimate_inversion_9round_simon_16bits_known(
        open_text: List[int],
        secret_key: List[int],
        task_type: str,
        use_input_decomposition: bool,
        decomposition_limit: int,
        length_decompose: str,
        shuffle_inputs: bool
):
    inverse_9round_simon_16bits_known_common(
        open_text, secret_key, task_type
    ).cnc_estimate(
        use_input_decomposition=use_input_decomposition,
        decomposition_limit=decomposition_limit,
        length_decompose=length_decompose,
        shuffle_inputs=shuffle_inputs
    )


def inverse_9round_simon_16bits_known(open_text: List[int], secret_key: List[int], task_type: str):
    inverse_9round_simon_16bits_known_common(
        open_text, secret_key, task_type
    ).cnc()


def inverse_9round_simon_16bits_known_example_randdist_1_1():
    open_text_1 = [0x65656877, 0x65656876]
    secret_key_1 = [0xb2fe, 0x7c97, 0xa734, 0x8a7f]
    inverse_9round_simon_16bits_known(open_text_1, secret_key_1, 'randdist-1')


def inverse_9round_simon_16bits_known_example_randdist_1_2():
    open_text_2 = [0x30d7f42b, 0x30d7f42a]
    secret_key_1 = [0xb2fe, 0x7c97, 0xa734, 0x8a7f]
    inverse_9round_simon_16bits_known(open_text_2, secret_key_1, 'randdist-1')


def inverse_9round_simon_16bits_known_example_randdist_1_3():
    open_text_3 = [0xd19a13c4, 0xd19a13c5]
    secret_key_1 = [0xb2fe, 0x7c97, 0xa734, 0x8a7f]
    inverse_9round_simon_16bits_known(open_text_3, secret_key_1, 'randdist-1')


def inverse_9round_simon_16bits_known_example_randdist_2_1():
    open_text_1 = [0x65656877, 0x65656876]
    secret_key_2 = [0xe5e1, 0x3e5c, 0xfe34, 0x7a47]
    inverse_9round_simon_16bits_known(open_text_1, secret_key_2, 'randdist-2')


def inverse_9round_simon_16bits_known_example_randdist_2_2():
    open_text_2 = [0x30d7f42b, 0x30d7f42a]
    secret_key_2 = [0xe5e1, 0x3e5c, 0xfe34, 0x7a47]
    inverse_9round_simon_16bits_known(open_text_2, secret_key_2, 'randdist-2')


def inverse_9round_simon_16bits_known_example_randdist_2_3():
    open_text_3 = [0xd19a13c4, 0xd19a13c5]
    secret_key_2 = [0xe5e1, 0x3e5c, 0xfe34, 0x7a47]
    inverse_9round_simon_16bits_known(open_text_3, secret_key_2, 'randdist-2')


def inverse_9round_simon_16bits_known_example_randdist_3_1():
    open_text_1 = [0x65656877, 0x65656876]
    secret_key_3 = [0xd8a6, 0x28f0, 0x4c35, 0xac81]
    inverse_9round_simon_16bits_known(open_text_1, secret_key_3, 'randdist-3')


def inverse_9round_simon_16bits_known_example_randdist_3_2():
    open_text_2 = [0x30d7f42b, 0x30d7f42a]
    secret_key_3 = [0xd8a6, 0x28f0, 0x4c35, 0xac81]
    inverse_9round_simon_16bits_known(open_text_2, secret_key_3, 'randdist-3')


def inverse_9round_simon_16bits_known_example_randdist_3_3():
    open_text_3 = [0xd19a13c4, 0xd19a13c5]
    secret_key_3 = [0xd8a6, 0x28f0, 0x4c35, 0xac81]
    inverse_9round_simon_16bits_known(open_text_3, secret_key_3, 'randdist-3')


def inverse_9round_simon_16bits_known_example_nulldist_1():
    open_text_1 = [0x0, 0x1]
    secret_key_1 = [0xb2fe, 0x7c97, 0xa734, 0x8a7f]
    inverse_9round_simon_16bits_known(open_text_1, secret_key_1, 'nulldist-1')


def inverse_9round_simon_16bits_known_example_nulldist_2():
    open_text_1 = [0x0, 0x1]
    secret_key_2 = [0xe5e1, 0x3e5c, 0xfe34, 0x7a47]
    inverse_9round_simon_16bits_known(open_text_1, secret_key_2, 'nulldist-2')


def inverse_9round_simon_16bits_known_example_nulldist_3():
    open_text_1 = [0x0, 0x1]
    secret_key_3 = [0xd8a6, 0x28f0, 0x4c35, 0xac81]
    inverse_9round_simon_16bits_known(open_text_1, secret_key_3, 'nulldist-3')


def inverse_9round_simon_16bits_known_example_rand_1_1():
    open_text_1 = [0x65656877, 0x34895467]
    secret_key_1 = [0xb2fe, 0x7c97, 0xa734, 0x8a7f]
    inverse_9round_simon_16bits_known(open_text_1, secret_key_1, 'rand-1')


def inverse_9round_simon_16bits_known_example_rand_1_2():
    open_text_2 = [0x30d7f42b, 0x591a97c5]
    secret_key_1 = [0xb2fe, 0x7c97, 0xa734, 0x8a7f]
    inverse_9round_simon_16bits_known(open_text_2, secret_key_1, 'rand-1')


def inverse_9round_simon_16bits_known_example_rand_1_3():
    open_text_3 = [0xd19a13c4, 0x0b597351]
    secret_key_1 = [0xb2fe, 0x7c97, 0xa734, 0x8a7f]
    inverse_9round_simon_16bits_known(open_text_3, secret_key_1, 'rand-1')


def inverse_9round_simon_16bits_known_example_rand_2_1():
    open_text_1 = [0x65656877, 0x34895467]
    secret_key_2 = [0xe5e1, 0x3e5c, 0xfe34, 0x7a47]
    inverse_9round_simon_16bits_known(open_text_1, secret_key_2, 'rand-2')


def inverse_9round_simon_16bits_known_example_rand_2_2():
    open_text_2 = [0x30d7f42b, 0x591a97c5]
    secret_key_2 = [0xe5e1, 0x3e5c, 0xfe34, 0x7a47]
    inverse_9round_simon_16bits_known(open_text_2, secret_key_2, 'rand-2')


def inverse_9round_simon_16bits_known_example_rand_2_3():
    open_text_3 = [0xd19a13c4, 0x0b597351]
    secret_key_2 = [0xe5e1, 0x3e5c, 0xfe34, 0x7a47]
    inverse_9round_simon_16bits_known(open_text_3, secret_key_2, 'rand-2')


def inverse_9round_simon_16bits_known_example_rand_3_1():
    open_text_1 = [0x65656877, 0x34895467]
    secret_key_3 = [0xd8a6, 0x28f0, 0x4c35, 0xac81]
    inverse_9round_simon_16bits_known(open_text_1, secret_key_3, 'rand-3')


def inverse_9round_simon_16bits_known_example_rand_3_2():
    open_text_2 = [0x30d7f42b, 0x591a97c5]
    secret_key_3 = [0xd8a6, 0x28f0, 0x4c35, 0xac81]
    inverse_9round_simon_16bits_known(open_text_2, secret_key_3, 'rand-3')


def inverse_9round_simon_16bits_known_example_rand_3_3():
    open_text_3 = [0xd19a13c4, 0x0b597351]
    secret_key_3 = [0xd8a6, 0x28f0, 0x4c35, 0xac81]
    inverse_9round_simon_16bits_known(open_text_3, secret_key_3, 'rand-3')


def estimate_inversion_9round_simon_16bits_known_rand_3_2():
    open_text_2 = [0x30d7f42b, 0x591a97c5]
    secret_key_3 = [0xd8a6, 0x28f0, 0x4c35, 0xac81]
    estimate_inversion_9round_simon_16bits_known(
        open_text_2,
        secret_key_3,
        'rand-3',
        use_input_decomposition=True,
        decomposition_limit=1000,
        length_decompose='2',
        shuffle_inputs=False
    )
