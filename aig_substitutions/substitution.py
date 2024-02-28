from .exceptions import IncompatibleSubstitution
from util import write_to_file
from encoding.impl.aag import AAG
from typing import List


def _check_if_substitute_and_substitute(nodes_and_values_to_substitute, current_node):
    if current_node % 2 == 1 and current_node - 1 in nodes_and_values_to_substitute:
        return 1 - nodes_and_values_to_substitute[current_node - 1]
    if current_node % 2 == 0 and current_node in nodes_and_values_to_substitute:
        return nodes_and_values_to_substitute[current_node]
    return current_node


def _get_gate_without_neg(gate_num: int):
    return (gate_num - 1, False) if gate_num % 2 else (gate_num, True)


def renumber_gates_of_aag(and_gates: List[List[int]], inputs: List[int], outputs: List[int]):
    global current_max_variable
    current_max_variable = 0
    already_enumerated = {}

    def find_number(gate):
        gate, positive = _get_gate_without_neg(gate)
        if gate not in already_enumerated:
            global current_max_variable
            current_max_variable += 1
            already_enumerated[gate] = current_max_variable * 2
        new_num = already_enumerated[gate]
        if positive:
            return new_num
        else:
            return new_num + 1

    for input_index, input_value in enumerate(inputs):
        inputs[input_index] = find_number(input_value)

    new_and_gates = []

    for gate in and_gates:
        new_and_gates.append([
            find_number(gate[0]),
            find_number(gate[1]),
            find_number(gate[2])
        ])

    for output_index, output_value in enumerate(outputs):
        outputs[output_index] = already_enumerated[output_value]

    return new_and_gates, already_enumerated, inputs, outputs


def make_and_process_substitution(nodes_and_values_to_substitute: dict, aag: AAG, compatibility_checker):
    substitution_is_compatible, solvetime, conflicts = compatibility_checker(
        nodes_and_values_to_substitute,
        aag
    )

    if not substitution_is_compatible:
        raise IncompatibleSubstitution(solvetime, conflicts)

    and_gates = aag.get_data().and_gates()

    for i in range(len(and_gates)):
        parts_of_and_gate = and_gates[i]
        and_gate = [
            int(parts_of_and_gate[0]),
            _check_if_substitute_and_substitute(nodes_and_values_to_substitute, int(parts_of_and_gate[1])),
            _check_if_substitute_and_substitute(nodes_and_values_to_substitute, int(parts_of_and_gate[2]))
        ]
        and_gates[i] = and_gate

    inputs = aag.get_data().inputs()

    output_assumptions = []

    for node_to_substitute in nodes_and_values_to_substitute:
        value_to_substitute = nodes_and_values_to_substitute[node_to_substitute]
        if node_to_substitute in inputs:
            continue
        if value_to_substitute == 0:
            output_assumptions.append(node_to_substitute)
        else:
            output_assumptions.append(node_to_substitute + 1)

    aag = AAG(from_gates_and_inputs=(inputs, aag.get_data().outputs(assumptions=output_assumptions), and_gates))

    write_to_file(
        "debug.log",
        "right after subst: %d -> %d\n%s" % (
            list(nodes_and_values_to_substitute.keys())[0],
            nodes_and_values_to_substitute[list(nodes_and_values_to_substitute.keys())[0]],
            aag.get_data().source()
        )
    )

    return (
        aag,
        solvetime,
        conflicts
    )
