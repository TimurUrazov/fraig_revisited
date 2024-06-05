from ..lec import LEC
from encoding import AAG, GatesAndInputs, AndGate
from util import call_biere_miter
import os


class AAGLEC(LEC):
    def __init__(
            self,
            left_scheme_from_file: str = None,
            right_scheme_from_file: str = None,
            left_scheme: AAG = None,
            right_scheme: AAG = None
    ):
        if left_scheme is None:
            self._left_aag = AAG(from_file=left_scheme_from_file)
            self._right_aag = AAG(from_file=right_scheme_from_file)
        else:
            self._left_aag = left_scheme
            self._right_aag = right_scheme
        super().__init__(self._left_aag, self._right_aag)
        self._lec_till_xor = None
        self._miter = None
        self._max_literal_for_subst = None

    def description(self) -> str:
        return os.path.basename(self._left_aag.filepath) + " " + os.path.basename(self._right_aag.filepath)

    @staticmethod
    def _get_and_gate_value_from_input_to_substitute(inputs_and_values_to_substitute, and_component, shift):
        if and_component not in inputs_and_values_to_substitute:
            return and_component + shift
        return inputs_and_values_to_substitute[and_component]

    # If we count on the fact that all gates in right scheme are ordered and each gate has minimal possible number than:
    # given two aags, we have to renumber gates of other aag
    # we should do the following operation:
    # new_gate = old_gate + delta
    # so that min(new_gate) = max_literal + 1
    # min(old_gate) = num_of_inputs + 1
    # delta = min(new_gate) - min(old_gate) = max_literal + 1 - 1 - num_of_inputs = max_literal - num_of_inputs
    def _shift_nodes_of_aag(self):
        left_scheme_data = self._left_aag.get_data()
        left_scheme_inputs = left_scheme_data.inputs()
        left_scheme_max_vertex = left_scheme_data.max_literal
        right_scheme_data = self._right_aag.get_data()
        right_scheme_inputs = right_scheme_data.inputs()
        right_scheme_outputs = right_scheme_data.outputs()
        right_scheme_and_gates = right_scheme_data.and_gates()
        max_vertex = right_scheme_data.max_literal
        n_of_and_gates = right_scheme_data.n_of_and_gates
        n_of_inputs = right_scheme_data.n_of_inputs
        n_of_outputs = right_scheme_data.n_of_outputs
        delta = left_scheme_max_vertex - right_scheme_data.n_of_inputs
        shift = delta * 2
        max_vertex += delta
        inputs_to_substitute = {}
        for i in range(len(right_scheme_inputs)):
            inputs_to_substitute[right_scheme_inputs[i]] = left_scheme_inputs[i]
            inputs_to_substitute[right_scheme_inputs[i] + 1] = left_scheme_inputs[i] + 1
        for i in range(len(right_scheme_outputs)):
            right_scheme_outputs[i] += shift
        shifted_right_scheme_and_gates = []
        for i in range(len(right_scheme_and_gates)):
            shifted_right_scheme_and_gate = [right_scheme_and_gates[i].gate_num + shift]
            for fan_in in right_scheme_and_gates[i].fan_ins:
                shifted_right_scheme_and_gate.append(
                    AAGLEC._get_and_gate_value_from_input_to_substitute(
                        inputs_to_substitute,
                        fan_in,
                        shift
                    )
                )
            shifted_right_scheme_and_gates.append(shifted_right_scheme_and_gate)

        return (
            max_vertex, n_of_and_gates, n_of_inputs, n_of_outputs, right_scheme_outputs, shifted_right_scheme_and_gates
        )

    @staticmethod
    def _negate_output_for_xor(output):
        if output % 2 == 1:
            return output - 1
        return output + 1

    @staticmethod
    def _create_xors_by_two_outputs(current_max_node_number, outputs_1, outputs_2):
        assert len(outputs_1) == len(outputs_2)

        miter_and_gates = []
        last_layer_and_xor_gates = []
        for output_1, output_2 in zip(outputs_1, outputs_2):
            current_max_node_number += 1
            miter_and_gates.append([
                current_max_node_number * 2,
                output_1,
                output_2
            ])
            current_max_node_number += 1
            miter_and_gates.append([
                current_max_node_number * 2,
                AAGLEC._negate_output_for_xor(output_1),
                AAGLEC._negate_output_for_xor(output_2)
            ])
            current_max_node_number += 1
            miter_and_gates.append([
                current_max_node_number * 2,
                current_max_node_number * 2 - 1,
                current_max_node_number * 2 - 3
            ])
            last_layer_and_xor_gates.append(current_max_node_number * 2)

        return miter_and_gates, last_layer_and_xor_gates, current_max_node_number * 2

    def _merge_schemes_till_xor(self) -> GatesAndInputs:
        left_aag_data = self._left_aag.get_data()
        max_vertex_2, n_of_and_gates_2, n_of_inputs_2, n_of_outputs_2, outputs_2, and_gates_2 = self._shift_nodes_of_aag()
        node_number_until_xor = max_vertex_2
        miter_and_gates, last_layer_and_miter_gates, max_literal = AAGLEC._create_xors_by_two_outputs(
            node_number_until_xor,
            left_aag_data.outputs(),
            outputs_2
        )
        assert left_aag_data.n_of_inputs == n_of_inputs_2

        aag_components = (
            left_aag_data.inputs(),
            last_layer_and_miter_gates,
            left_aag_data.and_gates() + list(map(
                lambda x: AndGate(x[0], (x[1], x[2])),
                and_gates_2 + miter_and_gates
            ))
        )

        self._aag_to_cnf_1 = {}
        self._aag_to_cnf_2 = {}

        for i in range(1, left_aag_data.max_literal + 1):
            self._aag_to_cnf_1[i * 2] = i

        for i in range(left_aag_data.max_literal + 1, max_vertex_2 + 1):
            self._aag_to_cnf_2[i * 2] = i - left_aag_data.max_literal + left_aag_data.n_of_inputs

        self._max_literal_for_subst = max_literal

        return aag_components

    def aag_gate_to_first_cnf_lit(self, gate_num):
        if gate_num in self._aag_to_cnf_1:
            return self._aag_to_cnf_1[gate_num]
        return None

    def aag_gate_to_second_cnf_lit(self, gate_num):
        if gate_num in self._aag_to_cnf_2:
            return self._aag_to_cnf_2[gate_num]
        return None

    @property
    def max_literal_for_subst(self):
        return self._max_literal_for_subst

    def get_template(self) -> AAG:
        raise NotImplementedError

    def get_scheme_till_xor(self) -> AAG:
        if self._lec_till_xor is None:
            self._lec_till_xor = AAG(from_gates_and_inputs=self._merge_schemes_till_xor())
        return self._lec_till_xor

    def get_biere_miter(self, miter_filepath: str) -> AAG:
        if self._miter is None:
            self._get_biere_miter(miter_filepath)
        return self._miter

    def _get_biere_miter(self, miter_filepath: str):
        call_biere_miter(self.left_scheme.filepath, self.right_scheme.filepath, miter_filepath)
        self._miter = AAG(from_file=miter_filepath)

    @staticmethod
    def _miter_ands_after_xors_iteration(
            max_node_number,
            last_layer_and_miter_gates,
            miter_and_gates,
            new_last_layer_and_miter_gates,
    ):
        current_max_node_number = max_node_number
        for i in range(0, len(last_layer_and_miter_gates), 2):
            current_max_node_number += 1
            if i + 1 < len(last_layer_and_miter_gates):
                miter_and_gates.append([
                    current_max_node_number * 2,
                    last_layer_and_miter_gates[i],
                    last_layer_and_miter_gates[i + 1]
                ])
                new_last_layer_and_miter_gates.append(current_max_node_number * 2)
            else:
                new_last_layer_and_miter_gates.append(last_layer_and_miter_gates[i])
        last_layer_and_miter_gates = new_last_layer_and_miter_gates
        return current_max_node_number, last_layer_and_miter_gates

    def get_handmade_miter(self) -> AAG:
        if self._miter is None:
            self.get_scheme_till_xor()
            lec_till_xor = self._lec_till_xor
            inputs = lec_till_xor.get_data().inputs()
            last_layer_and_miter_gates = lec_till_xor.get_data().outputs()
            miter_and_gates = lec_till_xor.get_data().and_gates()

            additional_node = None

            for i in range(len(last_layer_and_miter_gates)):
                last_layer_and_miter_gates[i] += 1

            new_last_layer_and_miter_gates = []
            current_max_node_number = lec_till_xor.get_data().max_literal

            current_max_node_number, last_layer_and_miter_gates = self._miter_ands_after_xors_iteration(
                current_max_node_number,
                last_layer_and_miter_gates,
                miter_and_gates,
                new_last_layer_and_miter_gates,
            )

            while len(last_layer_and_miter_gates) > 1:
                new_last_layer_and_miter_gates = []
                current_max_node_number, last_layer_and_miter_gates = self._miter_ands_after_xors_iteration(
                    current_max_node_number,
                    last_layer_and_miter_gates,
                    miter_and_gates,
                    new_last_layer_and_miter_gates,
                )

            if additional_node is not None:
                current_max_node_number += 1
                miter_and_gates.append([
                    current_max_node_number * 2,
                    additional_node,
                    last_layer_and_miter_gates[0]
                ])

            self._miter = AAG(
                from_gates_and_inputs=(inputs, [current_max_node_number * 2 + 1], miter_and_gates)
            )

        return self._miter
