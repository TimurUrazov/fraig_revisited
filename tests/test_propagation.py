import unittest
from encoding import AAG, Propagation
from util import WorkPath, solve_cnf_with_kissat, analyze_result
from copy import copy
from scheme import AAGLEC
from aig_substitutions import NonExistentAssignment, ControversialAssignment, ConflictAssignment
from util import call_aag_to_dot


class TestPropagation(unittest.TestCase):
    def test_non_existent_propagation_simple(self):
        aag = AAG(from_gates_and_inputs=([2, 4], [6, 8], [[6, 2, 4], [8, 2, 4], [10, 7, 9], [12, 6, 8], [14, 10, 12]]))
        self.assertRaises(
            NonExistentAssignment,
            Propagation(aag=aag, constraints=set(), conflicts_limit=10)._assign_node, 14, 1
        )
        Propagation(aag=aag, constraints=set(), conflicts_limit=10)._assign_node(14, 0)

    def test_controversial_assignment_simple(self):
        aag = AAG(from_gates_and_inputs=([2, 4, 6], [8], [[6, 2, 5], [8, 4, 6]]))
        self.assertRaises(
            ControversialAssignment,
            Propagation(aag=aag, constraints=set(), conflicts_limit=10)._assign, 8, 1
        )
        Propagation(aag=aag, constraints=set(), conflicts_limit=10)._assign(8, 0)

    def test_simple_propagation_without_constraints(self):
        aag = AAG(from_gates_and_inputs=([2, 4], [8], [[6, 2, 5], [8, 4, 6]]))
        Propagation(aag=aag, constraints=set(), conflicts_limit=10).propagate(2, 1)
        aag_lec_cnf = aag.to_cnf()
        result = solve_cnf_with_kissat(
            aag_lec_cnf.get_data().source(supplements=([], [aag_lec_cnf.get_data().get_outputs()]))
        )
        answer, _, _ = analyze_result(result)
        self.assertEqual(answer, "UNSAT")

    def test_assignment_simple_1(self):
        aag = AAG(from_gates_and_inputs=([2, 4], [6], [[6, 2, 4]]))
        Propagation(aag=aag, constraints=set(), conflicts_limit=10).propagate(6, 1)
        self.assertEqual(aag.get_data().and_gates(), [])

    def test_assignment_simple_2(self):
        aag = AAG(from_gates_and_inputs=([2, 4], [6], [[6, 2, 4]]))
        Propagation(aag=aag, constraints=set(), conflicts_limit=10).propagate(2, 0)
        self.assertEqual(aag.get_data().inputs(), [4])

    def test_assignment_simple_3(self):
        aag = AAG(from_gates_and_inputs=([2, 4], [6], [[6, 2, 4]]))
        Propagation(aag=aag, constraints=set(), conflicts_limit=10).propagate(2, 1)
        self.assertEqual(len(aag.get_data().inputs()), 1)
        self.assertEqual(len(aag.get_data().outputs()), 1)
        self.assertEqual(aag.get_data().inputs(), aag.get_data().outputs())
        self.assertTrue(aag.get_data().inputs()[0] in [4, 6])

    def test_constraints_simple_1(self):
        aag = AAG(from_gates_and_inputs=([2, 4], [6], [[6, 2, 4]]))
        Propagation(aag=aag, constraints=set(), conflicts_limit=10).propagate(6, 0)
        print(aag.get_data().inputs())

    def test_assignment_branching_unsat(self):
        root_path = WorkPath('tests')
        d_v_k = AAGLEC(
            left_scheme_from_file=root_path.to_file('dadda5x5.aag', 'encodings'),
            right_scheme_from_file=root_path.to_file('karatsuba5x5.aag', 'encodings')
        ).get_scheme_till_xor()

        for i in range(2, 934*2 + 2, 2):
            try:
                d_v_k_propagated = Propagation(aag=copy(d_v_k), constraints=set(), conflicts_limit=10).propagate(
                    i, 0
                )
                aag_lec_cnf = d_v_k_propagated.to_cnf_with_constraints()
                result = solve_cnf_with_kissat(
                    aag_lec_cnf.get_data().source(supplements=([], [aag_lec_cnf.get_data().get_outputs()]))
                )
                answer, _, _ = analyze_result(result)
                self.assertEqual(answer, "UNSAT")
            except ConflictAssignment as inc:
                continue

    def test_assignment_unsat_or_incorrect_assignment(self):
        root_path = WorkPath('tests')
        d_v_k = AAGLEC(
            left_scheme_from_file=root_path.to_file('dadda7x7.aag', 'encodings'),
            right_scheme_from_file=root_path.to_file('karatsuba7x7.aag', 'encodings'),
        ).get_scheme_till_xor()
        root_path = WorkPath('out')
        d_v_k.write_raw_data(root_path.to_file("example.aag"))
        call_aag_to_dot("example")
        incorrect_assignments = 0
        for i in range(2, 1870, 2):
            try:
                d_v_k_propagated = Propagation(aag=copy(d_v_k), constraints=set(), conflicts_limit=10).propagate(
                    i, 0
                )
                aag_lec_cnf = d_v_k_propagated.to_cnf_with_constraints()
                result = solve_cnf_with_kissat(
                    aag_lec_cnf.get_data().source(supplements=([], [aag_lec_cnf.get_data().get_outputs()]))
                )
                answer, _, _ = analyze_result(result)
                self.assertEqual(answer, "UNSAT")
            except ConflictAssignment:
                incorrect_assignments += 1

    def test_multilevel_assignment(self):
        root_path = WorkPath('tests')
        d_v_k = AAGLEC(
            left_scheme_from_file=root_path.to_file('dadda5x5.aag', 'encodings'),
            right_scheme_from_file=root_path.to_file('karatsuba5x5.aag', 'encodings'),
        ).get_scheme_till_xor()

        d_v_k_propagated_left = Propagation(aag=copy(d_v_k), constraints=set(), conflicts_limit=10).propagate(
            100, 0
        )

        d_v_k_propagated_right = Propagation(aag=copy(d_v_k), constraints=set(), conflicts_limit=10).propagate(
            100, 1
        )

        d_v_k_propagated_left_left = Propagation(
            aag=copy(d_v_k_propagated_left),
            constraints=set(), conflicts_limit=10
        ).propagate(
            854, 0
        )

        d_v_k_propagated_left_right = Propagation(
            aag=copy(d_v_k_propagated_left),
            constraints=set(), conflicts_limit=10
        ).propagate(
            800, 1
        )

        d_v_k_propagated_right_left = Propagation(
            aag=copy(d_v_k_propagated_right),
            constraints=set(), conflicts_limit=10
        ).propagate(
            854, 0
        )

        d_v_k_propagated_right_right = Propagation(
            aag=copy(d_v_k_propagated_right),
            constraints=set(), conflicts_limit=10
        ).propagate(
            854, 1
        )

        aag_lec_cnf = d_v_k.to_cnf()
        result = solve_cnf_with_kissat(
            aag_lec_cnf.get_data().source(supplements=([], [aag_lec_cnf.get_data().get_outputs()]))
        )
        answer, _, _ = analyze_result(result)
        self.assertEqual(answer, "UNSAT")

        aag_lec_cnf = d_v_k_propagated_left.to_cnf_with_constraints()
        result = solve_cnf_with_kissat(
            aag_lec_cnf.get_data().source(supplements=([], [aag_lec_cnf.get_data().get_outputs()]))
        )
        answer, _, _ = analyze_result(result)
        self.assertEqual(answer, "UNSAT")

        aag_lec_cnf = d_v_k_propagated_right.to_cnf_with_constraints()
        result = solve_cnf_with_kissat(
            aag_lec_cnf.get_data().source(supplements=([], [aag_lec_cnf.get_data().get_outputs()]))
        )
        answer, _, _ = analyze_result(result)
        self.assertEqual(answer, "UNSAT")

        aag_lec_cnf = d_v_k_propagated_left_left.to_cnf_with_constraints()
        result = solve_cnf_with_kissat(
            aag_lec_cnf.get_data().source(supplements=([], [aag_lec_cnf.get_data().get_outputs()]))
        )
        answer, _, _ = analyze_result(result)
        self.assertEqual(answer, "UNSAT")

        aag_lec_cnf = d_v_k_propagated_left_right.to_cnf_with_constraints()
        result = solve_cnf_with_kissat(
            aag_lec_cnf.get_data().source(supplements=([], [aag_lec_cnf.get_data().get_outputs()]))
        )
        answer, _, _ = analyze_result(result)
        self.assertEqual(answer, "UNSAT")

        aag_lec_cnf = d_v_k_propagated_right_left.to_cnf_with_constraints()
        result = solve_cnf_with_kissat(
            aag_lec_cnf.get_data().source(supplements=([], [aag_lec_cnf.get_data().get_outputs()]))
        )
        answer, _, _ = analyze_result(result)
        self.assertEqual(answer, "UNSAT")

        aag_lec_cnf = d_v_k_propagated_right_right.to_cnf_with_constraints()
        result = solve_cnf_with_kissat(
            aag_lec_cnf.get_data().source(supplements=([], [aag_lec_cnf.get_data().get_outputs()]))
        )
        answer, _, _ = analyze_result(result)
        self.assertEqual(answer, "UNSAT")

    def test_factorisation_simple(self):
        root_path = WorkPath('tests')
        dadda = AAG(
            from_file=root_path.to_file('dadda3x3.aag', 'encodings')
        )
        aag_lec_cnf = dadda.to_cnf()
        cnf_outputs = aag_lec_cnf.get_data().get_outputs()
        print(cnf_outputs)
        result = solve_cnf_with_kissat(
            aag_lec_cnf.get_data().source(
                supplements=(
                    [1, -cnf_outputs[0], cnf_outputs[1], -cnf_outputs[2], cnf_outputs[3], -cnf_outputs[4], cnf_outputs[5]],
                    []
                )
            )
        )
        print(result)
        answer, _, _ = analyze_result(result)
        self.assertEqual(answer, "SAT")

    def test_factorisation_cnc(self):
        root_path = WorkPath('tests')
        dadda = AAG(
            from_file=root_path.to_file('karatsuba3x3.aag', 'encodings')
        )

        dadda_propagated_left = Propagation(aag=copy(dadda), constraints=set(), conflicts_limit=10).propagate(
            20, 0
        )

        aag_lec_cnf = dadda_propagated_left.to_cnf_with_constraints()
        cnf_outputs = aag_lec_cnf.get_data().get_outputs()

        result = solve_cnf_with_kissat(
            aag_lec_cnf.get_data().source(supplements=([-cnf_outputs[0], cnf_outputs[1], -cnf_outputs[2], cnf_outputs[3], -cnf_outputs[4], cnf_outputs[5]], []))
        )
        answer, _, _ = analyze_result(result)
        self.assertEqual(answer, "SAT")

        prop = Propagation(aag=copy(dadda), constraints=set(), conflicts_limit=10)

        dadda_propagated_right = prop.propagate(20, 1)

        aag_lec_cnf = dadda_propagated_right.to_cnf_with_constraints()
        cnf_outputs = aag_lec_cnf.get_data().get_outputs()
        result = solve_cnf_with_kissat(
            aag_lec_cnf.get_data().source(supplements=([-cnf_outputs[0], cnf_outputs[1], -cnf_outputs[2], cnf_outputs[3], -cnf_outputs[4], cnf_outputs[5]], []))
        )
        answer, _, _ = analyze_result(result)
        self.assertEqual(answer, "SAT")

