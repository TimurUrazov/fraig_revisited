import unittest
from typing import Mapping, List
from encoding import AAG, Reducer, Gate, Simulator, CNF, Sweep
from util import WorkPath, solve_cnf_with_kissat, analyze_result
from copy import copy
from scheme import AAGLEC
from aig_substitutions import renumber_gates_of_aag
from util import write_aag_to_out_dir, CONFIG


class TestReducer(unittest.TestCase):
    def test_reducer_simple(self):
        aag = AAG(from_gates_and_inputs=([2, 4], [6, 8], [[6, 2, 4], [8, 2, 4]]))
        Reducer(aag=aag, constraints=set()).reduce()
        self.assertEqual(len(aag.get_data().outputs()), 2)
        self.assertTrue(aag.get_data().outputs()[0] in [1, 8])
        self.assertEqual(aag.get_data().outputs()[0], aag.get_data().outputs()[1])

    def test_output_negation_after_normalization_simple(self):
        aag = AAG(from_gates_and_inputs=([2, 4], [10], [[10, 7, 9], [6, 2, 4], [8, 2, 4]]))
        Reducer(aag=aag, constraints=set()).reduce()
        self.assertEqual(len(aag.get_data().outputs()), 1)
        self.assertTrue(aag.get_data().outputs()[0] in [7, 9, 11])

    def test_output_negation_after_merging_simple(self):
        aag = AAG(from_gates_and_inputs=([2, 4], [6, 12], [[6, 2, 4], [8, 3, 5], [10, 3, 5], [12, 8, 10]]))
        Reducer(aag=aag, constraints=set()).reduce()
        self.assertEqual(len(aag.get_data().outputs()), 2)
        self.assertEqual(aag.get_data().outputs()[0], 6)
        self.assertTrue(aag.get_data().outputs()[1] in [8, 13, 10])

    def test_equal_after_reduce_simple(self):
        outputs_before = [6, 8]
        aag = AAG(from_gates_and_inputs=([2, 4], outputs_before, [[6, 2, 4], [8, 2, 4]]))
        aag_before = copy(aag)
        Reducer(aag=aag, constraints=set()).reduce()
        aag_lec_cnf = AAGLEC(
            left_scheme=aag_before,
            right_scheme=aag
        ).get_scheme_till_xor().to_cnf()
        result = solve_cnf_with_kissat(
            aag_lec_cnf.get_data().source(supplements=([], [aag_lec_cnf.get_data().get_outputs()]))
        )
        answer, _, _ = analyze_result(result)
        self.assertEqual(answer, "UNSAT")

    def test_equal_after_reduce(self):
        outputs_before = [6, 8]
        aag = AAG(from_gates_and_inputs=([2, 4], outputs_before, [[6, 2, 4], [8, 2, 4]]))
        aag_before = copy(aag)
        Reducer(aag=aag, constraints=set()).reduce()
        aag_data = aag.get_data()
        new_gates, _, inputs, outputs = renumber_gates_of_aag(
            and_gates=aag_data.and_gates(),
            inputs=aag_data.inputs(),
            outputs=aag_data.outputs()
        )
        new_aag = AAG(from_gates_and_inputs=(inputs, outputs, new_gates))
        aag_lec_cnf = AAGLEC(
            left_scheme=aag_before,
            right_scheme=new_aag
        ).get_scheme_till_xor().to_cnf()
        result = solve_cnf_with_kissat(
            aag_lec_cnf.get_data().source(supplements=([], [aag_lec_cnf.get_data().get_outputs()]))
        )
        answer, _, _ = analyze_result(result)
        self.assertEqual(answer, "UNSAT")

    def test_d_v_k_2x2(self):
        root_path = WorkPath('tests')
        d_v_k = AAGLEC(
            left_scheme_from_file=root_path.to_file('dadda2x2.aag', 'encodings'),
            right_scheme_from_file=root_path.to_file('karatsuba2x2.aag', 'encodings')
        ).get_scheme_till_xor()
        aag_before = copy(d_v_k)
        Reducer(aag=d_v_k, constraints=set()).reduce()
        aag_data = d_v_k.get_data()
        new_gates, _, inputs, outputs = renumber_gates_of_aag(
            and_gates=aag_data.and_gates(),
            inputs=aag_data.inputs(),
            outputs=aag_data.outputs()
        )
        new_aag = AAG(from_gates_and_inputs=(inputs, outputs, new_gates))
        aag_lec_cnf = AAGLEC(
            left_scheme=aag_before,
            right_scheme=new_aag
        ).get_scheme_till_xor().to_cnf()
        result = solve_cnf_with_kissat(
            aag_lec_cnf.get_data().source(supplements=([], [aag_lec_cnf.get_data().get_outputs()]))
        )
        answer, _, _ = analyze_result(result)
        self.assertEqual(answer, "UNSAT")

    def test_d_v_k_3x3(self):
        root_path = WorkPath('tests')
        d_v_k = AAGLEC(
            left_scheme_from_file=root_path.to_file('dadda3x3.aag', 'encodings'),
            right_scheme_from_file=root_path.to_file('karatsuba3x3.aag', 'encodings')
        ).get_scheme_till_xor()
        aag_before = copy(d_v_k)
        Reducer(aag=d_v_k, constraints=set()).reduce(conflict_limit=4)
        aag_data = d_v_k.get_data()

        new_gates, _, inputs, outputs = renumber_gates_of_aag(
            and_gates=aag_data.and_gates(),
            inputs=aag_data.inputs(),
            outputs=aag_data.outputs()
        )
        new_aag = AAG(from_gates_and_inputs=(inputs, outputs, new_gates))

        aag_lec = AAGLEC(left_scheme=aag_before, right_scheme=new_aag).get_scheme_till_xor()

        aag_lec_cnf = aag_lec.to_cnf()

        result = solve_cnf_with_kissat(
            aag_lec_cnf.get_data().source(supplements=([], [aag_lec_cnf.get_data().get_outputs()]))
        )
        answer, _, _ = analyze_result(result)
        self.assertEqual(answer, "UNSAT")

    def test_d_v_k_5x5(self):
        root_path = WorkPath('tests')
        d_v_k = AAGLEC(
            left_scheme_from_file=root_path.to_file('dadda5x5.aag', 'encodings'),
            right_scheme_from_file=root_path.to_file('karatsuba5x5.aag', 'encodings')
        ).get_scheme_till_xor()
        aag_before = copy(d_v_k)
        Reducer(aag=d_v_k, constraints=set()).reduce(conflict_limit=10000)
        aag_data = d_v_k.get_data()
        new_gates, _, inputs, outputs = renumber_gates_of_aag(
            and_gates=aag_data.and_gates(),
            inputs=aag_data.inputs(),
            outputs=aag_data.outputs()
        )
        new_aag = AAG(from_gates_and_inputs=(inputs, outputs, new_gates))
        aag_lec = AAGLEC(left_scheme=aag_before, right_scheme=new_aag).get_scheme_till_xor()
        aag_lec_cnf = aag_lec.to_cnf()

        result = solve_cnf_with_kissat(
            aag_lec_cnf.get_data().source(supplements=([], [aag_lec_cnf.get_data().get_outputs()]))
        )
        answer, _, _ = analyze_result(result)
        self.assertEqual(answer, "UNSAT")

    @staticmethod
    def check_equivalence(
            parent_node: int,
            initial_graph: Mapping[int, Gate],
            h_class: List[int],
            subgraph_c: Mapping[int, List[List[int]]],
            current_node: int,
            edge: int
    ) -> bool:
        if parent_node not in h_class or current_node not in h_class:
            return False

        constraints = []
        visited = set()

        def dfs(node_number: int):
            if node_number in visited:
                return
            visited.add(node_number)

            assert initial_graph[node_number].in_degree % 2 == 0

            constraints.extend(subgraph_c[node_number])

            for node_num, _ in initial_graph[node_number].parents:
                dfs(node_num)

        dfs(current_node)
        dfs(parent_node)

        new_constraints, current_max_node_number, new_node_1_number, new_node_2_number = Reducer._renumber_constraints(
            constraints, current_node, parent_node
        )

        max_node_number = Reducer._create_xors_by_two_outputs(
            current_max_node_number,
            new_node_1_number,
            new_node_2_number,
            new_constraints
        )

        # check if nodes are functionally equivalent: fix xor as 1 and this is unsatisfiable
        new_constraints.append([max_node_number if edge == 0 else -max_node_number])

        result = solve_cnf_with_kissat(
            cnf_str=CNF(from_clauses=new_constraints).get_data().source(),
            conflicts_limit=10000
        )

        answer, _, _ = analyze_result(
            result, 'testing nodes with one parent', print_output=False
        )

        return answer == "UNSAT"

    # Sometimes conflict limit doesn't allow us to check whether nodes are equivalent or equivalent up to negation,
    # and we have nodes with one parent this doesn't preserve the invariant. So we have to proceed reduction.
    # Let's check that nodes with one parent and its parent are equal or equal up to negation
    def test_d_v_k_3x3_merge_not_up_to_equivalent(self):
        root_path = WorkPath('tests')
        d_v_k = AAGLEC(
            left_scheme_from_file=root_path.to_file('dadda3x3.aag', 'encodings'),
            right_scheme_from_file=root_path.to_file('karatsuba3x3.aag', 'encodings')
        ).get_scheme_till_xor()
        graph_before = copy(d_v_k).get_graph()
        num_of_simulate_samples = 64 * 127
        one = 2**num_of_simulate_samples - 1
        reducer = Reducer(
            aag=d_v_k,
            constraints=set(),
            num_of_simulate_samples=num_of_simulate_samples,
            include_complementary=True
        )
        subgraph_constraints = reducer._build_subgraph_constraints()
        reducer._merge_equivalent(conflict_limit=11)
        simulator = Simulator(graph_before)
        hash_classes = simulator.simulate(num_of_simulate_samples, include_complementary=True)
        random_numbers = simulator._random_numbers
        for node in d_v_k._graph:
            if d_v_k._graph[node].in_degree == 1:
                if random_numbers[node] in hash_classes:
                    hash_class = hash_classes[random_numbers[node]]
                else:
                    complementary_hash = one ^ random_numbers[node]
                    assert complementary_hash in hash_classes
                    hash_class = hash_classes[complementary_hash]
                self.assertTrue(TestReducer.check_equivalence(
                    node, graph_before, hash_class, subgraph_constraints, *d_v_k._graph[node].parents[0]
                ))

    def test_d_v_k_3x3_merge_up_to_equivalent(self):
        root_path = WorkPath('tests')
        d_v_k = AAGLEC(
            left_scheme_from_file=root_path.to_file('dadda3x3.aag', 'encodings'),
            right_scheme_from_file=root_path.to_file('karatsuba3x3.aag', 'encodings')
        ).get_scheme_till_xor()
        graph_before = copy(d_v_k).get_graph()
        num_of_simulate_samples = 64 * 127
        one = 2**num_of_simulate_samples - 1
        reducer = Reducer(
            aag=d_v_k,
            constraints=set(),
            num_of_simulate_samples=num_of_simulate_samples,
            include_complementary=False
        )
        subgraph_constraints = reducer._build_subgraph_constraints()
        reducer._merge_equivalent(conflict_limit=11)
        simulator = Simulator(graph_before)
        hash_classes = simulator.simulate(num_of_simulate_samples, include_complementary=False)
        random_numbers = simulator._random_numbers
        for node in d_v_k._graph:
            if d_v_k._graph[node].in_degree == 1:
                hash_class = hash_classes[random_numbers[node]]
                complementary_hash = one ^ random_numbers[node]
                assert complementary_hash in hash_classes
                hash_class.extend(hash_classes[complementary_hash])
                self.assertTrue(TestReducer.check_equivalence(
                    node, graph_before, hash_class, subgraph_constraints, *d_v_k._graph[node].parents[0]
                ))

    def test_d_v_k_5x5_merge_not_up_to_equivalent(self):
        root_path = WorkPath('tests')
        d_v_k = AAGLEC(
            left_scheme_from_file=root_path.to_file('dadda5x5.aag', 'encodings'),
            right_scheme_from_file=root_path.to_file('karatsuba5x5.aag', 'encodings')
        ).get_scheme_till_xor()
        graph_before = copy(d_v_k).get_graph()
        num_of_simulate_samples = 64 * 127
        one = 2 ** num_of_simulate_samples - 1
        reducer = Reducer(
            aag=d_v_k,
            constraints=set(),
            num_of_simulate_samples=num_of_simulate_samples,
            include_complementary=False
        )
        subgraph_constraints = reducer._build_subgraph_constraints()
        reducer._merge_equivalent(conflict_limit=2000)
        simulator = Simulator(graph_before)
        hash_classes = simulator.simulate(num_of_simulate_samples, include_complementary=False)
        random_numbers = simulator._random_numbers
        for node in d_v_k._graph:
            if d_v_k._graph[node].in_degree == 1:
                hash_class = hash_classes[random_numbers[node]]
                complementary_hash = one ^ random_numbers[node]
                assert complementary_hash in hash_classes
                hash_class.extend(hash_classes[complementary_hash])
                self.assertTrue(TestReducer.check_equivalence(
                    node, graph_before, hash_class, subgraph_constraints, *d_v_k._graph[node].parents[0]
                ))

    def test_d_v_k_7x7_merge_not_up_to_equivalent(self):
        root_path = WorkPath('tests')
        d_v_k = AAGLEC(
            left_scheme_from_file=root_path.to_file('dadda7x7.aag', 'encodings'),
            right_scheme_from_file=root_path.to_file('karatsuba7x7.aag', 'encodings')
        ).get_scheme_till_xor()
        graph_before = copy(d_v_k).get_graph()
        num_of_simulate_samples = 64 * 127
        one = 2 ** num_of_simulate_samples - 1
        reducer = Reducer(
            aag=d_v_k,
            constraints=set(),
            num_of_simulate_samples=num_of_simulate_samples,
            include_complementary=False
        )
        subgraph_constraints = reducer._build_subgraph_constraints()
        reducer._merge_equivalent(conflict_limit=300)
        simulator = Simulator(graph_before)
        hash_classes = simulator.simulate(num_of_simulate_samples, include_complementary=False)
        random_numbers = simulator._random_numbers
        for node in d_v_k._graph:
            if d_v_k._graph[node].in_degree == 1:
                hash_class = hash_classes[random_numbers[node]]
                complementary_hash = one ^ random_numbers[node]
                assert complementary_hash in hash_classes
                hash_class.extend(hash_classes[complementary_hash])
                self.assertTrue(TestReducer.check_equivalence(
                    node, graph_before, hash_class, subgraph_constraints, *d_v_k._graph[node].parents[0]
                ))

    def test_d_v_k_12x12_merge_not_up_to_equivalent(self):
        root_path = WorkPath('tests')
        d_v_k = AAGLEC(
            left_scheme_from_file=root_path.to_file('dadda12x12.aag', 'encodings'),
            right_scheme_from_file=root_path.to_file('column12x12.aag', 'encodings')
        ).get_scheme_till_xor()
        graph_before = copy(d_v_k).get_graph()
        num_of_simulate_samples = 64 * 127
        one = 2 ** num_of_simulate_samples - 1
        reducer = Reducer(
            aag=d_v_k,
            constraints=set(),
            num_of_simulate_samples=num_of_simulate_samples,
            include_complementary=False
        )
        subgraph_constraints = reducer._build_subgraph_constraints()
        reducer._merge_equivalent(conflict_limit=5000)
        simulator = Simulator(graph_before)
        hash_classes = simulator.simulate(num_of_simulate_samples, include_complementary=False)
        random_numbers = simulator._random_numbers
        for node in d_v_k._graph:
            if d_v_k._graph[node].in_degree == 1:
                hash_class = hash_classes[random_numbers[node]]
                complementary_hash = one ^ random_numbers[node]
                assert complementary_hash in hash_classes
                hash_class.extend(hash_classes[complementary_hash])
                self.assertTrue(TestReducer.check_equivalence(
                    node, graph_before, hash_class, subgraph_constraints, *d_v_k._graph[node].parents[0]
                ))

    def test_c_v_k_12x12(self):
        root_path = WorkPath('tests')
        d_v_k = AAGLEC(
            left_scheme_from_file=root_path.to_file('dadda12x12.aag', 'encodings'),
            right_scheme_from_file=root_path.to_file('column12x12.aag', 'encodings')
        ).get_scheme_till_xor()
        aag_before = copy(d_v_k)
        Reducer(aag=d_v_k, constraints=set()).reduce()
        aag_data = d_v_k.get_data()

        new_gates, _, inputs, outputs = renumber_gates_of_aag(
            and_gates=aag_data.and_gates(),
            inputs=aag_data.inputs(),
            outputs=aag_data.outputs()
        )
        new_aag = AAG(from_gates_and_inputs=(inputs, outputs, new_gates))

        aag_lec = AAGLEC(left_scheme=aag_before, right_scheme=new_aag).get_scheme_till_xor()

        aag_lec_cnf = aag_lec.to_cnf()

        result = solve_cnf_with_kissat(
            aag_lec_cnf.get_data().source(supplements=([], [aag_lec_cnf.get_data().get_outputs()]))
        )
        answer, _, _ = analyze_result(result)
        self.assertEqual(answer, "UNSAT")

    def test_d_v_k_7x7(self):
        root_path = WorkPath('tests')
        d_v_k = AAGLEC(
            left_scheme_from_file=root_path.to_file('dadda7x7.aag', 'encodings'),
            right_scheme_from_file=root_path.to_file('karatsuba7x7.aag', 'encodings')
        ).get_scheme_till_xor()
        aag_before = copy(d_v_k)
        Reducer(aag=d_v_k, constraints=set()).reduce(conflict_limit=1000)
        aag_data = d_v_k.get_data()
        new_gates, _, inputs, outputs = renumber_gates_of_aag(
            and_gates=aag_data.and_gates(),
            inputs=aag_data.inputs(),
            outputs=aag_data.outputs()
        )
        new_aag = AAG(from_gates_and_inputs=(inputs, outputs, new_gates))
        aag_lec = AAGLEC(left_scheme=aag_before, right_scheme=new_aag).get_scheme_till_xor()

        aag_lec_cnf = aag_lec.to_cnf()

        result = solve_cnf_with_kissat(
            aag_lec_cnf.get_data().source(supplements=([], [aag_lec_cnf.get_data().get_outputs()]))
        )
        answer, _, _ = analyze_result(result)
        self.assertEqual(answer, "UNSAT")

    def test_d_v_k_5x5_excluding_complementary(self):
        root_path = WorkPath('tests')
        d_v_k = AAGLEC(
            left_scheme_from_file=root_path.to_file('dadda5x5.aag', 'encodings'),
            right_scheme_from_file=root_path.to_file('karatsuba5x5.aag', 'encodings')
        ).get_scheme_till_xor()
        aag_before = copy(d_v_k)
        Reducer(aag=d_v_k, constraints=set(), include_complementary=False).reduce(conflict_limit=10000)
        aag_data = d_v_k.get_data()
        new_gates, _, inputs, outputs = renumber_gates_of_aag(
            and_gates=aag_data.and_gates(),
            inputs=aag_data.inputs(),
            outputs=aag_data.outputs()
        )
        new_aag = AAG(from_gates_and_inputs=(inputs, outputs, new_gates))
        aag_lec = AAGLEC(left_scheme=aag_before, right_scheme=new_aag).get_scheme_till_xor()

        aag_lec_cnf = aag_lec.to_cnf()

        result = solve_cnf_with_kissat(
            aag_lec_cnf.get_data().source(supplements=([], [aag_lec_cnf.get_data().get_outputs()]))
        )
        answer, _, _ = analyze_result(result)
        self.assertEqual(answer, "UNSAT")

    def test_d_v_k_7x7_excluding_complementary(self):
        root_path = WorkPath('tests')
        d_v_k = AAGLEC(
            left_scheme_from_file=root_path.to_file('dadda7x7.aag', 'encodings'),
            right_scheme_from_file=root_path.to_file('karatsuba7x7.aag', 'encodings')
        ).get_scheme_till_xor()
        aag_before = copy(d_v_k)
        Reducer(aag=d_v_k, constraints=set(), include_complementary=False).reduce(conflict_limit=10000)
        aag_data = d_v_k.get_data()
        new_gates, _, inputs, outputs = renumber_gates_of_aag(
            and_gates=aag_data.and_gates(),
            inputs=aag_data.inputs(),
            outputs=aag_data.outputs()
        )
        new_aag = AAG(from_gates_and_inputs=(inputs, outputs, new_gates))
        aag_lec = AAGLEC(left_scheme=aag_before, right_scheme=new_aag).get_scheme_till_xor()
        aag_lec_cnf = aag_lec.to_cnf()

        result = solve_cnf_with_kissat(
            aag_lec_cnf.get_data().source(supplements=([], [aag_lec_cnf.get_data().get_outputs()]))
        )
        answer, _, _ = analyze_result(result)
        self.assertEqual(answer, "UNSAT")

    def test_d_v_k_12x12_excluding_complementary(self):
        root_path = WorkPath('tests')
        d_v_k = AAGLEC(
            left_scheme_from_file=root_path.to_file('dadda12x12.aag', 'encodings'),
            right_scheme_from_file=root_path.to_file('column12x12.aag', 'encodings')
        ).get_scheme_till_xor()
        aag_before = copy(d_v_k)
        Reducer(aag=d_v_k, constraints=set(), include_complementary=False).reduce(conflict_limit=10000)
        aag_data = d_v_k.get_data()
        new_gates, _, inputs, outputs = renumber_gates_of_aag(
            and_gates=aag_data.and_gates(),
            inputs=aag_data.inputs(),
            outputs=aag_data.outputs()
        )
        new_aag = AAG(from_gates_and_inputs=(inputs, outputs, new_gates))
        aag_lec = AAGLEC(left_scheme=aag_before, right_scheme=new_aag).get_scheme_till_xor()

        aag_lec_cnf = aag_lec.to_cnf()

        result = solve_cnf_with_kissat(
            aag_lec_cnf.get_data().source(supplements=([], [aag_lec_cnf.get_data().get_outputs()]))
        )
        answer, _, _ = analyze_result(result)
        self.assertEqual(answer, "UNSAT")

    def test_d_v_k_3x3_lec(self):
        root_path = WorkPath('tests')
        d_v_k = AAGLEC(
            left_scheme_from_file=root_path.to_file('dadda3x3.aag', 'encodings'),
            right_scheme_from_file=root_path.to_file('karatsuba3x3.aag', 'encodings')
        ).get_scheme_till_xor()
        Reducer(aag=d_v_k, constraints=set(), include_complementary=False).reduce(conflict_limit=17)
        aag_data = d_v_k.get_data()
        new_gates, _, inputs, outputs = renumber_gates_of_aag(
            and_gates=aag_data.and_gates(),
            inputs=aag_data.inputs(),
            outputs=list(set(aag_data.outputs()))
        )
        new_aag = AAG(from_gates_and_inputs=(inputs, outputs, new_gates))
        aag_lec_cnf = new_aag.to_cnf()
        result = solve_cnf_with_kissat(
            aag_lec_cnf.get_data().source(supplements=([], [aag_lec_cnf.get_data().get_outputs()]))
        )
        answer, _, _ = analyze_result(result)
        self.assertEqual(answer, "UNSAT")

    def test_reducing_degree_after_several_propagations(self):
        aag_source = """aag 1982 2 0 10 707
1988
3724
3910
3916
3922
3928
3934
3940
3946
3952
3958
3964
404 3724 3725
406 3724 3725
1822 1989 3724
1824 1988 3724
1826 1989 3725
550 3724 3725
556 3724 3725
1838 1988 1989
566 3724 3725
568 3724 3725
1850 1988 1989
3910 3724 3725
1992 1989 3725
1994 1988 3724
1242 1988 1989
2010 1989 3724
1244 1988 1989
1252 1988 1989
3300 1988 3725
486 1988 1989
488 1988 1989
3704 3724 3725
3706 3724 3725
1376 1245 1243
674 489 487
1828 1825 1827
1860 1823 1851
1254 1253 1989
1884 1838 1988
1862 1822 1850
586 407 405
1866 1822 1851
1868 1839 1988
2012 1988 2011
558 551 556
1870 1838 1989
2064 1993 1995
2032 1988 2010
788 567 569
3708 3707 3705
2014 1989 2010
2108 2032 2064
2110 2033 2065
580 559 587
3912 3709 789
1864 1863 1861
3914 3708 788
2634 1828 3724
588 559 586
2636 1829 3725
590 558 587
2122 2032 2065
2640 1828 3725
1872 1869 1871
2388 2064 3725
2390 2065 3724
1880 1866 1988
1882 1866 1838
2400 2064 3724
1378 1376 1988
1380 1377 1989
2020 2013 2015
1386 1377 1989
1274 1254 1988
810 591 589
2102 2021 3724
2104 2020 3725
2112 2111 2109
2114 2021 3725
2244 2033 2021
2372 2033 2020
2246 2032 2020
3916 3915 3913
2124 2021 3725
2638 2635 2637
2128 2021 3725
1874 1866 1872
2642 1864 2641
1876 1867 1873
2644 1865 2640
2648 1864 2640
2392 2389 2391
602 580 3724
1886 1885 1883
610 580 3725
612 581 3724
1382 1379 1381
1390 1387 1988
1264 1274 1989
1266 1275 1988
1272 1275 1988
2706 2639 3725
832 613 611
2116 2115 2113
2118 2114 2112
2248 2245 2247
1486 1383 1989
2126 2124 2032
1488 1382 1988
2130 2128 2065
1878 1877 1875
2646 2645 2643
2138 2103 2105
1506 1383 1988
1900 1881 1886
632 602 1989
1392 1390 1376
624 602 1988
1398 1267 1265
634 603 1988
2680 2638 3725
2682 2639 3724
1276 1273 1274
3200 1489 1487
646 624 1989
654 624 1988
656 625 1989
1298 1276 1989
1432 1393 1399
2724 2647 2706
2864 2681 2683
2132 2127 2131
856 3724 833
2650 1878 2649
2652 1879 2648
2268 2020 2139
2270 2021 2138
2656 1878 2648
2144 2119 2117
2280 2021 2139
2290 2021 2139
2294 2021 2139
1400 1392 1399
1402 1393 1398
636 635 633
1422 1433 1298
2190 2123 2132
1424 1432 1299
1430 1432 1299
668 646 674
676 646 675
678 647 674
2868 3725 2865
2870 3724 2864
844 636 3724
846 637 3725
2904 3724 2865
858 637 833
860 637 3724
2654 2651 2653
2658 2657 1900
2660 2656 1901
870 657 655
2408 2269 2271
1514 1403 1401
2154 2139 2145
2156 2139 2144
2670 2656 1900
2158 2138 2145
2296 2294 2249
1426 1423 1425
1434 1431 1433
2172 2145 2190
690 668 1988
698 668 1989
700 669 1988
848 845 847
2516 2408 2065
3028 2869 2871
2518 2409 2064
2522 2409 2065
862 861 859
1508 1506 1515
1510 1507 1514
2662 2659 2661
1516 1506 1515
2160 2157 2159
2170 2145 2191
892 677 679
2174 2144 2191
1536 1427 1988
2182 2175 2173
3208 1509 1511
914 701 699
2200 2171 2191
2202 2170 2190
932 691 1988
934 690 1989
1448 1434 1298
2216 2170 2191
948 690 1988
850 848 833
852 849 832
3284 1989 3029
3286 1988 3028
2520 2517 2519
3296 1988 3029
2274 2161 2248
3298 3725 3029
2276 2160 2249
878 857 862
1520 1517 1507
2288 2161 2249
2292 2290 2161
2720 2647 2521
2722 2521 2706
2278 2277 2275
3302 3301 3299
936 933 935
872 871 879
874 870 878
880 871 879
2192 2155 2183
1522 1520 1514
2194 2154 2182
2708 2646 2521
854 851 853
2198 2154 2183
2710 2647 2520
2298 2293 2297
2204 2203 2201
3288 3287 3285
1538 1523 1427
1540 1523 1988
2196 2193 2195
2712 2711 2709
2206 2198 2204
2208 2199 2205
2212 2198 2191
2214 2198 2170
2726 2723 2725
3290 3289 3724
3292 3288 3725
3304 3297 3302
2282 2279 2281
876 875 873
2284 2278 2280
1524 1522 1988
884 881 870
1526 1523 1989
2300 2298 2289
2304 2196 2373
2210 2209 2207
900 884 878
1542 1541 1539
2728 2726 2721
2314 2197 2373
2218 2215 2217
2316 2301 2197
3322 3305 3200
2318 2301 2373
2286 2285 2283
3732 3293 3291
1528 1525 1527
2714 2713 2707
2716 2712 2706
2302 2197 2372
896 892 900
2306 2303 2305
2436 2287 3725
902 893 901
1550 1542 1537
3726 3724 3732
2320 2317 2319
3728 3725 3733
2324 2211 2372
2326 2210 2373
3734 3724 3732
2336 2211 2373
2232 2213 2218
2746 2655 2729
2892 2717 2715
2410 2287 3724
3562 3725 3733
2412 2286 3725
1530 1427 1528
1532 1426 1529
894 893 901
898 897 895
2308 2307 2300
2310 2306 2301
906 903 892
3216 1531 1533
1552 1550 1448
1554 1551 1449
2322 2315 2320
3730 3727 3729
2454 2393 2436
2328 2327 2325
3738 3735 3725
1566 1551 1448
2346 2372 2233
2348 2373 2232
2358 2373 2233
2894 2905 2892
2896 2904 2893
2902 2904 2893
2528 2411 2413
2338 2211 2323
2530 2523 2529
2340 2323 2373
2532 2522 2528
3746 3738 3733
2312 2309 2311
2330 2329 2322
2350 2349 2347
3918 3731 811
2544 2522 2529
3920 3730 810
2898 2895 2897
3224 1553 1555
922 906 900
2332 2328 2323
2906 2903 2905
2562 2021 2544
2534 2531 2533
2438 2313 2392
2440 2312 2393
2342 2339 2341
3046 2899 3725
3048 2898 3724
2926 2906 2892
3922 3921 3919
2450 2313 2393
916 915 923
2546 2021 2545
918 914 922
2452 2313 2436
2548 2020 2544
3060 2899 3724
924 915 923
2334 2331 2333
928 925 914
3078 1829 3060
2344 2337 2342
2472 2335 2401
2442 2439 2441
2730 2654 2535
2732 2655 2534
3050 3049 3047
2744 2535 2729
920 919 917
2742 2655 2535
2550 2547 2549
2456 2453 2455
3062 1829 3061
3064 1828 3060
2460 2335 2400
2462 2334 2401
2464 2463 2461
3320 3051 3305
930 928 922
3066 3065 3063
3306 3051 3201
2444 2443 2437
2362 2345 2373
2446 2442 2436
2734 2731 2733
2352 2344 2351
3308 3050 3200
2354 2345 2350
3318 3051 3200
2360 2345 2233
2458 2456 2451
2748 2747 2745
2466 2458 2465
2468 2459 2464
938 936 931
2474 2459 2335
940 937 930
2476 2459 2401
3310 3307 3309
944 931 1988
2448 2445 2447
946 931 690
2736 2735 2728
2738 2734 2729
2356 2353 2355
3324 3321 3323
2364 2361 2363
2750 2748 2743
2560 2449 2544
2914 2739 2737
3326 3319 3324
2494 2357 2401
2470 2469 2467
2558 2449 2021
2478 2475 2477
942 941 939
2768 2751 2663
3312 3311 3304
2482 2357 2400
3314 3310 3305
2484 2356 2401
950 949 947
2552 2449 2550
2554 2448 2551
2366 2359 2364
2564 2561 2563
2916 2926 2914
2374 2366 2373
2918 2927 2915
2376 2367 2372
2924 2927 2915
2480 2473 2478
3344 3327 3208
2580 2471 2032
3316 3313 3315
2486 2485 2483
952 945 950
2556 2553 2555
2496 2481 2357
2752 2557 2662
2754 2556 2663
2498 2481 2401
2566 2559 2564
3558 3317 3725
3560 3317 3733
2920 2919 2917
2378 2375 2377
2764 2557 2663
2766 2751 2557
2928 2925 2926
3546 3317 3724
2488 2487 2480
2490 2486 2481
3548 3316 3725
3074 1829 2921
2500 2499 2497
2756 2753 2755
2948 2928 2914
3076 2921 3060
2568 2566 2032
2504 2379 2400
2570 2567 2033
2506 2378 2401
3564 3561 3563
3068 3066 2921
2770 2769 2767
2582 2471 2567
3070 3067 2920
2584 2567 2032
2492 2491 2489
3550 3547 3549
3552 3551 3732
3072 3069 3071
3554 3550 3733
3584 3559 3564
2502 2495 2500
2758 2750 2757
2760 2751 2756
3080 3077 3079
2572 2569 2571
2508 2507 2505
2772 2765 2770
2586 2583 2585
3328 3073 3209
2594 2586 2581
3330 3072 3208
3556 3553 3555
2790 2773 2671
3082 3075 3080
3340 3073 3208
2510 2502 2509
2574 2471 2572
2576 2470 2573
2512 2503 2508
3342 3327 3073
2936 2761 2759
2946 2949 2937
2596 2493 2594
3748 3557 3746
2598 2492 2595
3332 3329 3331
3750 3556 3747
3084 1865 3082
3100 1865 3083
3086 1864 3083
3768 3557 3747
2514 2511 2513
2578 2577 2575
3346 3343 3345
2616 2493 2595
2938 2948 2936
2940 2949 2937
2786 2579 2671
2788 2579 2773
2950 2947 2948
3334 3333 3326
2632 2515 2616
2600 2597 2599
3752 3749 3751
3336 3332 3327
3088 3087 3085
3348 3346 3341
2774 2579 2670
2776 2578 2671
2618 2515 2617
2620 2514 2616
2942 2939 2941
3098 3083 2943
3366 3349 3216
2792 2791 2789
2796 2601 2670
2798 2600 2671
3090 2943 3088
3924 3753 832
3092 2942 3089
3926 3752 833
3572 3337 3335
2808 2601 2671
3096 1865 2943
2778 2775 2777
2622 2619 2621
2970 2950 2936
2818 2623 2670
2820 2622 2671
3582 3585 3573
3576 3585 3573
2794 2792 2787
2800 2799 2797
3094 3091 3093
3102 3101 3099
3928 3927 3925
3574 3584 3572
2780 2779 2772
2782 2778 2773
3104 3102 3097
3776 3575 3577
3586 3583 3584
3362 3095 3216
3364 3349 3095
2822 2819 2821
2958 2781 2783
2802 2801 2794
2804 2800 2795
3350 3095 3217
3352 3094 3216
2810 2601 2795
2812 2795 2671
3106 1879 3104
3778 3768 3777
2980 2803 2805
3108 1878 3105
3368 3365 3367
3770 3768 3777
2960 2970 2958
2962 2971 2959
3122 1879 3105
3606 3586 3572
2968 2971 2959
3354 3351 3353
3772 3769 3776
2814 2811 2813
2816 2814 2809
3110 3109 3107
3782 3779 3769
3370 3368 3363
3774 3771 3773
2964 2963 2961
3356 3348 3355
2972 2969 2970
3358 3349 3354
3932 854 3774
2824 2823 2816
3112 3110 2965
2826 2822 2817
3114 3111 2964
3784 3782 3776
3594 3359 3357
3118 2965 1879
2992 2972 2958
3120 2965 3105
3930 855 3775
3388 3371 3224
2982 2992 2980
2984 2993 2981
3786 3784 1989
3116 3113 3115
3788 3785 1988
2990 2993 2981
3596 3606 3594
3598 3607 3595
3124 3121 3123
3604 3607 3595
3802 3785 1989
3002 2827 2825
3934 3933 3931
3608 3605 3606
2986 2985 2983
3372 3117 3225
3374 3116 3224
3790 3789 3787
3600 3597 3599
2994 2991 2992
3126 3124 3119
3384 3117 3224
3386 3117 3371
3140 2987 1901
3142 2987 3127
3144 3127 1901
3628 3608 3594
3800 3785 3601
3376 3375 3373
3792 3790 3601
3794 3791 3600
2996 2994 2980
3798 3601 1989
3128 3126 1901
3130 3127 1900
3390 3389 3387
3804 3803 3801
3146 3145 3143
3132 3129 3131
3378 3377 3370
3410 3390 3385
3380 3376 3371
3796 3793 3795
3004 2996 3002
3006 2997 3003
3008 3007 3005
3136 2986 3133
3616 3379 3381
3936 877 3797
3938 876 3796
3806 3804 3799
3154 3146 3141
3134 2987 3132
3808 3806 3201
3170 3009 3155
3618 3628 3616
3940 3939 3937
3620 3629 3617
3398 3137 3135
3810 3807 3200
3626 3629 3617
3824 3807 3201
3156 3009 3154
3158 3008 3155
3812 3811 3809
3622 3619 3621
3400 3410 3398
3402 3411 3399
3630 3627 3628
3408 3411 3399
3420 3157 3159
3650 3630 3616
3814 3812 3623
3816 3813 3622
3820 3623 3201
3822 3807 3623
3412 3409 3410
3638 3403 3401
3648 3651 3639
3818 3817 3815
3826 3825 3823
3414 3412 3398
3640 3650 3638
3642 3651 3639
3424 3415 3421
3652 3649 3650
3942 899 3819
3944 898 3818
3828 3826 3821
3644 3641 3643
3422 3414 3420
3842 3645 3209
3844 3645 3829
3654 3652 3638
3846 3829 3209
3946 3945 3943
3660 3425 3423
3830 3828 3209
3832 3829 3208
3848 3845 3847
3834 3831 3833
3664 3655 3661
3662 3654 3660
3850 3848 3843
3836 3834 3645
3666 3665 3663
3838 3835 3644
3840 3837 3839
3852 3850 3217
3854 3851 3216
3864 3667 3217
3866 3851 3667
3868 3851 3217
3856 3853 3855
3948 921 3841
3950 920 3840
3870 3867 3869
3952 3951 3949
3872 3870 3865
3858 3856 3667
3860 3857 3666
3880 3873 3225
3874 3872 3225
3876 3873 3224
3862 3859 3861
3954 943 3863
3956 942 3862
3878 3875 3877
3960 953 3879
3962 952 3878
3958 3957 3955
3964 3963 3961
"""
        write_aag_to_out_dir(aag_source, "test_find_problem_with_miter")
        aag = AAG(from_file=CONFIG.path_to_out().to_file("test_find_problem_with_miter.aag"))
        aag = Sweep(aag).sweep()
        Reducer(aag=aag, constraints=set()).reduce(conflict_limit=1000)
        aag = Sweep(aag).sweep()
        print(aag.get_data().source())
        aag_lec_cnf = aag.to_cnf_with_constraints()
        result = solve_cnf_with_kissat(
            aag_lec_cnf.get_data().source(supplements=([], [aag_lec_cnf.get_data().get_outputs()]))
        )
        answer, _, _ = analyze_result(result)
        self.assertEqual(answer, "UNSAT")

    def test_reduce_dummy(self):
        aag_dummy = AAG(
            from_gates_and_inputs=([], [], [])
        )
        Reducer(aag=aag_dummy, constraints=set()).reduce(conflict_limit=1000)
        aag = Sweep(aag_dummy).sweep()
        print(aag.get_data().source())
