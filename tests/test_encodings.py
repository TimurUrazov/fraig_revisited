import unittest
from encoding import AAG, AAGData
from util import WorkPath, CONFIG, write_aag_to_out_dir
from copy import copy
from scheme import AAGLEC
from aig_substitutions import make_subst_and_fraig


class TestEncodings(unittest.TestCase):
    def test_aag_from_components(self):
        aag = AAG(from_gates_and_inputs=([2, 4], [7], [[6, 3, 5]]))
        aag_data = aag.get_data()

        self.assertIsInstance(aag_data, AAGData)
        self.assertEqual(
            aag_data.source(),
            'aag 3 2 0 1 1\n2\n4\n7\n6 3 5\n'
        )

    def test_aag_to_cnf(self):
        aag = AAG(from_gates_and_inputs=([2, 4], [7], [[6, 3, 5]]), constraints=[(3, 4)])
        cnf = aag.to_cnf_with_constraints()
        print(cnf.get_data().source())

    def test_aag_from_file(self):
        root_path = WorkPath('tests')
        aag = AAG(from_file=root_path.to_file('dadda5x5.aag', 'encodings'))
        aag_data = aag.get_data()

        self.assertEqual(aag_data.outputs()[1], 285)
        self.assertEqual(aag_data.inputs()[9], 20)
        self.assertEqual(aag_data.and_gates()[4], [30, 10, 12])
        self.assertEqual(aag_data.source(), aag.get_raw_data())

        aag_copy = copy(aag)
        cnf_copy_data = aag_copy.get_data()

        self.assertEqual(aag_data.inputs(), cnf_copy_data.inputs())
        self.assertEqual(aag_data.outputs(), cnf_copy_data.outputs())
        self.assertEqual(aag_data.and_gates(), cnf_copy_data.and_gates())
        self.assertEqual(aag_data.max_literal, cnf_copy_data.max_literal)

    def _get_all_strings_of_given_length(self, n, array, current_index, result):
        if current_index == n:
            result.append(array.copy())
            return

        array[current_index] = 0
        self._get_all_strings_of_given_length(n, array, current_index + 1, result)
        array[current_index] = 1
        self._get_all_strings_of_given_length(n, array, current_index + 1, result)

    @staticmethod
    def _check_outputs_after_fraiging(fraig_filename: str, _):
        fraig = AAG(from_file=CONFIG.path_to_out().to_file(fraig_filename) + '.aag')
        assert fraig.get_data().outputs()[0] == 0

    def test_handmade_miter(self):
        schemes_path = CONFIG.path_to_schemes()

        multiplier_size = 3
        multiplier_type = '{0}x{0}'.format(multiplier_size)

        miter = AAGLEC(
            left_scheme_from_file=schemes_path.to_file("dadda{}.aag".format(multiplier_type)),
            right_scheme_from_file=schemes_path.to_file("karatsuba{}.aag".format(multiplier_type))
        ).get_handmade_miter()

        bit_vectors = []

        vector_len = multiplier_size * 2

        self._get_all_strings_of_given_length(vector_len, [0] * vector_len, 0, bit_vectors)

        inputs = list(map(lambda x: x * 2, range(1, 1 + vector_len)))

        for vector in bit_vectors:
            nodes_and_values_to_subst = dict(zip(inputs, vector))
            make_subst_and_fraig(
                substitution_dict=nodes_and_values_to_subst,
                new_filename_without_extension='test_handmade_miter',
                aag=miter,
                output_checker=self._check_outputs_after_fraiging,
                compatibility_checker=lambda x, y: (True, 0, 0),
                xor_len_before_subst=multiplier_size
            )

    def test_encodings_with_constraints(self):
        aag_source = """c constraints: [[410, 410]]
aag 502 1 0 6 82
410
974
980
986
992
998
1004
386 410 411
360 410 411
362 410 411
568 410 411
974 410 411
566 410 411
408 410 411
652 567 569
364 363 361
672 653 410
676 653 410
740 652 411
684 652 410
686 653 411
758 652 411
760 653 410
688 685 687
746 741 410
834 759 761
678 673 677
836 835 410
680 678 411
784 746 653
690 688 410
692 689 411
854 835 410
856 834 411
794 681 410
858 855 857
842 837 411
694 691 693
776 695 785
880 842 834
976 365 859
786 695 784
978 364 858
788 694 785
800 410 776
808 410 777
810 411 776
980 979 977
796 681 776
862 789 787
864 863 881
802 801 797
812 809 811
882 863 881
884 862 880
804 802 795
870 865 880
814 812 681
816 813 680
886 883 885
908 870 862
818 815 817
982 387 887
918 805 410
984 386 886
912 818 909
986 985 983
900 819 909
910 819 908
932 410 901
934 411 900
914 913 911
920 805 900
924 410 900
936 933 935
988 409 915
990 408 914
926 921 925
992 991 989
938 936 805
940 937 804
928 926 919
1000 929 410
1002 928 411
942 939 941
994 943 411
1004 1003 1001
996 942 410
998 997 995
"""

        write_aag_to_out_dir(aag_source, "test_find_problem_with_miter")
        aag = AAG(from_file=CONFIG.path_to_out().to_file("test_find_problem_with_miter.aag"))

        self.assertEqual(aag.get_data().constraints(), [[410, 410]])
        self.assertEqual(aag.get_data().source(), aag_source)
        self.assertEqual([974, 980, 986, 992, 998, 1004], aag.get_data().outputs())
        self.assertEqual([410], aag.get_data().inputs())
        self.assertEqual(aag.get_graph()[410].constraints, [(410, 410)])
