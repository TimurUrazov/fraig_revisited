from .substitution import make_and_process_substitution
from util import (call_fraig_and_rewrite, call_aag_to_aig, solve_cnf_with_kissat, write_cnf_to_out_dir,
                  write_aag_to_out_dir, solve_cnf_with_rokk)
from encoding.impl.aag import AAG
from encoding.impl.cnf import CNF


def process_and_solve_cnf(
        cnf: CNF, xor_len: int, fraig_checked_filename: str, constraints=None, limit=None, kissat=True
):
    if constraints is None:
        constraints = []
    outputs = cnf.get_data().get_outputs()
    assumptions = [" ".join(map(str, outputs[:xor_len]))] + [-output for output in outputs[xor_len:]]
    str_cnf = cnf.get_data().source(supplements=(assumptions, constraints))
    write_cnf_to_out_dir([str_cnf], fraig_checked_filename)
    if kissat:
        return solve_cnf_with_kissat(str_cnf, conflicts_limit=limit)
    else:
        return solve_cnf_with_rokk(str_cnf)


def make_subst_and_fraig(
        substitution_dict: dict,
        new_filename_without_extension: str,
        aag: AAG,
        output_checker,
        compatibility_checker,
        xor_len_before_subst: int
):
    aag, solvetime, conflicts = make_and_process_substitution(
        substitution_dict, aag, compatibility_checker
    )

    substitution_filename = f'{new_filename_without_extension}_substitution'

    write_aag_to_out_dir(aag.get_data().source(), substitution_filename)

    call_aag_to_aig(substitution_filename)

    fraig_filename = f'{substitution_filename}_fraig'

    call_fraig_and_rewrite(substitution_filename, fraig_filename)

    return output_checker(fraig_filename, xor_len_before_subst), solvetime, conflicts
