from .config import CONFIG
import subprocess
import json
from .file_ext import FileExtension
from .work_path import WorkPath


def _get_filename_without_ext(path: str):
    return (path.split("/")[-1]).split(".")[0]


def call_balance_checker(filename: str, limit: int):
    subprocess.call([
        "./scripts/LEC_balance_checker_till_xor.py",
        "-cn",
        CONFIG.path_to_out().to_file(f'{filename}{FileExtension.CNF}'),
        "-lim",
        str(limit)
    ])


def read_metadata(filename: str):
    new_filename = CONFIG.path_to_out().to_file('metadata_' + filename + '.json')
    with open(new_filename, "r") as fp:
        return json.load(fp)


def _call_abc(args):
    path_to_abc = CONFIG.path_to_abc()
    subprocess.call([
        path_to_abc.to_file('abc'), "-c", "source " + path_to_abc.to_file('abc.rc') + "; " + args
    ])


def _call_biere_script(script_name: str, *args):
    subprocess.call([
        CONFIG.path_to_aiger().to_file(script_name),
        *args
    ])


def _call_biere_conversion_script(script_name: str, new_filename_without_ext: str, ext_from: FileExtension, ext_to: FileExtension):
    path_to_out = CONFIG.path_to_out()
    _call_biere_script(
        script_name,
        path_to_out.to_file(f'{new_filename_without_ext}{ext_from}'),
        path_to_out.to_file(f'{new_filename_without_ext}{ext_to}')
    )


def call_aiger_to_aiger(new_filename_without_ext: str, ext_from: FileExtension, ext_to: FileExtension):
    _call_biere_conversion_script('aigtoaig', new_filename_without_ext, ext_from, ext_to)


def call_aig_to_aag(new_filename_without_ext: str):
    call_aiger_to_aiger(new_filename_without_ext, FileExtension.AIG, FileExtension.AAG)


def call_aag_to_aig(new_filename_without_ext: str):
    call_aiger_to_aiger(new_filename_without_ext, FileExtension.AAG, FileExtension.AIG)


def call_aiger_to_dot(new_filename_without_ext: str, ext_from: FileExtension):
    _call_biere_conversion_script('aigtodot', new_filename_without_ext, ext_from, FileExtension.DOT)


def call_aig_to_dot(new_filename_without_ext: str):
    call_aiger_to_dot(new_filename_without_ext, FileExtension.AIG)


def call_aag_to_dot(new_filename_without_ext: str):
    call_aiger_to_dot(new_filename_without_ext, FileExtension.AAG)


def call_biere_aiger_to_cnf(new_filename_without_ext: str, ext_from: FileExtension):
    _call_biere_conversion_script('aigtocnf', new_filename_without_ext, ext_from, FileExtension.CNF)


def call_biere_aig_to_cnf(new_filename_without_ext: str):
    call_biere_aiger_to_cnf(new_filename_without_ext, FileExtension.AIG)


def call_biere_aag_to_cnf(new_filename_without_ext: str):
    call_biere_aiger_to_cnf(new_filename_without_ext, FileExtension.AAG)


def call_handmade_aag_to_cnf(new_filename_without_ext: str, input_path: WorkPath):
    subprocess.call([
        CONFIG.path_to_scripts().to_file('circuit2cnf.py'),
        "-n",
        input_path.to_file(f'{new_filename_without_ext}{FileExtension.AAG}'),
        "-o",
        CONFIG.path_to_out().to_file(f'{new_filename_without_ext}{FileExtension.CNF}')
    ])


def call_handmade_aag_to_cnf_from_src(new_filename_without_ext: str):
    call_handmade_aag_to_cnf(new_filename_without_ext, CONFIG.path_to_schemes())


def call_handmade_aag_to_cnf_from_out(new_filename_without_ext: str):
    call_handmade_aag_to_cnf(new_filename_without_ext, CONFIG.path_to_out())


def call_fraig_and_rewrite(filename_before: str, filename_after: str):
    path_to_out = CONFIG.path_to_out()
    filepath = path_to_out.to_file(f'{filename_before}{FileExtension.AIG}')
    fraig_filepath = path_to_out.to_file(f'{filename_after}{FileExtension.AIG}')

    _call_abc('r ' + filepath + '; ps; fraig; b; rw -l; rw -lz; b; rw -lz; b; write ' + fraig_filepath)

    call_aig_to_aag(filename_after)


def call_biere_miter(filepath_first: str, filepath_second: str, miter_filepath: str):
    _call_biere_script('aigmiter', '-o', miter_filepath, filepath_first, filepath_second)


__all__ = [
    'call_balance_checker',
    'read_metadata',
    'call_aiger_to_aiger',
    'call_aig_to_aag',
    'call_aag_to_aig',
    'call_aiger_to_dot',
    'call_aig_to_dot',
    'call_aag_to_dot',
    'call_biere_aiger_to_cnf',
    'call_biere_aig_to_cnf',
    'call_biere_aag_to_cnf',
    'call_handmade_aag_to_cnf',
    'call_handmade_aag_to_cnf_from_src',
    'call_handmade_aag_to_cnf_from_out',
    'call_fraig_and_rewrite',
    'call_biere_miter'
]
