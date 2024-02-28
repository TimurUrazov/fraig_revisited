from .config import CONFIG
from .file_ext import FileExtension


def _write_to_out_dir(file_lines, base_filename, ext):
    with open(CONFIG.path_to_out().to_file(f'{base_filename}{ext}'), 'w') as new_file:
        new_file.writelines(file_lines)


def write_cnf_to_out_dir(file_lines, base_filename):
    _write_to_out_dir(file_lines, base_filename, FileExtension.CNF)


def write_aag_to_out_dir(file_lines, base_filename):
    _write_to_out_dir(file_lines, base_filename, FileExtension.AAG)


def write_aag(file_lines, base_filename):
    with open(CONFIG.path_to_out().to_file(f'{base_filename}{FileExtension.AAG}'), 'w') as new_file:
        new_file.writelines(file_lines)


def write_cnf(file_lines, base_filename):
    with open(CONFIG.path_to_out().to_file(f'{base_filename}{FileExtension.CNF}'), 'w') as new_file:
        new_file.writelines(file_lines)


def write_to_file(filename, content, mode="a"):
    with open(CONFIG.path_to_out().to_file(filename), mode) as f:
        print(content, file=f)


__all__ = [
    'write_cnf_to_out_dir',
    'write_aag_to_out_dir',
    'write_to_file',
    'write_aag',
    'write_cnf'
]
