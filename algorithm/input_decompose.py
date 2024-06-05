import random
import copy
import itertools
import subprocess
import math
from itertools import combinations, product
from statistics import mean, median, variance
from util import MultithreadingLog, SatParser, solve_cnf_source
import time


def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)


def round_up(number): return int(number) + (number % 1 > 0)


def getCNF(cnf_file_lines):
    clauses = []
    header = None
    comments = []
    inputs = []
    vars_left = []
    outputs_first = []
    vars_right = []
    outputs_second = []
    miter_vars = []
    cubes_vars = []
    outputs = []
    gates_vars = []
    var_set = []
    dimacs = cnf_file_lines[:-1]
    for i in range(len(dimacs)):
        if dimacs[i][0] == 'p':
            header = dimacs[i][:-1] if dimacs[i][-1] == '\n' else dimacs[i]
        elif dimacs[i][0] == 'c':
            comments.append(dimacs[i][:-1] if dimacs[i][-1] == '\n' else dimacs[i])
            if 'c inputs: ' in dimacs[i]:
                inputs = list(map(int, dimacs[i].split(':')[1].split()))
            elif 'c variables for gates in first scheme' in dimacs[i]:
                vars_right = [x for x in range(int(dimacs[i].split()[-3]), int(dimacs[i].split()[-1]) + 1)]
            elif 'c outputs first scheme' in dimacs[i]:
                outputs_first = list(map(int, dimacs[i].split(':')[1].split()))
            elif 'c variables for gates in second scheme' in dimacs[i]:
                vars_left = [x for x in range(int(dimacs[i].split()[-3]), int(dimacs[i].split()[-1]) + 1)]
            elif 'c outputs second scheme' in dimacs[i]:
                outputs_second = list(map(int, dimacs[i].split(':')[1].split()))
            elif 'c miter variables' in dimacs[i]:
                miter_vars = list(map(int, dimacs[i].split(':')[1].split()))
            elif 'c cubes variables:' in dimacs[i]:
                cubes_vars = list(map(int, dimacs[i].split(':')[1].split()))
            elif 'c outputs: ' in dimacs[i]:
                outputs = list(map(int, dimacs[i].split(':')[1].split()))
            elif 'c variables for gates:' in dimacs[i]:
                gates_vars = [x for x in range(int(dimacs[i].split()[-3]), int(dimacs[i].split()[-1]) + 1)]
            elif 'c var_set:' in dimacs[i]:
                var_set = list(map(int, dimacs[i].split(':')[1].split()))
        else:
            if len(dimacs[i]) > 1:
                clauses.append(list(map(int, dimacs[i].split()[:-1])))
    return header, comments, inputs, outputs, gates_vars, vars_left, outputs_first, vars_right, outputs_second, miter_vars, cubes_vars, var_set, clauses


def make_pairs(*lists):
    for t in combinations(lists, 2):
        for pair in product(*t):
            yield pair


def create_and_solve_CNF(task_index, clauses, new_clauses, max_var, solver):
    cnf_str = create_str_CNF(clauses, new_clauses, max_var)
    return solve_cnf_source(solver, cnf_str)


def create_str_CNF(clauses, new_clauses, max_var):
    lines = []
    header_ = 'p cnf ' + str(max_var) + ' ' + str(len(clauses) + len(new_clauses))
    lines.append(header_)
    lines.extend([' '.join(list(map(str, clause))) + ' 0' for clause in clauses])
    lines.extend([' '.join(list(map(str, clause))) + ' 0' for clause in new_clauses])
    cnf = '\n'.join(lines)
    return cnf


def solve_CNF(cnf, solver):
    solver = subprocess.run([solver], capture_output=True, text=True, input=cnf)
    result = solver.stdout.split('\n')
    errors = solver.stderr
    if len(errors) > 0:
        print(errors)
    return result


def solve_CNF_timelimit(cnf, solver, timelim):
    params = [solver]
    if 'cadical' in solver:
        params.append('-t')
        params.append(str(timelim))
    elif 'kissat' in solver:
        params.append(f'--time={timelim}')
    else:
        params.append(f'-cpu-time={timelim}')
    solver = subprocess.run(params, capture_output=True, text=True, input=cnf)
    result = solver.stdout.split('\n')
    errors = solver.stderr
    if len(errors) > 0:
        print(errors)
    return result


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield tuple(lst[i:i + n])


def solve_vector(task_index, binaryvector, input_decompose, clauses, max_var, solver, force_xor_flag, force_disj_flag):
    if len(binaryvector) == len(input_decompose):
        new_clauses = []
        current_var = max_var
        for bit, inp_chunk in zip(binaryvector, input_decompose):
            if len(inp_chunk) == 2:
                if force_disj_flag == False:
                    new_clauses_, current_var = encode_XOR_clauses(inp_chunk, current_var)
                else:
                    new_clauses_, current_var = encode_DISJUNCTION_clauses(inp_chunk, current_var)
            elif len(inp_chunk) == 3:
                if force_xor_flag == False:
                    new_clauses_, current_var = encode_MAJORITY_clauses(inp_chunk, current_var)
                else:
                    new_clauses_, current_var = encode_TRIPLEXOR_clauses(inp_chunk, current_var)
            elif len(inp_chunk) == 4:
                if force_xor_flag == False:
                    new_clauses_, current_var = encode_BENT_clauses(inp_chunk, current_var)
                else:
                    new_clauses_, current_var = encode_QUADROXOR_clauses(inp_chunk, current_var)
            elif len(inp_chunk) == 5:
                new_clauses_, current_var = encode_PENTAXOR_clauses(inp_chunk, current_var)
            elif len(inp_chunk) == 8:
                new_clauses_, current_var = encode_clauses_by_vector_function_for_8_vars(inp_chunk, current_var)
            elif len(inp_chunk) > 1:
                new_clauses_, current_var = encode_XOR_Tseitin(inp_chunk, current_var)
            new_clauses.extend(new_clauses_)
            new_clauses.append([current_var if bit == 1 else -current_var])
        return create_and_solve_CNF(task_index, clauses, new_clauses, current_var, solver)
    else:
        raise 'len(binaryvector) != len(input_decompose)'


def encode_XOR_Tseitin(inp_chunk, current_var):
    new_clauses = []
    current_chunk = [inp_chunk[0], inp_chunk[1]]
    new_clauses_, current_var = encode_XOR_clauses(current_chunk, current_var)
    new_clauses.extend(new_clauses_)
    index = 2
    while index < len(inp_chunk):
        current_chunk = [current_var, inp_chunk[index]]
        new_clauses_, current_var = encode_XOR_clauses(current_chunk, current_var)
        new_clauses.extend(new_clauses_)
        index += 1
    return new_clauses, current_var


def encode_DISJUNCTION_clauses(inp_pair, current_var):
    current_var += 1
    clauses = [[-current_var, inp_pair[0], inp_pair[1]],
               [current_var, inp_pair[0], -inp_pair[1]],
               [-current_var, inp_pair[0], inp_pair[1]],
               [current_var, -inp_pair[0], -inp_pair[1]]]
    return clauses, current_var


def encode_XOR_clauses(inp_pair, current_var):
    current_var += 1
    clauses = [[current_var, -inp_pair[0], inp_pair[1]],
               [current_var, inp_pair[0], -inp_pair[1]],
               [-current_var, inp_pair[0], inp_pair[1]],
               [-current_var, -inp_pair[0], -inp_pair[1]]]
    return clauses, current_var


def encode_MAJORITY_clauses(inp_triple, current_var):
    current_var += 1
    clauses = [[-current_var, inp_triple[0], inp_triple[1], inp_triple[2]],
               [-current_var, inp_triple[0], inp_triple[1], -inp_triple[2]],
               [-current_var, inp_triple[0], -inp_triple[1], inp_triple[2]],
               [current_var, inp_triple[0], -inp_triple[1], -inp_triple[2]],
               [-current_var, -inp_triple[0], inp_triple[1], inp_triple[2]],
               [current_var, -inp_triple[0], inp_triple[1], -inp_triple[2]],
               [current_var, -inp_triple[0], -inp_triple[1], inp_triple[2]],
               [current_var, -inp_triple[0], -inp_triple[1], -inp_triple[2]]]
    return clauses, current_var


def encode_BENT_clauses(inp_cuadro, current_var):
    current_var += 1
    clauses = [[-current_var, inp_cuadro[0], inp_cuadro[1], inp_cuadro[2], inp_cuadro[3]],
               [-current_var, inp_cuadro[0], inp_cuadro[1], inp_cuadro[2], -inp_cuadro[3]],
               [-current_var, inp_cuadro[0], inp_cuadro[1], -inp_cuadro[2], inp_cuadro[3]],
               [-current_var, inp_cuadro[0], inp_cuadro[1], -inp_cuadro[2], -inp_cuadro[3]],
               [-current_var, inp_cuadro[0], -inp_cuadro[1], inp_cuadro[2], inp_cuadro[3]],
               [current_var, inp_cuadro[0], -inp_cuadro[1], inp_cuadro[2], -inp_cuadro[3]],
               [-current_var, inp_cuadro[0], -inp_cuadro[1], -inp_cuadro[2], inp_cuadro[3]],
               [current_var, inp_cuadro[0], -inp_cuadro[1], -inp_cuadro[2], -inp_cuadro[3]],
               [-current_var, -inp_cuadro[0], inp_cuadro[1], inp_cuadro[2], inp_cuadro[3]],
               [-current_var, -inp_cuadro[0], inp_cuadro[1], inp_cuadro[2], -inp_cuadro[3]],
               [current_var, -inp_cuadro[0], inp_cuadro[1], -inp_cuadro[2], inp_cuadro[3]],
               [current_var, -inp_cuadro[0], inp_cuadro[1], -inp_cuadro[2], -inp_cuadro[3]],
               [-current_var, -inp_cuadro[0], -inp_cuadro[1], inp_cuadro[2], inp_cuadro[3]],
               [current_var, -inp_cuadro[0], -inp_cuadro[1], inp_cuadro[2], -inp_cuadro[3]],
               [current_var, -inp_cuadro[0], -inp_cuadro[1], -inp_cuadro[2], inp_cuadro[3]],
               [-current_var, -inp_cuadro[0], -inp_cuadro[1], -inp_cuadro[2], -inp_cuadro[3]], ]
    return clauses, current_var


def encode_TRIPLEXOR_clauses(inp_chunk, current_var):
    current_var += 1
    clauses = [[-current_var, inp_chunk[0], inp_chunk[1], inp_chunk[2]],
               [current_var, inp_chunk[0], inp_chunk[1], -inp_chunk[2]],
               [current_var, inp_chunk[0], -inp_chunk[1], inp_chunk[2]],
               [-current_var, inp_chunk[0], -inp_chunk[1], -inp_chunk[2]],
               [current_var, -inp_chunk[0], inp_chunk[1], inp_chunk[2]],
               [-current_var, -inp_chunk[0], inp_chunk[1], -inp_chunk[2]],
               [-current_var, -inp_chunk[0], -inp_chunk[1], inp_chunk[2]],
               [current_var, -inp_chunk[0], -inp_chunk[1], -inp_chunk[2]]
               ]
    return clauses, current_var


def encode_QUADROXOR_clauses(inp_cuadro, current_var):
    current_var += 1
    clauses = [[-current_var, inp_cuadro[0], inp_cuadro[1], inp_cuadro[2], inp_cuadro[3]],
               [current_var, inp_cuadro[0], inp_cuadro[1], inp_cuadro[2], -inp_cuadro[3]],
               [current_var, inp_cuadro[0], inp_cuadro[1], -inp_cuadro[2], inp_cuadro[3]],
               [-current_var, inp_cuadro[0], inp_cuadro[1], -inp_cuadro[2], -inp_cuadro[3]],
               [current_var, inp_cuadro[0], -inp_cuadro[1], inp_cuadro[2], inp_cuadro[3]],
               [-current_var, inp_cuadro[0], -inp_cuadro[1], inp_cuadro[2], -inp_cuadro[3]],
               [-current_var, inp_cuadro[0], -inp_cuadro[1], -inp_cuadro[2], inp_cuadro[3]],
               [current_var, inp_cuadro[0], -inp_cuadro[1], -inp_cuadro[2], -inp_cuadro[3]],
               [current_var, -inp_cuadro[0], inp_cuadro[1], inp_cuadro[2], inp_cuadro[3]],
               [-current_var, -inp_cuadro[0], inp_cuadro[1], inp_cuadro[2], -inp_cuadro[3]],
               [-current_var, -inp_cuadro[0], inp_cuadro[1], -inp_cuadro[2], inp_cuadro[3]],
               [current_var, -inp_cuadro[0], inp_cuadro[1], -inp_cuadro[2], -inp_cuadro[3]],
               [-current_var, -inp_cuadro[0], -inp_cuadro[1], inp_cuadro[2], inp_cuadro[3]],
               [current_var, -inp_cuadro[0], -inp_cuadro[1], inp_cuadro[2], -inp_cuadro[3]],
               [current_var, -inp_cuadro[0], -inp_cuadro[1], -inp_cuadro[2], inp_cuadro[3]],
               [-current_var, -inp_cuadro[0], -inp_cuadro[1], -inp_cuadro[2], -inp_cuadro[3]], ]
    return clauses, current_var


def encode_PENTAXOR_clauses(inp_penta, current_var):
    current_var += 1
    clauses = [[-current_var, inp_penta[0], inp_penta[1], inp_penta[2], inp_penta[3], inp_penta[4]],
               [current_var, inp_penta[0], inp_penta[1], inp_penta[2], inp_penta[3], -inp_penta[4]],
               [current_var, inp_penta[0], inp_penta[1], inp_penta[2], -inp_penta[3], inp_penta[4]],
               [-current_var, inp_penta[0], inp_penta[1], inp_penta[2], -inp_penta[3], -inp_penta[4]],
               [current_var, inp_penta[0], inp_penta[1], -inp_penta[2], inp_penta[3], inp_penta[4]],
               [-current_var, inp_penta[0], inp_penta[1], -inp_penta[2], inp_penta[3], -inp_penta[4]],
               [-current_var, inp_penta[0], inp_penta[1], -inp_penta[2], -inp_penta[3], inp_penta[4]],
               [current_var, inp_penta[0], inp_penta[1], -inp_penta[2], -inp_penta[3], -inp_penta[4]],
               [current_var, inp_penta[0], -inp_penta[1], inp_penta[2], inp_penta[3], inp_penta[4]],
               [-current_var, inp_penta[0], -inp_penta[1], inp_penta[2], inp_penta[3], -inp_penta[4]],
               [-current_var, inp_penta[0], -inp_penta[1], inp_penta[2], -inp_penta[3], inp_penta[4]],
               [current_var, inp_penta[0], -inp_penta[1], inp_penta[2], -inp_penta[3], -inp_penta[4]],
               [-current_var, inp_penta[0], -inp_penta[1], -inp_penta[2], inp_penta[3], inp_penta[4]],
               [current_var, inp_penta[0], -inp_penta[1], -inp_penta[2], inp_penta[3], -inp_penta[4]],
               [current_var, inp_penta[0], -inp_penta[1], -inp_penta[2], -inp_penta[3], inp_penta[4]],
               [-current_var, inp_penta[0], -inp_penta[1], -inp_penta[2], -inp_penta[3], -inp_penta[4]],
               [current_var, -inp_penta[0], inp_penta[1], inp_penta[2], inp_penta[3], inp_penta[4]],
               [-current_var, -inp_penta[0], inp_penta[1], inp_penta[2], inp_penta[3], -inp_penta[4]],
               [-current_var, -inp_penta[0], inp_penta[1], inp_penta[2], -inp_penta[3], inp_penta[4]],
               [current_var, -inp_penta[0], inp_penta[1], inp_penta[2], -inp_penta[3], -inp_penta[4]],
               [-current_var, -inp_penta[0], inp_penta[1], -inp_penta[2], inp_penta[3], inp_penta[4]],
               [current_var, -inp_penta[0], inp_penta[1], -inp_penta[2], inp_penta[3], -inp_penta[4]],
               [current_var, -inp_penta[0], inp_penta[1], -inp_penta[2], -inp_penta[3], inp_penta[4]],
               [-current_var, -inp_penta[0], inp_penta[1], -inp_penta[2], -inp_penta[3], -inp_penta[4]],
               [-current_var, -inp_penta[0], -inp_penta[1], inp_penta[2], inp_penta[3], inp_penta[4]],
               [current_var, -inp_penta[0], -inp_penta[1], inp_penta[2], inp_penta[3], -inp_penta[4]],
               [current_var, -inp_penta[0], -inp_penta[1], inp_penta[2], -inp_penta[3], inp_penta[4]],
               [-current_var, -inp_penta[0], -inp_penta[1], inp_penta[2], -inp_penta[3], -inp_penta[4]],
               [current_var, -inp_penta[0], -inp_penta[1], -inp_penta[2], inp_penta[3], inp_penta[4]],
               [-current_var, -inp_penta[0], -inp_penta[1], -inp_penta[2], inp_penta[3], -inp_penta[4]],
               [-current_var, -inp_penta[0], -inp_penta[1], -inp_penta[2], -inp_penta[3], inp_penta[4]],
               [current_var, -inp_penta[0], -inp_penta[1], -inp_penta[2], -inp_penta[3], -inp_penta[4]]]
    return clauses, current_var


def encode_clauses_by_vector_function_for_8_vars(inp_chunk, current_var):
    current_var += 1
    vector_func = [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1,
                   1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0,
                   1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1,
                   1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1,
                   1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1,
                   1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0,
                   1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1,
                   1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0,
                   1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1,
                   1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0,
                   1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1,
                   1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1,
                   1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1,
                   1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0,
                   1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1,
                   1, 0]
    truth_table_lines = list(itertools.product([0, 1], repeat=len(inp_chunk) + 1))
    truth_table_lines_with_vec = zip(truth_table_lines, vector_func)
    truth_table_zero_lines = [(x) for x, y in truth_table_lines_with_vec if y == 0]
    clauses = []
    for line in truth_table_zero_lines:
        if len(line) != len(inp_chunk) + 1:
            raise Exception('len(line) != len(inp_chunk) + 1')
        current_var_ = current_var if line[-1] == 0 else -current_var
        clause = [current_var_]
        for i in range(len(inp_chunk)):
            inp_ = inp_chunk[i] if line[i] == 0 else -inp_chunk[i]
            clause.append(inp_)
        clauses.append(clause)
    return clauses, current_var


def random_binary_vector(n):
    return tuple([random.randint(0, 1) for i in range(n)])


def make_template_clauses(clauses, miter_vars, outputs):
    template_clauses = []
    removed_clauses = []
    for clause in clauses:
        if (len(miter_vars) > 0) and (len([lit_ for lit_ in clause if abs(lit_) in miter_vars]) < len(miter_vars)):
            template_clauses.append(clause)
        elif (len(miter_vars) == 0) and (len(clause) > 1) and (
                len([lit_ for lit_ in clause if abs(lit_) in outputs]) != len(clause)):
            template_clauses.append(clause)
        else:
            removed_clauses.append(clause)
    return template_clauses, removed_clauses


def cnf_input_decompose(
        shuffle_inputs,
        cnf_file_source: str,
        length_decompose,
        limit,
        solver,
        logger: MultithreadingLog
):
    cnf_file_lines = cnf_file_source.split('\n')
    start_time = time.time()
    nof_additional_chunks = 0
    force_xor_flag = False
    force_disj_flag = False
    if '_' in length_decompose:
        len_inp_chunks = [int(length_decompose.split('_')[0])]
        nof_additional_chunks = int(length_decompose.split('_')[1])
    elif '+' in length_decompose:
        len_inp_chunks = list(map(int, length_decompose.split('+')))
    elif 'x' in length_decompose:
        len_inp_chunks = [int(length_decompose.split('x')[0])]
        force_xor_flag = True
    elif 'd' in length_decompose:
        len_inp_chunks = [int(length_decompose.split('d')[0])]
        force_disj_flag = True
    else:
        len_inp_chunks = [int(length_decompose)]
    tasklimit = limit
    (
        header, comments, inputs, outputs, gates_vars, vars_left, outputs_first, vars_right, outputs_second, miter_vars,
        current_buckets, var_set, clauses
    ) = getCNF(
        cnf_file_lines
    )
    max_var = int(header.split()[2])

    vars_for_chunks = copy.copy(inputs)
    if shuffle_inputs == True:
        random.shuffle(vars_for_chunks)

    input_decompose = []
    for i_ in len_inp_chunks:
        input_decompose_ = list(chunks(vars_for_chunks, i_))
        input_decompose_[-1] = vars_for_chunks[-i_:]
        input_decompose += input_decompose_

    i = 0
    while i < nof_additional_chunks:
        new_chunk = tuple(sorted(random.sample(vars_for_chunks, len_inp_chunks[0])))
        if new_chunk not in input_decompose:
            input_decompose.append(new_chunk)
            i += 1

    full_solve_flag = False
    total_tasks = pow(2, len(input_decompose))
    if tasklimit >= total_tasks:
        tasklimit = total_tasks
        full_solve_flag = True
        binary_vectors = list(itertools.product([0, 1], repeat=len(input_decompose)))

    sats = []
    unsats = []
    results_table = []
    if full_solve_flag == True:
        logger.emit({
            'Full solving. Number of tasks:': tasklimit
        })
    else:
        logger.emit({
            'Sample size:': tasklimit
        })
    task_index = 0

    satisfying_assignment_cnf_dict_result = None
    answer_result = 'UNSAT'

    while task_index < tasklimit:
        if full_solve_flag == True:
            binary_vector = binary_vectors[task_index]
        else:
            binary_vector = random_binary_vector(len(input_decompose))
        result = solve_vector(
            task_index, binary_vector, input_decompose, clauses, max_var, solver, force_xor_flag,
            force_disj_flag
        )
        satisfying_assignment = SatParser(result)

        cpu_time, conflicts, satisfying_assignment_cnf_dict, answer = satisfying_assignment.parse_complete()
        data = (answer, cpu_time, conflicts)
        results_table.append(data)
        if 'UNSAT' == answer:
            unsats.append(cpu_time)
        elif 'SAT' == answer:
            answer_result = 'SAT'
            satisfying_assignment_cnf_dict_result = satisfying_assignment_cnf_dict
            sats.append(cpu_time)
        current_avg_solvetime = round(sum([x[1] for x in results_table]) / len(results_table), 2)
        current_avg_conflicts = round(sum([x[2] for x in results_table]) / len(results_table), 2)
        if len(results_table) >= 2:
            denom = 0.01 * sum([x[1] for x in results_table]) * sum([x[1] for x in results_table]) / len(results_table)
            if denom == 0:
                task_index += 1
                continue
            pr = variance([x[1] for x in results_table]) / denom
            logger.emit({
                'Sample size': tasklimit,
                'Current time estimate': current_avg_solvetime * total_tasks,
                'Current ratio': (current_avg_conflicts * total_tasks) / pow(2, len(inputs)),
                'Current pr': pr
            })
            if pr < 0.05:
                break
        task_index += 1

    logger.emit({
        'Comment': 'master finishing',
        'Total runtime': time.time() - start_time
    })
    res_time_ = [x[1] for x in results_table]
    avg_solvetime = round(sum(res_time_) / len(results_table), 2)
    res_conflicts_ = [x[2] for x in results_table]
    avg_conflicts = round(sum(res_conflicts_) / len(results_table), 2)
    if full_solve_flag == True:
        logger.emit({
            'Comment': 'All possible tasks solved.',
            'Total runtime': time.time() - start_time
        })
    logger.emit({
        'Decomposition type': length_decompose,
        'Solver': solver.name,
        'Solved tasks': len(results_table),
        'Total tasks': total_tasks,
        'Average solvetime': avg_solvetime,
        'Median time': round(median(res_time_), 2),
        'Min solvetime': round(min(res_time_), 2),
        'Max solvetime': round(max(res_time_), 2),
        'Variance of time': round(variance(res_time_), 2)
    })
    if len(results_table) == total_tasks:
        cpu_time_result = sum(res_time_)
        logger.emit({
            'Sigma': round(math.sqrt(variance(res_time_)), 2),
            'Real time for solving all tasks is': cpu_time_result
        })
    else:
        cpu_time_result = sum([x[1] for x in results_table]) * total_tasks / len(results_table)
        logger.emit({
            'Sd': round(math.sqrt(variance(res_time_)), 2),
            'Estimate time for solving all tasks is': cpu_time_result
        })
    denom = (0.01 * len(results_table) * avg_solvetime * avg_solvetime)
    logger.emit({
        'Average number of conflicts:': avg_conflicts,
        'Median number of conflicts:': round(median(res_conflicts_), 2),
        'Min number of conflicts:': min(res_conflicts_),
        'Max number of conflicts:': max(res_conflicts_),
        'Variance of number of conflicts:': round(variance(res_conflicts_), 2),
        'Pr:': 1 - (variance([x[1] for x in results_table]) / denom) if denom != 0 else 'div by zero'
    })
    if len(results_table) == total_tasks:
        conflicts_result = sum(res_conflicts_)
        logger.emit({
            'Real total number of conflicts for solving all tasks is': conflicts_result,
            '(Number of conflicts / Brutforce actions) ratio:': round(sum(res_conflicts_) / pow(2, len(inputs)), 10)
        })
    else:
        conflicts_result = sum([x[2] for x in results_table]) * total_tasks / len(results_table)
        logger.emit({
            'Estimate number of conflicts for solving all tasks is ': conflicts_result,
            '(Number of conflicts / Brutforce actions) ratio:': round((avg_conflicts * total_tasks) / pow(2, len(inputs)), 10)
        })
    logger.emit({
        'Number of SATs:': len(sats)
    })
    if len(sats) > 0:
        logger.emit({
            'SATs total runtime:': sum(sats),
            'SATs average runtime:': mean(sats),
            'SATs median runtime:': median(sats),
            'SATs variance of runtime:': variance(sats)
        })
    logger.emit({
        'Number of UNSATs:': len(unsats)
    })
    if len(unsats) > 0:
        logger.emit({
            'UNSATs total runtime:': sum(unsats),
            'UNSATs average runtime:': mean(unsats),
            'UNSATs median runtime:': median(unsats),
            'UNSATs variance of runtime:': variance(unsats)
        })

    return cpu_time_result, int(conflicts_result), satisfying_assignment_cnf_dict_result, answer_result


__all__ = [
    'cnf_input_decompose'
]
