from toposort import toposort
from typing import List, Tuple, Dict


def get_bench_header(bench):
    bench_testname = ''
    nof_inputs = 0
    nof_outputs = 0
    nof_and_gates = 0
    nof_not_gates = 0
    i = 0
    while i < len(bench):
        line = bench[i]
        if len(line) == 0:
            del bench[i]
        elif line[0] == '#':
            if 'testname' in line:
                bench_testname = line.split()[-1]
            elif 'input' in line:
                nof_inputs = int(line.split()[-1])
            elif 'output' in line:
                nof_outputs = int(line.split()[-1])
            elif 'not' in line:
                nof_not_gates = int(line.split()[-1])
            elif 'and' in line:
                nof_and_gates = int(line.split()[-1])
            del bench[i]
        else:
            i += 1
    return bench, bench_testname, nof_inputs, nof_outputs, nof_and_gates, nof_not_gates


def parse_bench_to_map(bench_lines, start_var_id, circuit_number):
    vars_dict = dict()
    outputs_names = []
    inputs_names = []
    current_var_id = start_var_id
    input_var_ids = 1
    for line in bench_lines:
        if len(line) > 0:
            if 'INPUT' in line:
                input_gate_name = int(line[7:-4])
                if input_gate_name not in vars_dict:
                    if circuit_number == 1:
                        vars_dict[input_gate_name] = current_var_id
                        inputs_names.append([input_gate_name, current_var_id])
                        current_var_id += 1
                    elif circuit_number == 2:
                        vars_dict[input_gate_name] = input_var_ids
                        inputs_names.append([input_gate_name, input_var_ids])
                        input_var_ids += 1
                    else:
                        raise Exception('Wrong scheme number while parsing bench')
            elif 'OUTPUT' in line:
                output_gate_name = int(line[8:-4])
                outputs_names.append([output_gate_name, None])
            elif line[0] == 'G':
                internal_gate_name = int(line.split(' =')[0][1:-3])
                if internal_gate_name not in vars_dict:
                    if internal_gate_name % 2 == 0:
                        vars_dict[internal_gate_name] = current_var_id
                        current_var_id += 1
                    else:
                        if internal_gate_name - 1 in vars_dict:
                            vars_dict[internal_gate_name] = -vars_dict[internal_gate_name - 1]
                        else:
                            vars_dict[internal_gate_name - 1] = current_var_id
                            vars_dict[internal_gate_name] = -vars_dict[internal_gate_name - 1]
                            current_var_id += 1
    for output in outputs_names:
        if output[0] in vars_dict:
            output[1] = vars_dict[output[0]]
        elif output[0] - 1 in vars_dict:
            output[1] = vars_dict[output[0] - 1]
        else:
            raise Exception('Output not in vars dict')
    return vars_dict, outputs_names, inputs_names, current_var_id


def encode_gates(bench_lines, vars_dict):
    clauses = []
    for line in bench_lines:
        clauses_ = None
        if 'AND' in line:
            clauses_ = encode_and_gate(line, vars_dict)
        if clauses_ != None:
            clauses.extend(clauses_)
    return clauses


def encode_and_gate(line, var_map, circuit_flag=False):
    clauses = []
    gate_outp = int(line.split()[0][1:-3])
    gate_input1 = int(line.split('(')[1].split(',')[0][1:-3])
    gate_input2 = int(line.split()[3][:-1][1:-3])
    if not circuit_flag:
        var_outp = var_map[gate_outp]
        var_input1 = var_map[gate_input1]
        var_input2 = var_map[gate_input2]
    else:
        var_outp = var_map[gate_outp]
        var_input1 = var_map[gate_input1]
        var_input2 = var_map[gate_input2]
    clauses.extend([[var_outp, -var_input1, -var_input2], [-var_outp, var_input1], [-var_outp, var_input2]])
    return clauses


def list_to_clauses(clist, vars_dict, constraints: List[Tuple[int, int]]):
    clauses = []
    for l in clist:
        clause = ' '.join(list(map(str, l))) + ' 0'
        clauses.append(clause)
    if constraints:
        for constraint in constraints:
            first, second = constraint
            if first % 2 > 0:
                fst = -vars_dict[first - 1]
            else:
                fst = vars_dict[first]

            if second % 2 > 0:
                scd = -vars_dict[second - 1]
            else:
                scd = vars_dict[second]

            clause = ' '.join(list(map(str, [fst, scd]))) + ' 0'
            clauses.append(clause)
    return clauses


def list_to_clause(clist, vars_dict, constraints: List[Tuple[int, int]]):
    clauses = []
    for l in clist:
        clauses.append(l)
    if constraints:
        for constraint in constraints:
            first, second = constraint
            if first % 2 > 0:
                fst = -vars_dict[first - 1]
            else:
                fst = vars_dict[first]

            if second % 2 > 0:
                scd = -vars_dict[second - 1]
            else:
                scd = vars_dict[second]

            clauses.append([fst, scd])
    return clauses


def aig2bench_from_file_lines(file_lines):
    aig_file_lines = file_lines.split('\n')
    aig_file_lines = aig_file_lines[:-1]
    test_name = 'test_name'
    aig_header = aig_file_lines[0].split()
    nof_inputs = int(aig_header[2])
    nof_outputs = int(aig_header[4])
    del aig_file_lines[0]
    not_gates = dict()

    inputs_list = []
    for i in range(nof_inputs):
        inputs_list.append(int(aig_file_lines[0]))
        del aig_file_lines[0]
    outputs_list = []
    for i in range(nof_outputs):
        outputs_list.append(int(aig_file_lines[0]))
        if int(aig_file_lines[0]) % 2 > 0:
            not_tuple = (int(aig_file_lines[0]), int(aig_file_lines[0]) - 1)
            if not_tuple not in not_gates:
                not_gates[not_tuple] = 1
        del aig_file_lines[0]

    const_false = False
    const_true = False
    and_gates = []
    for line in aig_file_lines:
        if line[0].isdigit():
            and_gate_name = int(line.split()[0])
            first_input = int(line.split()[1])
            second_input = int(line.split()[2])
            if first_input % 2 > 0 and first_input > 1:
                not_tuple = (first_input, first_input - 1)
                if not_tuple not in not_gates:
                    not_gates[not_tuple] = 1
            if second_input % 2 > 0 and second_input > 1:
                not_tuple = (second_input, second_input - 1)
                if not_tuple not in not_gates:
                    not_gates[not_tuple] = 1
            if first_input == 0 or second_input == 0:
                const_false = True
            if first_input == 1 or second_input == 1:
                const_true = True
            and_tuple = tuple([and_gate_name, first_input, second_input])
            and_gates.append(and_tuple)

    if const_false or const_true:
        if tuple([3, 2]) not in not_gates.keys():
            not_gates[tuple([3, 2])] = 1
        and_gates.append(tuple([0, 2, 3]))
        if const_true:
            not_gates[tuple([1, 0])] = 1
    result = ['# testname: ' + str(test_name),
              '#         lines from primary input  gates .......     ' + str(len(inputs_list)),
              '#         lines from primary output gates .......     ' + str(len(outputs_list)),
              '#         lines from primary not gates .......     ' + str(len(not_gates)),
              '#         lines from primary and gates .......     ' + str(len(and_gates))]
    for inp in sorted(inputs_list):
        input_line = 'INPUT(G' + str(inp) + 'gat)'
        result.append(input_line)
    for out in outputs_list:
        output_line = 'OUTPUT(G' + str(out) + 'gat)'
        result.append(output_line)
    result.append('')
    lines = []
    for not_gate in not_gates.keys():
        not_gate_out = not_gate[0]
        not_gate_inp = not_gate[1]
        not_line = 'G' + str(not_gate_out) + 'gat = NOT(G' + str(not_gate_inp) + 'gat)'
        lines.append((not_line, not_gate))
    for and_gate in and_gates:
        and_gate_out = and_gate[0]
        and_gate_inp1 = and_gate[1]
        and_gate_inp2 = and_gate[2]
        and_line = 'G' + str(and_gate_out) + 'gat = AND(G' + str(and_gate_inp1) + 'gat, G' + str(
            and_gate_inp2) + 'gat)'
        lines.append((and_line, and_gate))
    lines.sort(key=lambda x: int(max(x[1])))
    lines = [x[0] for x in lines]
    result += lines
    return result


def aig2bench(filename):
    aig_filename = filename
    test_name = (aig_filename.split('/')[-1]).split('.')[0]
    with open(aig_filename, "r") as f:
        aig_file_lines = f.readlines()
    aig_header = aig_file_lines[0].split()
    nof_inputs = int(aig_header[2])
    nof_outputs = int(aig_header[4])
    del aig_file_lines[0]
    not_gates = dict()

    inputs_list = []
    for i in range(nof_inputs):
        inputs_list.append(int(aig_file_lines[0]))
        del aig_file_lines[0]

    outputs_list = []
    for i in range(nof_outputs):
        outputs_list.append(int(aig_file_lines[0]))
        if int(aig_file_lines[0]) % 2 > 0:
            not_tuple = (int(aig_file_lines[0]), int(aig_file_lines[0]) - 1)
            if not_tuple not in not_gates:
                not_gates[not_tuple] = 1
        del aig_file_lines[0]

    const_false = False
    const_true = False
    and_gates = []
    for line in aig_file_lines:
        if line[0].isdigit():
            and_gate_name = int(line.split()[0])
            first_input = int(line.split()[1])
            second_input = int(line.split()[2])
            if first_input % 2 > 0 and first_input > 1:
                not_tuple = (first_input, first_input - 1)
                if not_tuple not in not_gates:
                    not_gates[not_tuple] = 1
            if second_input % 2 > 0 and second_input > 1:
                not_tuple = (second_input, second_input - 1)
                if not_tuple not in not_gates:
                    not_gates[not_tuple] = 1
            if first_input == 0 or second_input == 0:
                const_false = True
            if first_input == 1 or second_input == 1:
                const_true = True
            and_tuple = tuple([and_gate_name, first_input, second_input])
            and_gates.append(and_tuple)

    if const_false or const_true:
        if tuple([3, 2]) not in not_gates.keys():
            not_gates[tuple([3, 2])] = 1
        and_gates.append(tuple([0, 2, 3]))
        if const_true:
            not_gates[tuple([1, 0])] = 1
    result = ['# testname: ' + str(test_name),
              '#         lines from primary input  gates .......     ' + str(len(inputs_list)),
              '#         lines from primary output gates .......     ' + str(len(outputs_list)),
              '#         lines from primary not gates .......     ' + str(len(not_gates)),
              '#         lines from primary and gates .......     ' + str(len(and_gates))]
    for inp in sorted(inputs_list):
        input_line = 'INPUT(G' + str(inp) + 'gat)'
        result.append(input_line)
    for out in outputs_list:
        output_line = 'OUTPUT(G' + str(out) + 'gat)'
        result.append(output_line)
    result.append('')
    lines = []
    for not_gate in not_gates.keys():
        not_gate_out = not_gate[0]
        not_gate_inp = not_gate[1]
        not_line = 'G' + str(not_gate_out) + 'gat = NOT(G' + str(not_gate_inp) + 'gat)'
        lines.append((not_line, not_gate))
    for and_gate in and_gates:
        and_gate_out = and_gate[0]
        and_gate_inp1 = and_gate[1]
        and_gate_inp2 = and_gate[2]
        and_line = 'G' + str(and_gate_out) + 'gat = AND(G' + str(and_gate_inp1) + 'gat, G' + str(
            and_gate_inp2) + 'gat)'
        lines.append((and_line, and_gate))
    lines.sort(key=lambda x: int(max(x[1])))
    lines = [x[0] for x in lines]
    result += lines
    return result


def encode_output_equiv(max_var, outputs_names):
    new_clauses = []
    current_var = max_var
    for output_name in outputs_names:
        output_var = output_name[1]
        new_clauses_, current_var = encode_equiv_clauses(output_var, current_var)
        output_name[1] = current_var
        new_clauses.extend(new_clauses_)
    return new_clauses, current_var


def encode_equiv_clauses(output_var, current_var):
    current_var += 1
    clauses = [[-current_var, output_var], [current_var, -output_var]]
    return clauses, current_var


def make_header_and_comments(len_clauses_list, max_var, inputs_names, outputs_names, layers_vars):
    header = 'p cnf ' + str(max_var) + ' ' + str(len_clauses_list)
    comment_inputs = 'c inputs: ' + ' '.join([str(x[1]) for x in inputs_names])
    comment_outputs = 'c outputs: ' + ' '.join([str(x[1]) for x in outputs_names])
    comment_layers = make_layers_comment(layers_vars)
    comments = [comment_inputs, comment_outputs, comment_layers]
    return header, comments


def make_layers_comment(layers_vars):
    comment = 'c layers:'
    n = 0
    for layer in layers_vars:
        comment += ' [[' if n == 0 else ', ['
        comment += ', '.join([str(x) for x in layer])
        comment += ']'
    comment += ']'
    return comment


def write_out_cnf(lec_filename, header, comments, dimacs_cnf):
    with open(lec_filename, 'w') as outf:
        print(header, file=outf)
        print(*comments, sep='\n', file=outf)
        print(*dimacs_cnf, sep='\n', file=outf)


def get_raw_cnf_data(header, comments, dimacs_cnf):
    return ''.join([
        header + '\n',
        *[f'{x}\n' for x in comments],
        *[f'{x}\n' for x in dimacs_cnf]
    ])


def get_topsort_layers(bench, var_map):
    vars_layers = dict()
    for line in bench:
        if 'and' in line:
            gate_out = int(line.split()[0][1:-3])
            gate_input1 = int(line.split('(')[1].split(',')[0][1:-3])
            gate_input2 = int(line.split()[3][:-1][1:-3])
            var_out = var_map[gate_out]
            var_input1 = var_map[gate_input1]
            var_input2 = var_map[gate_input2]
            vars_layers[var_out] = {var_input1, var_input2}
    vars_layers_list = list(toposort(vars_layers))
    return vars_layers_list


class AAG2CNF:
    def __init__(self):
        self._cnf_to_aag_inputs_vars = None
        self._aag_to_cnf_inputs_vars = None

    def aag_to_cnf_raw_data(self, file_lines, constraints):
        bench = aig2bench_from_file_lines(file_lines)

        bench, bench_test_name, nof_inputs, nof_outputs, nof_and_gates, nof_not_gates = get_bench_header(bench)
        current_var_id = 1
        vars_dict, outputs_names, inputs_names, current_var_id = parse_bench_to_map(bench, current_var_id, 1)
        max_var = current_var_id - 1

        clauses_list = encode_gates(bench, vars_dict)

        if [x[1] for x in outputs_names] != list(range(max_var - len(outputs_names) + 1, max_var + 1)):
            output_equiv_clauses, max_var = encode_output_equiv(max_var, outputs_names)
            clauses_list.extend(output_equiv_clauses)

        layers_vars = get_topsort_layers(bench, vars_dict)

        header, comments = make_header_and_comments(
            len(clauses_list), max_var, inputs_names, outputs_names, layers_vars
        )

        self._cnf_to_aag_inputs_vars = {inputs_name[1]: inputs_name[0] for inputs_name in inputs_names}
        self._aag_to_cnf_inputs_vars = {inputs_name[0]: inputs_name[1] for inputs_name in inputs_names}

        clauses = list_to_clauses(clauses_list, vars_dict, constraints)
        raw_cnf_data = get_raw_cnf_data(header, comments, clauses)
        return raw_cnf_data

    @property
    def aag_to_cnf_inputs_vars(self) -> Dict[int, int]:
        return self._aag_to_cnf_inputs_vars

    @property
    def cnf_to_aag_inputs_vars(self) -> Dict[int, int]:
        return self._cnf_to_aag_inputs_vars


__all__ = [
    'AAG2CNF'
]
