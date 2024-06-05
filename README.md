# fraig_revisited
Algorithms with And-Inverter graphs which use techniques based on fraig transformation of ABC verification tool. 

The well-known Cube-and-Conquer decomposition strategy was used not on Boolean formulas, but on a special class of 
graphs (And-Inverter Graphs, AIG). This made it possible to solve a number of extremely difficult problems much more
efficiently than using the standard approach, in which SAT solvers are applied directly to Boolean formulas encoding
the problems under consideration. This made it possible to solve a number of extremely difficult problems much more 
efficiently than using the standard approach, in which SAT solvers are applied directly to Boolean formulas encoding 
the problems under consideration.

## [FRAIG-transformation](./fraig) comprises:
- assignment of any vertex in scheme (with is possible due to additional restrictions &mdash; analogue of learnt clauses)
- upper propagation (towards inputs, can't be done in [ABC](https://github.com/berkeley-abc/abc))
- propagation towards outputs (downwards)
- reduction:
    - simulation (sampling)
    - SAT-oracle equivalence checking
    - equivalent vertices merging

### Code examples:

Example of solving logical equivalence checking problem for DvW<sub>12</sub> encoding
(Dadda and Karatsuba algorithms for multiplying pairs of 12-bit numbers) using branching heuristics based on choosing 
vertex with maximum degree, cutoff heuristics based on reaching cube tree maximum depth or leaf 
(which is actually AIG) size limit and transformation heuristic based on propagation and removing of assigned vertices.

```python
root_path = WorkPath('aiger')

LogicalEquivalenceChecking(
    left_scheme_file=root_path.to_file('dadda12x12.aag'),
    right_scheme_file=root_path.to_file('karatsuba12x12.aag'),
    branching_heuristic=DegreeBranchingHeuristic(),
    halt_heuristic=LeafSizeHaltingHeuristic(
        leafs_size_lower_bound=4450,
        max_depth=8,
        reduction_conflicts=300
    ),
    transformation_heuristics=FraigTransformationHeuristic(),
    work_path=WorkPath('out').to_path(date_now()),
    worker_solver=Solvers.ROKK_LRB
).solve_using_processes(
    number_of_executor_workers=number_of_workers
)
```

Example of solving inversion task of Simon 32/64 cryptographic function with 16 out of 64 bits of secret key known.

```python
Simon16KnownSecretKeyBitsInversion(
    aag=AAG(from_file=WorkPath('aiger').to_file(
      'simon_encrypt_encoding_keysize64_blocksize32_blocks2_rounds9.aag'
    )),
    open_text=[0x65656877, 0x65656876],
    secret_key=[0xe5e1, 0x3e5c, 0xfe34, 0x7a47],
    work_path=WorkPath('out').to_path(date_now()),
    branching_heuristic=DegreeBranchingHeuristic(),
    halt_heuristic=LeafSizeHaltingHeuristic(
        leafs_size_lower_bound=1000,
        max_depth=8,
        reduction_conflicts=300
    ),
    transformation_heuristics=FraigTransformationHeuristic(),
    executor_workers=32,
    worker_solver=Solvers.ROKK_LRB,
    task_type='randdist-2'
).cnc()
```

Example of estimating hardness of solving logical equivalence checking problem for DvW<sub>12</sub> encoding
(Dadda and Wallace algorithms for multiplying pairs of 12-bit numbers).
```python
root_path = WorkPath('aiger')

LogicalEquivalenceChecking(
    left_scheme_file=root_path.to_file('dadda12x12.aag'),
    right_scheme_file=root_path.to_file('wallace12x12.aag'),
    branching_heuristic=DegreeBranchingHeuristic(),
    halt_heuristic=HaltingHeuristic(
        max_depth=10,
        reduction_conflicts=300
    ),
    transformation_heuristics=FraigTransformationHeuristic(),
    work_path=WorkPath('out').to_path(date_now()),
    worker_solver=Solvers.ROKK_LRB
).estimate_using_mpi(
    use_input_decomposition=True,
    decomposition_limit=1000,
    length_decompose='2',
    shuffle_inputs=False
)
```

Other examples can be found in [examples](./examples) directory of project.