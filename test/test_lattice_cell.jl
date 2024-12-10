@testset "Test Lattice Cell" begin
    rng = MersenneTwister(42)
    true_initial_states = [random_state(Float64, rng, 2) for _ in 1:6]
    gates = [map(u -> kron(u, conj(u)), [random_unitary(Float64, rng, 4) for _ in 1:5]) for _ in 1:6]
    true_kernels = map(y -> map(x -> reshape(permutedims(reshape(x, (2, 2, 2, 2, 2, 2, 2, 2)), (3, 1, 4, 2, 7, 5, 8, 6)), (4, 4, 4, 4)), y), gates)
    true_flipped_kernels = map(y -> map(x -> Node(x, :first_out, :second_out, :first_inp, :second_inp), y), true_kernels)
    true_kernels = map(y -> map(x -> Node(x, :second_out, :first_out, :second_inp, :first_inp), y), true_kernels)

    lattice_cell = LatticeCell(true_initial_states)
    add_two_qubit_gates!(lattice_cell, 1, 2, gates[1])
    add_two_qubit_gates!(lattice_cell, 4, 5, gates[2])
    add_two_qubit_gates!(lattice_cell, 3, 5, gates[3])
    add_two_qubit_gates!(lattice_cell, 5, 1, gates[4])
    add_two_qubit_gates!(lattice_cell, 4, 2, gates[5])
    add_two_qubit_gates!(lattice_cell, 3, 2, gates[6])
    (; self_consistency_eqs, marginal_eqs, kernels, initial_states) = get_equations(lattice_cell)

    # marginal equations testing
    @test marginal_eqs[1] == [
        IMBP.IMID(1, false, (1, 2)),
        IMBP.IMID(4, true, (5, 1)),
    ]
    @test marginal_eqs[2] == [
        IMBP.IMID(1, true, (1, 2)),
        IMBP.IMID(5, true, (4, 2)),
        IMBP.IMID(6, true, (3, 2)),
    ]
    @test marginal_eqs[3] == [
        IMBP.IMID(3, false, (3, 5)),
        IMBP.IMID(6, false, (3, 2)),
    ]
    @test marginal_eqs[4] == [
        IMBP.IMID(2, false, (4, 5)),
        IMBP.IMID(5, false, (4, 2)),
    ]
    @test marginal_eqs[5] == [
        IMBP.IMID(2, true, (4, 5)),
        IMBP.IMID(3, true, (3, 5)),
        IMBP.IMID(4, false, (5, 1)),
    ]

    # self consistency testing (order is important)
    @test self_consistency_eqs == [
        [
            IMBP.KernelID(1, false, (1, 2)),
            IMBP.IMID(5, true, (4, 2)),
            IMBP.IMID(6, true, (3, 2)),
        ],
        [
            IMBP.KernelID(1, true, (1, 2)),
            IMBP.IMID(4, true, (5, 1)),
        ],
        [
            IMBP.KernelID(2, false, (4, 5)),
            IMBP.IMID(3, true, (3, 5)),
            IMBP.IMID(4, false, (5, 1)),
        ],
        [
            IMBP.KernelID(2, true, (4, 5)),
            IMBP.IMID(5, false, (4, 2)),
        ],
        [
            IMBP.IMID(2, true, (4, 5)),
            IMBP.KernelID(3, false, (3, 5)),
            IMBP.IMID(4, false, (5, 1)),
        ],
        [
            IMBP.KernelID(3, true, (3, 5)),
            IMBP.IMID(6, false, (3, 2)),
        ],
        [
            IMBP.IMID(1, false, (1, 2)),
            IMBP.KernelID(4, false, (5, 1)),
        ],
        [
            IMBP.IMID(2, true, (4, 5)),
            IMBP.IMID(3, true, (3, 5)),
            IMBP.KernelID(4, true, (5, 1)),
        ],
        [
            IMBP.IMID(1, true, (1, 2)),
            IMBP.KernelID(5, false, (4, 2)),
            IMBP.IMID(6, true, (3, 2)),
        ],
        [
            IMBP.IMID(2, false, (4, 5)),
            IMBP.KernelID(5, true, (4, 2)),
        ],
        [
            IMBP.IMID(1, true, (1, 2)),
            IMBP.IMID(5, true, (4, 2)),
            IMBP.KernelID(6, false, (3, 2)),
        ],
        [
            IMBP.IMID(3, false, (3, 5)),
            IMBP.KernelID(6, true, (3, 2)),
        ],
    ]

    # kernels testing
    @test kernels == Dict{IMBP.KernelID, Vector{Node{Array{ComplexF64, 4}}}}(
        IMBP.KernelID(1, false, (1, 2)) => true_flipped_kernels[1],
        IMBP.KernelID(2, false, (4, 5)) => true_flipped_kernels[2],
        IMBP.KernelID(3, false, (3, 5)) => true_flipped_kernels[3],
        IMBP.KernelID(4, false, (5, 1)) => true_flipped_kernels[4],
        IMBP.KernelID(5, false, (4, 2)) => true_flipped_kernels[5],
        IMBP.KernelID(6, false, (3, 2)) => true_flipped_kernels[6],
        IMBP.KernelID(1, true, (1, 2)) => true_kernels[1],
        IMBP.KernelID(2, true, (4, 5)) => true_kernels[2],
        IMBP.KernelID(3, true, (3, 5)) => true_kernels[3],
        IMBP.KernelID(4, true, (5, 1)) => true_kernels[4],
        IMBP.KernelID(5, true, (4, 2)) => true_kernels[5],
        IMBP.KernelID(6, true, (3, 2)) => true_kernels[6],
    )

    # test initial state
    @test initial_states == map(x -> Node(reshape(x, 4), :pout), true_initial_states)
end
