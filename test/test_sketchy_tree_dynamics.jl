@testset "Test Sketchy Tree Dynamics" begin
    rng = MersenneTwister(42)

    layers_number = 4

    qs = QSim(Float64, 6)
    one_qubit_gate_list = [random_unitary(Float64, rng, 2) for _ in 1:6]
    for (pos, u) in enumerate(one_qubit_gate_list)
        apply_one_qubit_gate!(qs, u, pos)
    end

    lc = LatticeCell([get_one_qubit_dens(qs, pos) for pos in 1:6])

    two_qubit_gate_list = [random_unitary(Float64, rng, 4) for _ in 1:5]
    one_qubit_gate_list = [random_unitary(Float64, rng, 2) for _ in 1:6]
    initial_dens = get_one_qubit_dens(qs, 3)
    exact_dyn = Matrix{ComplexF64}[initial_dens]
    for _ in 1:layers_number
        for pos in 1:6
            apply_one_qubit_gate!(qs, one_qubit_gate_list[pos], pos)
        end
        apply_two_qubit_gate!(qs, two_qubit_gate_list[1], 1, 3)
        apply_two_qubit_gate!(qs, two_qubit_gate_list[2], 3, 2)
        apply_two_qubit_gate!(qs, two_qubit_gate_list[3], 4, 3)
        apply_two_qubit_gate!(qs, two_qubit_gate_list[4], 4, 5)
        apply_two_qubit_gate!(qs, two_qubit_gate_list[5], 6, 4)
        push!(exact_dyn, get_one_qubit_dens(qs, 3))
    end

    for pos in 1:6
        one_qubit_channel = kron(one_qubit_gate_list[pos], conj(one_qubit_gate_list[pos]))
        add_one_qubit_gate!(lc, pos, one_qubit_channel)
    end

    two_qubit_channel = kron(two_qubit_gate_list[1], conj(two_qubit_gate_list[1]))
    add_two_qubit_gate!(lc, 1, 3, two_qubit_channel)
    two_qubit_channel = kron(two_qubit_gate_list[2], conj(two_qubit_gate_list[2]))
    add_two_qubit_gate!(lc, 3, 2, two_qubit_channel)
    two_qubit_channel = kron(two_qubit_gate_list[3], conj(two_qubit_gate_list[3]))
    add_two_qubit_gate!(lc, 4, 3, two_qubit_channel)
    two_qubit_channel = kron(two_qubit_gate_list[4], conj(two_qubit_gate_list[4]))
    add_two_qubit_gate!(lc, 4, 5, two_qubit_channel)
    two_qubit_channel = kron(two_qubit_gate_list[5], conj(two_qubit_gate_list[5]))
    add_two_qubit_gate!(lc, 6, 4, two_qubit_channel)

    eqs = get_equations(lc)
    ims = initialize_ims_by_perfect_dissipators(IMBP.SketchyIM{Array{ComplexF64, 4}}, lc, layers_number)
    iterate_equations!(eqs, ims, 50, 30, 1e-5; rng)

    im_dyn = simulate_dynamics(3, eqs, ims, initial_dens)
    for (im_dens, exact_dens) in zip(im_dyn, exact_dyn)
        @test norm(im_dens - exact_dens) < 1e-3
    end
end
