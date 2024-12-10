using LinearAlgebra
using Random
using Plots
using IMBP
using IMBP.ExactSim

# This example is the simulation of a single qubit dynamics for an infinite IBM heavy hex lattice
# one can compare results of https://arxiv.org/pdf/2306.14887 with given simulation

# mixing quantum channel
theta = 0.6  # is the duration of mixing gate from https://arxiv.org/pdf/2306.14887
time_steps = 20  # the simulation time in https://arxiv.org/pdf/2306.14887 is up to 20
bond_dim = 64  # must not be too large, e.g. < 100, otherwise simulation either runs out of memory or is painfully slow
seed = 42  # random seed

rng = MersenneTwister(seed)

# mixing quantum channels (along the time line all of them the same for a given example)
mixing_gate = exp((-im * theta / 2) * ComplexF64[0 1 ; 1 0])
mixing_channel = [kron(mixing_gate, conj(mixing_gate)) for _ in 1:time_steps]

# interaction quantum channels (along the time line all of them the same for a given example)
int_gate = diagm(exp.((pi * im / 4) * ComplexF64[1, -1, -1, 1]))
int_channel = [kron(int_gate, conj(int_gate)) for _ in 1:time_steps]

# creating a cell lattice with all the initial density matrices being the same
lattice_cell = IMBP.LatticeCell([ComplexF64[1 0 ; 0 0] for _ in 1:5])

# populating a lattice cell with one-qubit gates
for pos in 1:5
    add_one_qubit_gates!(lattice_cell, pos, mixing_channel)
end

# populating a lattice with two-qubit gates, lattice unit cell has the following topology
#    1 -- 2 -- 3
#     \   |   /
#      \  4  /
#       \ | /
#         5
# (See Fig. 5 Unit Cell + PBC in https://arxiv.org/pdf/2306.14887)
# 2-nd qubit's density matrix is evaluated
# the relative order of two-qubit gates does not matter for this example,
# since they commute with each other,
# but in general it matters -- the order of `add_two_qubit_gates` is executed
# is the order of gates application to the state
add_two_qubit_gates!(lattice_cell, 1, 2, int_channel)
add_two_qubit_gates!(lattice_cell, 4, 5, int_channel)
add_two_qubit_gates!(lattice_cell, 2, 4, int_channel)
add_two_qubit_gates!(lattice_cell, 5, 3, int_channel)
add_two_qubit_gates!(lattice_cell, 2, 3, int_channel)
add_two_qubit_gates!(lattice_cell, 1, 5, int_channel)

# extracting self-consistency equations from the lattice cell
equations = get_equations(lattice_cell)

# initializing IM by empty ones (LCGA generalized to BP)
# SketchyIM implements fast (randomized sketching based) implementation of self consistency equation update
# using IM instead is painfully slow in most of the cases
ims = initialize_ims(SketchyIM{Array{ComplexF64, 4}}, lattice_cell)

# run BP to solve equations (one-qubit gates layer applied before two-qubit gates layer)
info = iterate_equations!(equations, ims, bond_dim,
    # callback that prints information about past iteration and some parameters of IMs from past iteration
    (iter_num, info, ims) -> begin
        println("Iteration number: ", iter_num)
        println(info)
        println("Some IMs parameters:")
        for (key, val) in ims
            println("\t", key, ":", "\n\t\ttime steps number: ", get_time_steps_number(val), "\n\t\tbond dimensions: ", get_bond_dimensions(val))
        end
        print('\n')
    end;
    rng=rng, sample_size=25)

    # compute dynamics of a qubit
dens_dyn = simulate_dynamics(
    2,                      # id of a qubit whose dynamics is computed
    equations,              # self consistency equations
    ims,                    # current influence matrices
)

# extracting dynamics of Z component
z_dyn = real(map(x -> x[1, 1] - x[2, 2], dens_dyn))


# exact (state vector) simulation of part of the infinite IBM processor that has the following topology
#
# 1 -- 2 -- 3 -- 4 -- 5 -- 6 -- 7 -- 8 -- 9
# |                   |                   |
# 10                  11                  12
# |                   |                   |
# 13-- 14-- 15-- 16-- 17-- 18-- 19-- 20-- 21
#           |                   |
#           22                  23
#           |                   |
#           24-- 25-- 26-- 27 --28
#
# the density matrix of 17-th qubit is evaluated
# this simulation also takes some time
qs = QSim(Float64, 28)
sv_dens_dyn = Matrix{ComplexF64}[]
for t in 1:time_steps
    push!(sv_dens_dyn, get_one_qubit_dens(qs, 17))
    for i in 1:28
        apply_one_qubit_gate!(qs, mixing_gate, i)
    end
    apply_two_qubit_gate!(qs, int_gate, 1, 2)
    apply_two_qubit_gate!(qs, int_gate, 2, 3)
    apply_two_qubit_gate!(qs, int_gate, 3, 4)
    apply_two_qubit_gate!(qs, int_gate, 4, 5)
    apply_two_qubit_gate!(qs, int_gate, 5, 6)
    apply_two_qubit_gate!(qs, int_gate, 6, 7)
    apply_two_qubit_gate!(qs, int_gate, 7, 8)
    apply_two_qubit_gate!(qs, int_gate, 8, 9)
    apply_two_qubit_gate!(qs, int_gate, 1, 10)
    apply_two_qubit_gate!(qs, int_gate, 5, 11)
    apply_two_qubit_gate!(qs, int_gate, 9, 12)
    apply_two_qubit_gate!(qs, int_gate, 10, 13)
    apply_two_qubit_gate!(qs, int_gate, 11, 17)
    apply_two_qubit_gate!(qs, int_gate, 12, 21)
    apply_two_qubit_gate!(qs, int_gate, 13, 14)
    apply_two_qubit_gate!(qs, int_gate, 14, 15)
    apply_two_qubit_gate!(qs, int_gate, 15, 16)
    apply_two_qubit_gate!(qs, int_gate, 16, 17)
    apply_two_qubit_gate!(qs, int_gate, 17, 18)
    apply_two_qubit_gate!(qs, int_gate, 18, 19)
    apply_two_qubit_gate!(qs, int_gate, 19, 20)
    apply_two_qubit_gate!(qs, int_gate, 20, 21)
    apply_two_qubit_gate!(qs, int_gate, 15, 22)
    apply_two_qubit_gate!(qs, int_gate, 19, 23)
    apply_two_qubit_gate!(qs, int_gate, 22, 24)
    apply_two_qubit_gate!(qs, int_gate, 23, 28)
    apply_two_qubit_gate!(qs, int_gate, 24, 25)
    apply_two_qubit_gate!(qs, int_gate, 25, 26)
    apply_two_qubit_gate!(qs, int_gate, 26, 27)
    apply_two_qubit_gate!(qs, int_gate, 27, 28)
end
push!(sv_dens_dyn, get_one_qubit_dens(qs, 17))

# extracting dynamics of Z component
sv_z_dyn = real(map(x -> x[1, 1] - x[2, 2], sv_dens_dyn))

# plot the dynamics comparison (must diverge for long time since exact simulation is for a finite system (28 qubits))
plot([z_dyn, sv_z_dyn])
