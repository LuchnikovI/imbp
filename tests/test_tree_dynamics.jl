include("../src/IMBP.jl")
include("utils.jl")

using .Utils
using .IMBP
using Random
using LinearAlgebra
using TensorOperations

rng = MersenneTwister(42)

layers_number = 4

qs = QSim(Float64, 6)
u_list = [random_unitary(Float64, rng, 2) for _ in 1:6]
for (pos, u) in enumerate(u_list)
    apply_one_qubit_gate!(qs, u, pos)
end

lc = LatticeCell(ComplexF64, [get_one_qubit_dens(qs, pos) for pos in 1:6])

u_list = [random_unitary(Float64, rng, 4) for _ in 1:5]
initial_dens = get_one_qubit_dens(qs, 3)
exact_dyn = Matrix{ComplexF64}[initial_dens]
for _ in 1:layers_number
    apply_two_qubit_gate!(qs, u_list[1], 1, 3)
    apply_two_qubit_gate!(qs, u_list[2], 3, 2)
    apply_two_qubit_gate!(qs, u_list[3], 4, 3)
    apply_two_qubit_gate!(qs, u_list[4], 4, 5)
    apply_two_qubit_gate!(qs, u_list[5], 6, 4)
    push!(exact_dyn, get_one_qubit_dens(qs, 3))
end

channel = kron(u_list[1], conj(u_list[1]))
add_two_qubit_gate!(lc, 1, 3, channel)
channel = kron(u_list[2], conj(u_list[2]))
add_two_qubit_gate!(lc, 3, 2, channel)
channel = kron(u_list[3], conj(u_list[3]))
add_two_qubit_gate!(lc, 4, 3, channel)
channel = kron(u_list[4], conj(u_list[4]))
add_two_qubit_gate!(lc, 4, 5, channel)
channel = kron(u_list[5], conj(u_list[5]))
add_two_qubit_gate!(lc, 6, 4, channel)

eqs = get_equations(lc)
ims = initialize_ims_by_perfect_dissipators(IM{ComplexF64}, lc, layers_number)
IMBP.iterate_equations!(eqs, ims, 1e-6, 30, -1e-12)

im_dyn = simulate_dynamics(eqs.marginal_eqs[3], ims, initial_dens)
for (im_dens, exact_dens) in zip(im_dyn, exact_dyn)
    @assert norm(im_dens - exact_dens) < 1e-6
end
