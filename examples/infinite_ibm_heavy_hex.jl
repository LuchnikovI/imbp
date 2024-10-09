using LinearAlgebra
using TensorOperations
using Plots
include("../src/IMBP.jl")
using .IMBP

# mixing quantum channl
theta = 0.6
mixing_gate = exp((-im * theta / 2) * ComplexF64[0 1 ; 1 0])
mixing_channel = kron(mixing_gate, conj(mixing_gate))

# interaction quantum channel 
int_gate = diagm(exp.((pi * im / 4) * ComplexF64[1, -1, -1, 1]))
int_channel = kron(int_gate, conj(int_gate))


initial_state = ComplexF64[1 0 ; 0 0]

lattice_cell = IMBP.LatticeCell(ComplexF64, [initial_state for _ in 1:5])
for pos in 1:5
    add_one_qubit_gate!(lattice_cell, pos, mixing_channel)
end
add_two_qubit_gate!(lattice_cell, 1, 2, int_channel)
add_two_qubit_gate!(lattice_cell, 4, 5, int_channel)
add_two_qubit_gate!(lattice_cell, 2, 4, int_channel)
add_two_qubit_gate!(lattice_cell, 5, 3, int_channel)
add_two_qubit_gate!(lattice_cell, 2, 3, int_channel)
add_two_qubit_gate!(lattice_cell, 1, 5, int_channel)
equations = get_equations(lattice_cell)
ims = initialize_ims_by_perfect_dissipators(IMBP.IM{ComplexF64}, lattice_cell, 20)
iterate_equations!(equations, ims, 8, 100, -1e-8)
dens_dyn = IMBP.simulate_dynamics(
    2,
    equations,
    ims,
    ComplexF64[1 0 ; 0 0],
)
z_dyn = real(map(x -> x[1, 1] - x[2, 2], dens_dyn))
plot(z_dyn, ylim = (0, 1))
