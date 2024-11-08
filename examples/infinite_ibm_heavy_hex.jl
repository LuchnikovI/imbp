using LinearAlgebra
using Random
using Plots
using CUDA
using IMBP

rng = MersenneTwister(42)

# mixing quantum channel
theta = 0.6
CUDA_ON = false

mixing_gate = exp((-im * theta / 2) * ComplexF64[0 1 ; 1 0])
mixing_gate = CUDA_ON ? CuArray(mixing_gate) : mixing_gate
mixing_channel = kron(mixing_gate, conj(mixing_gate))

# interaction quantum channel 
int_gate = diagm(exp.((pi * im / 4) * ComplexF64[1, -1, -1, 1]))
int_gate = CUDA_ON ? CuArray(int_gate) : int_gate
int_channel = kron(int_gate, conj(int_gate))

initial_state = ComplexF64[1 0 ; 0 0]
initial_state = CUDA_ON ? CuArray(initial_state) : initial_state

lattice_cell = IMBP.LatticeCell([initial_state for _ in 1:5])
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
ims = initialize_ims_by_perfect_dissipators(CUDA_ON ? SketchyIM{CuArray{ComplexF64, 4}} : SketchyIM{Array{ComplexF64, 4}}, lattice_cell, 40)
info = iterate_equations!(equations, ims, 25, 20, 1e-5; rng)
dens_dyn = simulate_dynamics(
    2,
    equations,
    ims,
    ComplexF64[1 0 ; 0 0],
)
z_dyn = real(map(x -> x[1, 1] - x[2, 2], dens_dyn))
plot(z_dyn, ylim = (0, 1))
