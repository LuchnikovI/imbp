using LinearAlgebra
using Random
using Plots
using IMBP
using IMBP.ExactSim

# 1D TFI Floquet model with Z field
g = 0.685    # mixing gate amplitude
J = 0.31 # coupling amplitude
h = 0.2 # Z field amplitude
time_steps = 20 # time steps number
bond_dimension = 128

initial_state = ComplexF64[0.5 0.5 ; 0.5 0.5]

# the API is designed in such a way that a single-qubit gates come always before two-qubit ones
# this is why below we aggregate one qubit gates into two-qubit ones
int_gate = exp(im * (
    J * ComplexF64[1 0 0 0 ; 0 -1 0 0 ; 0 0 -1 0 ; 0 0 0 1]
    +
    h * ComplexF64[1 0 0 0 ; 0 0 0 0 ; 0 0 0 0 ; 0 0 0 -1]
))
int_channel = kron(int_gate, conj(int_gate))  # purely interaction gate
mixing_gate = exp(im * g * ComplexF64[0 1 ; 1 0])
tf_gate = kron(mixing_gate, mixing_gate)
tf_int_gate = tf_gate * int_gate
tf_int_channel = kron(tf_int_gate, conj(tf_int_gate))  # interaction gate with mixing gates aggregated

rng = MersenneTwister(42)

# see `infinite_ibm_heavy_hex.jl`, here one uses the same API but for single loop topology
# of a unit cell that corresponds to an infinite chain
lattice_cell = IMBP.LatticeCell([initial_state for _ in 1:2])
add_two_qubit_gates!(lattice_cell, 1, 2, [int_channel for _ in 1:time_steps])
add_two_qubit_gates!(lattice_cell, 2, 1, [tf_int_channel for _ in 1:time_steps])
equations = get_equations(lattice_cell)

ims = initialize_ims(SketchyIM{Array{ComplexF64, 4}}, lattice_cell)
info = iterate_equations!(equations, ims, bond_dimension,
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
dens_dyn = simulate_dynamics(
    2,
    equations,
    ims,
    ComplexF64[0.5 0.5 ; 0.5 0.5],
)
x_dyn = real(map(x -> x[1, 2] + x[2, 1], dens_dyn))

# exact (state vector) simulation for 20 qubits chain
qs = QSim(Float64, 26)
hadamard = (1 / sqrt(2)) * ComplexF64[1 1 ; 1 -1]
for i in 1:26
    apply_one_qubit_gate!(qs, hadamard, i)
end
sv_dens_dyn = Matrix{ComplexF64}[]
for t in 1:time_steps
    push!(sv_dens_dyn, get_one_qubit_dens(qs, 13))
    for pos in 1:2:26
        apply_two_qubit_gate!(qs, int_gate, pos, pos + 1)
    end
    for pos in 2:2:26
        apply_two_qubit_gate!(qs, tf_int_gate, pos, (pos + 1) % 26)
    end
end
sv_x_dyn = real(map(x -> x[1, 2] + x[2, 1], sv_dens_dyn))

# plot the dynamics comparison (must diverge for long time since exact simulation is for a finite system (26 qubits))
plot([x_dyn, sv_x_dyn], ylim = (0, 1))