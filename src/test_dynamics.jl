using LinearAlgebra
using TensorOperations
include("IMBP.jl")

# gates initialization
theta = 1.
x = (-im * theta / 2) * reshape(ComplexF64[0, 1, 1, 0], 2, 2)
x_gate = exp(x)
zz_gate = reshape(Diagonal(exp.((pi * im / 4) * ComplexF64[1, -1, -1, 1])), 2, 2, 2, 2)
@tensor zz_x_gate[i, j, k, l] := zz_gate[i, j, k, q] * x_gate[q, l]
@tensor zz_x_x_gate[i, j, k, l] := zz_gate[i, j, q, p] * x_gate[q, k] * x_gate[p, l]

@tensor zz_gate[i1, j1, k1, l1, i2, j2, k2, l2] := zz_gate[i1, j1, k1, l1] * conj(zz_gate[i2, j2, k2, l2])
@tensor zz_x_gate[i1, j1, k1, l1, i2, j2, k2, l2] := zz_x_gate[i1, j1, k1, l1] * conj(zz_x_gate[i2, j2, k2, l2])
@tensor zz_x_x_gate[i1, j1, k1, l1, i2, j2, k2, l2] := zz_x_x_gate[i1, j1, k1, l1] * conj(zz_x_x_gate[i2, j2, k2, l2])

initial_state = reshape(ComplexF64[1, 0, 0, 0], 2, 2)

lattice_cell = IMBP.LatticeCell(ComplexF64, [initial_state for _ in 1:5])
IMBP.add_gate!(lattice_cell, 1, 2, zz_x_x_gate)
IMBP.add_gate!(lattice_cell, 4, 5, zz_x_x_gate)
IMBP.add_gate!(lattice_cell, 2, 4, zz_gate)
IMBP.add_gate!(lattice_cell, 5, 3, zz_x_gate)
IMBP.add_gate!(lattice_cell, 2, 3, zz_gate)
IMBP.add_gate!(lattice_cell, 1, 5, zz_gate)
equations = IMBP.get_equations(lattice_cell)
ims = IMBP.initialize_ims_by_perfect_dissipators(IMBP.IM{ComplexF64}, lattice_cell, 50)
IMBP.iterate_equations!(equations, ims, 10, 100, -1e-8)
