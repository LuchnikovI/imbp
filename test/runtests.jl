using IMBP
using Test
using Random
using LinearAlgebra
using SimpleTN

include("./utils.jl")

using .Utils

include("./test_exact_sim.jl")
include("./test_lattice_cell.jl")
include("./test_im.jl")
include("./test_tree_dynamics.jl")
include("./test_sketchy_im.jl")
include("./test_sketchy_tree_dynamics.jl")