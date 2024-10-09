include("../src/IMBP.jl")

using Random

dissipator = IMBP.perfect_dissipator_kernel(Int32, 2)
@assert(size(dissipator) == (1, 4, 4, 1))
@assert(reshape(dissipator, :) == [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1])
diss_im = IMBP.get_perfect_dissipator_im(IMBP.IM{ComplexF64}, 20)
@assert(IMBP.get_time_steps_number(diss_im) == 20)
@assert(IMBP.get_bond_dimensions(diss_im) == [1 for _ in 1:21])

rng = MersenneTwister(1234);
rand_im = IMBP._random_im(ComplexF64, rng, 64, 20)
@assert(IMBP.get_time_steps_number(rand_im) == 20)
@assert(IMBP.get_bond_dimensions(rand_im) == vcat([1], [64 for _ in 1:19], [1]))
IMBP._set_to_left_canonical!(rand_im)
@assert(abs(IMBP.log_fidelity(rand_im, rand_im)) < 1e-10)
rand_im_clone1 = deepcopy(rand_im)
rand_im_clone2 = deepcopy(rand_im)
IMBP._truncate_left_canonical!(rand_im_clone1, 63)
IMBP._truncate_left_canonical!(rand_im_clone2, 1e-5)
@assert(abs(IMBP.log_fidelity(rand_im_clone2, rand_im)) < 1e-5)

ims = Dict{IMBP.IMID, IMBP.IM{ComplexF64}}(
    IMBP.IMID(1, true, (1, 0)) => IMBP._random_im(ComplexF64, rng, 3, 20),
    IMBP.IMID(2, false, (0, 2)) => IMBP._random_im(ComplexF64, rng, 3, 20),
    IMBP.IMID(4, false, (0, 3)) => IMBP._random_im(ComplexF64, rng, 3, 20),
)

kernels = Dict{IMBP.KernelID, Array{ComplexF64, 4}}(
    IMBP.KernelID(3, true, (0, 4)) => randn(rng, ComplexF64, 4, 4, 4, 4),
)

one_qubit_gate = randn(rng, ComplexF64, 4, 4)

equation = IMBP.ElementID[
    IMBP.IMID(1, true, (1, 0)),
    IMBP.IMID(2, false, (0, 2)),
    IMBP.KernelID(3, true, (0, 4)),
    IMBP.IMID(4, false, (0, 3)),
]

initial_state = randn(rng, ComplexF64, 4)

new_im = IMBP.contract(equation, ims, kernels, one_qubit_gate, initial_state)
for (i, ker) in enumerate(new_im.kernels)
    if i == 1
        @assert size(ker) == (1, 4, 4, 108)
    elseif i == 20
        @assert size(ker) == (108, 4, 4, 1)
    else
        @assert size(ker) == (108, 4, 4, 108)
    end
end