@testset "Test Influence Matrix" begin
    dispatch_arr = ones(ComplexF64, 1)
    dissipator = IMBP.perfect_dissipator_kernel(dispatch_arr, 2)
    @test(size(dissipator) == (1, 4, 4, 1))
    @test(reshape(dissipator, :) == [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1])
    diss_im = IMBP.get_perfect_dissipator_im(IMBP.IM{Array{ComplexF64}}, dispatch_arr, 20)
    @test(IMBP.get_time_steps_number(diss_im) == 20)
    @test(IMBP.get_bond_dimensions(diss_im) == [1 for _ in 1:21])

    rng = MersenneTwister(1234);
    rand_im = IMBP._random_im(ComplexF64, rng, 64, 20)
    @test(IMBP.get_time_steps_number(rand_im) == 20)
    @test(IMBP.get_bond_dimensions(rand_im) == vcat([1], [64 for _ in 1:19], [1]))
    IMBP._set_to_left_canonical!(rand_im)
    @test(abs(IMBP.log_fidelity(rand_im, rand_im)) < 1e-10)
    rand_im_clone1 = deepcopy(rand_im)
    rand_im_clone2 = deepcopy(rand_im)
    IMBP._truncate_left_canonical!(rand_im_clone1, 63)
    IMBP._truncate_left_canonical!(rand_im_clone2, 1e-5)
    @test(abs(IMBP.log_fidelity(rand_im_clone2, rand_im)) < 1e-5)

    ims = Dict{IMBP.IMID, IMBP.IM{Array{ComplexF64}}}(
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

    new_im, _ = IMBP.contract(equation, ims, kernels, one_qubit_gate, initial_state, 1e-20)
    for (i, ker) in enumerate(new_im.kernels)
        if i == 1
            @test size(ker) == (1, 4, 4, 16)
        elseif i == 2
            @test size(ker) == (16, 4, 4, 108)
        elseif i ==19
            @test size(ker) == (108, 4, 4, 16)
        elseif i == 20
            @test size(ker) == (16, 4, 4, 1)
        else
            @test size(ker) == (108, 4, 4, 108)
        end
    end
end