@testset "Test Sketchy Influence Matrix" begin
    dissipator = get_array(IMBP.perfect_dissipator_kernel(Array{ComplexF64, 4}), :binp, :pinp, :pout, :bout)
    @test(size(dissipator) == (1, 4, 4, 1))
    @test(reshape(dissipator, :) == [0.5, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0.5])
    diss_im = IMBP.get_perfect_dissipator_im(IMBP.SketchyIM{Array{ComplexF64, 4}}, 20)
    @test(IMBP.get_time_steps_number(diss_im) == 20)
    @test(IMBP.get_bond_dimensions(diss_im) == [1 for _ in 1:21])

    rng = MersenneTwister(1234);
    ims = Dict{IMBP.IMID, IMBP.SketchyIM{Array{ComplexF64, 4}}}(
        IMBP.IMID(1, true, (1, 0)) => IMBP._random_im(IMBP.SketchyIM{Array{ComplexF64, 4}}, rng, 3, 20),
        IMBP.IMID(2, false, (0, 2)) => IMBP._random_im(IMBP.SketchyIM{Array{ComplexF64, 4}}, rng, 3, 20),
        IMBP.IMID(4, false, (0, 3)) => IMBP._random_im(IMBP.SketchyIM{Array{ComplexF64, 4}}, rng, 3, 20),
    )

    kernels = Dict{IMBP.KernelID, Node{Array{ComplexF64, 4}}}(
        IMBP.KernelID(3, true, (0, 4)) => Node(randn(rng, ComplexF64, 4, 4, 4, 4), :first_inp, :second_inp, :first_out, :second_out),
    )

    one_qubit_gate = Node(randn(rng, ComplexF64, 4, 4), :pinp, :pout)

    equation = IMBP.ElementID[
        IMBP.IMID(1, true, (1, 0)),
        IMBP.IMID(2, false, (0, 2)),
        IMBP.KernelID(3, true, (0, 4)),
        IMBP.IMID(4, false, (0, 3)),
    ]

    initial_state = Node(randn(rng, ComplexF64, 4), :pout)

    new_im, _ = IMBP.contract(equation, ims, kernels, one_qubit_gate, initial_state, 50; rng)
    for (i, ker) in enumerate(new_im.kernels)
        if i == 1
            @test size(get_array(ker, :binp, :pinp, :pout, :bout)) == (1, 4, 4, 16)
        elseif i == 2
            @test size(get_array(ker, :binp, :pinp, :pout, :bout)) == (16, 4, 4, 50)
        elseif i == 20
            @test size(get_array(ker, :binp, :pinp, :pout, :bout)) == (50, 4, 4, 1)
        else
            @test size(get_array(ker, :binp, :pinp, :pout, :bout)) == (50, 4, 4, 50)
        end
    end
end