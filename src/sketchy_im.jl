mutable struct SketchyIM{A<:AbstractArray} <: AbstractIM
    kernels::Vector{Node{A}}
    function SketchyIM(kernels::Vector{Node{A}}) where {A<:AbstractArray}
        _check_dims_consistency(kernels)
        new{A}(kernels)
    end
end

array_type(::Type{SketchyIM{Node{A}}}) where {A<:AbstractArray} = A
get_time_steps_number(im::SketchyIM) = length(im.kernels)

function _random_im(
    ::Type{SketchyIM{A}},
    rng,
    bond_dim::Integer,
    time_steps_number::Integer,
) where {A<:AbstractArray}
    kernels = Node{A}[]
    push!(kernels, Node(randn(rng, eltype(A), (1, 4, 4, bond_dim)), :binp, :pinp, :pout, :bout))
    for _ in 1:(time_steps_number - 2)
        push!(kernels, Node(randn(rng, eltype(A), (bond_dim, 4, 4, bond_dim)), :binp, :pinp, :pout, :bout))
    end
    push!(kernels, Node(randn(rng, eltype(A), (bond_dim, 4, 4, 1)), :binp, :pinp, :pout, :bout))
    SketchyIM(kernels)
end

#function get_perfect_dissipator_im(
#    ::Type{SketchyIM{A}},
#    time_steps_number::Integer,
#) where {A<:AbstractArray}
#    SketchyIM(Node{A}[perfect_dissipator_kernel(A) for _ in 1:time_steps_number])
#end

function initialize_im(
    ::Type{SketchyIM{A}},
) where {A<:AbstractArray}
    SketchyIM(Node{A}[])
end

function unitary_to_channel(u::Node)
    u_conj = conj.(u)
    change_id!(u_conj, :pinp, :pinp_conj)
    change_id!(u_conj, :pout, :pout_conj)
    phi = u[] * u_conj[]
    merge_axes(merge_axes(phi, :pinp, :pinp, :pinp_conj), :pout, :pout, :pout_conj)
end

function single_im_backward_dynamics(im::SketchyIM{A}) where {A<:AbstractArray}
    rhs_state = merge_axes(identity_from_array_type(A, 1, :bra, :ket), :binp, :bra, :ket)
    trace_out = merge_axes(identity_from_array_type(A, 2, :bra, :ket), :tr, :bra, :ket)
    rhs_states = []
    push!(rhs_states, rhs_state)
    for ker in im.kernels[length(im.kernels):-1:2]
        ker = ker[:pinp] * trace_out[:tr]
        ker = ker[:pout] * trace_out[:tr]
        rhs_state = ker[:bout] * rhs_state[:binp]
        push!(rhs_states, rhs_state)
    end
    reverse(rhs_states)
end

function single_im_dynamics(im::SketchyIM{A}, initial_state::Node, gates::Vector{<:Node}) where {A<:AbstractArray}
    lhs_state = split_axis(merge_axes(initial_state, :out, :bra, :ket), :out, (:bout, 1), (:pout, 4))
    rhs_states = single_im_backward_dynamics(im)
    dynamics = []
    for (ker, gate, rhs_state) in zip(im.kernels, gates, rhs_states)
        lhs_state = lhs_state[:pout, :bout] * ker[:pinp, :binp]
        lhs_state = lhs_state[:pout] * unitary_to_channel(gate)[:pinp]
        dens = lhs_state[:bout] * rhs_state[:binp]
        dens = split_axis(dens, :pout, (:bra, 2), (:ket, 2))
        dens = dens ./ tr(get_array(dens, :bra, :ket))
        push!(dynamics, dens)
    end
    dynamics
end

function dens_distance(lhs::Node, rhs::Node)
    lhs_arr = get_array(lhs, :bra, :ket)
    rhs_arr = get_array(rhs, :bra, :ket)
    trace_dist(lhs_arr, rhs_arr)
end

function prediction_distance(lhs::SketchyIM{A}, rhs::SketchyIM{A}, rng, sample_size::Integer) where {A<:AbstractArray}
    dist = zero(eltype(A))
    time_steps = get_time_steps_number(lhs)
    for _ in 1:sample_size
        dens = random_pure_dens(A, rng, 2, :bra, :ket)
        gates = [random_unitary(A, rng, 2, :pout, :pinp) for _ in 1:time_steps]
        lhs_dynamics = single_im_dynamics(lhs, dens, gates)
        rhs_dynamics = single_im_dynamics(rhs, dens, gates)
        for (lhs_dens, rhs_dens) in zip(lhs_dynamics, rhs_dynamics)
            dist += dens_distance(lhs_dens, rhs_dens)
        end
    end
    dist / (sample_size * time_steps)
end

function _bwd_contraction(
    state::Node,
    equation::Equation,
    time_step::Integer,
    ims::Dict{IMID, SketchyIM{A}},
    kernels::Dict{KernelID, Vector{N}} where {N<:Node},
    one_qubit_gates::Vector{<:Node},
    sketch_im::SketchyIM,
) where {A<:AbstractArray}
    is_kernel_passed = false
    for id in reverse(equation)
        if isa(id, IMID)
            ker = if length(ims[id].kernels) < time_step
                @assert !is_kernel_passed
                perfect_dissipator_kernel(A)
            else
                ims[id].kernels[time_step]
            end
            state = ker[:pout, :bout] * state[:inp, id.time_position]
            change_id!(state, :binp, id.time_position)
            change_id!(state, :pinp, :inp)
        else
            is_kernel_passed = true
            sketch_ker = sketch_im.kernels[time_step]
            state = sketch_ker[:bout] * state[:sketch_inp]
            change_id!(state, :binp, :sketch_inp)
            ker = kernels[id][time_step]
            state = ker[:second_inp, :second_out, :first_out] * state[:pinp, :pout, :inp]
            change_id!(state, :first_inp, :inp)
        end
    end
    state = one_qubit_gates[time_step][:pout] * state[:inp]
    change_id!(state, :pinp, :inp)
    n = norm(state)
    state ./ n
    state
end

# indices(sketch) = (im_ids..., :inp, :sketch_inp)
function _build_sketches(
    equation::Equation,
    ims::Dict{IMID, SketchyIM{A}},
    kernels::Dict{KernelID, Vector{N}} where {N<:Node},
    one_qubit_gates::Vector{<:Node},
    rank::Integer,
    iter_num::Int,
    rng,
) where {A<:AbstractArray}
    axes = map(x -> x.time_position, filter(x -> isa(x, IMID), equation))
    new_im = _random_im(SketchyIM{A}, rng, rank, iter_num)
    rhs_states = []
    rhs_state = split_axis(
        merge_axes(identity_from_array_type(A, 2, :bra, :ket), :out, :bra, :ket),
        :out,
        (:inp, 4),
        (:sketch_inp, 1),
        map(x -> (x, 1), axes)...,
    )
    push!(rhs_states, rhs_state)
    for ts in iter_num:-1:2
        rhs_state = _bwd_contraction(
            rhs_state, equation, ts, ims, kernels, one_qubit_gates, new_im,
        )
        push!(rhs_states, rhs_state)
    end
    reverse!(rhs_states)
    rhs_states
end

# indices(msg) = (im_ids..., :binp, :out)
function _apply_msg(
    msg::Node,
    equation,
    time_step::Integer,
    one_qubit_gates::Vector{<:Node},
    ims::Dict{IMID, SketchyIM{A}},
    kernels::Dict{KernelID, Vector{N}} where {N<:Node},
) where {A<:AbstractArray}
    is_kernel_passed = false
    msg = one_qubit_gates[time_step][:pinp] * msg[:out]
    change_id!(msg, :pout, :out)
    for id in equation
        if isa(id, IMID)
            ker = if length(ims[id].kernels) < time_step
                @assert is_kernel_passed
                perfect_dissipator_kernel(A)
            else
                ims[id].kernels[time_step]
            end
            msg = msg[:out, id.time_position] * ker[:pinp, :binp]
            change_id!(msg, :bout, id.time_position)
            change_id!(msg, :pout, :out)
        else
            is_kernel_passed = true
            ker = kernels[id][time_step]
            msg = msg[:out] * ker[:first_inp]
            change_id!(msg, :first_out, :out)
        end
    end
    change_id!(msg, :second_inp, :pinp)
    change_id!(msg, :second_out, :pout)
    msg
end

function contract(
    equation::Equation,
    ims::Dict{IMID, SketchyIM{A}},
    kernels::Dict{KernelID, Vector{N}} where {N<:Node},
    one_qubit_gates::Vector{<:Node},
    initial_state::Node,
    rank_or_eps::Union{Integer, AbstractFloat, Nothing},
    iter_num::Int;
    kwargs...,
) where {A<:AbstractArray}
    if !isa(rank_or_eps, Integer)
        error("Only truncation by fixed rank is supported for this type of IM")
    end
    rng = get_or_default(kwargs, :rng, Random.default_rng())
    sketches = _build_sketches(equation, ims, kernels, one_qubit_gates, rank_or_eps, iter_num, rng)
    axes = map(x -> x.time_position, filter(x -> isa(x, IMID), equation))
    im_bonds = map(x -> (x, 1), axes)
    msg = split_axis(
        initial_state,
        :pout,
        (:binp, 1),
        (:out, 4),
        im_bonds...,
    )
    new_kernels = Node{A}[]
    for ts in 1:iter_num
        msg_ker = _apply_msg(
            msg,
            equation,
            ts,
            one_qubit_gates,
            ims,
            kernels,
        )
        new_ker, _ = qr((msg_ker[:out, axes...] * sketches[ts][:inp, axes...])[:sketch_inp], :bout)
        push!(new_kernels, new_ker)
        if ts != iter_num
            conj_new_ker = conj.(new_ker)
            msg = conj_new_ker[:binp, :pinp, :pout] * msg_ker[:binp, :pinp, :pout]
            change_id!(msg, :bout, :binp)
        end
    end
    SketchyIM(new_kernels), nothing
end

function im_distance(lhs::SketchyIM{A}, rhs::SketchyIM{A}; kwargs...) where {T<:Number, A<:AbstractArray{T}}
    rng = get_or_default(kwargs, :rng, Random.default_rng())
    sample_size = get_or_default(kwargs, :sample_size, 10)
    dist = prediction_distance(lhs, rhs, rng, sample_size)
    @assert abs(imag(dist)) < 1e-12
    real(dist)
end

function get_bond_dimensions(im::SketchyIM)
    bond_dims_vec = Int[]
    shp = shape(im.kernels[1])
    push!(bond_dims_vec, shp[:binp])
    for ker in im.kernels
        shp = shape(ker)
        push!(bond_dims_vec, shp[:bout])
    end
    bond_dims_vec
end

function _fwd_evolution(
    state::Node,
    equation::Equation,
    time_step::Integer,
    one_qubit_gates::Vector{<:Node},
    ims::Dict{IMID, <:SketchyIM},
)
    state = one_qubit_gates[time_step][:pinp] * state[:pout]
    for (i, id) in enumerate(equation)
        @assert isa(id, IMID)
        ker = ims[id].kernels[time_step]
        state = ker[:pinp, :binp] * state[:pout, i]
        change_id!(state, :bout, i)
    end
    n = norm(state)
    state ./ n
end

function _bwd_evolution(
    state::Node,
    equation::Equation,
    time_step::Integer,
    ims::Dict{IMID, <:SketchyIM},
)
    eqs_number = length(equation)
    for (i, id) in zip(eqs_number:-1:1, reverse(equation))
        @assert isa(id, IMID)
        ker = ims[id].kernels[time_step]
        ker = trace_out(ker, :pinp)
        ker = trace_out(ker, :pout)
        state = ker[:bout] * state[i]
        change_id!(state, :binp, i)
    end
    n = norm(state)
    state ./ n
end

function simulate_dynamics(
    node_id::Integer,
    equations::Equations,
    ims::Dict{IMID, SketchyIM{A}},
    initial_state::Union{AbstractArray, Nothing} = nothing,
) where {T<:Number, A<:AbstractArray{T}}
    initial_state = isnothing(initial_state) ? equations.initial_states[node_id] : Node(reshape(initial_state, :), :pout)  
    equation = equations.marginal_eqs[node_id]
    eqs_num = length(equation)
    axes = 1:eqs_num
    time_steps = get_time_steps_number(ims)
    rhs_states = []
    rhs_state = split_axis(
        merge_axes(identity_from_array_type(A, 1, :bra, :ket), :out, :bra, :ket),
        :out,
        map(x -> (x, 1), axes)...,
    )
    for time_step in reverse(1:time_steps)
        rhs_state = _bwd_evolution(rhs_state, equation, time_step, ims)
        push!(rhs_states, rhs_state)
    end
    reverse!(rhs_states)
    lhs_state = split_axis(
        initial_state,
        :pout,
        (:pout, 4),
        map(x -> (x, 1), axes)...,
    )
    dynamics = AbstractArray{T}[]
    for (time_step, rhs_state) in enumerate(rhs_states)
        dens = _get_dens(lhs_state, rhs_state)
        push!(dynamics, dens)
        lhs_state = _fwd_evolution(lhs_state, equation, time_step, equations.one_qubit_gates[node_id], ims)
    end
    rhs_state = split_axis(
        merge_axes(identity_from_array_type(A, 1, :bra, :ket), :out, :bra, :ket),
        :out,
        map(x -> (x, 1), axes)...,
    )
    dens = _get_dens(lhs_state, rhs_state)
    push!(dynamics, dens)
    dynamics
end
