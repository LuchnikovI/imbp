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

function get_perfect_dissipator_im(
    ::Type{SketchyIM{A}},
    time_steps_number::Integer,
) where {A<:AbstractArray}
    SketchyIM(Node{A}[perfect_dissipator_kernel(A) for _ in 1:time_steps_number])
end

function _bwd_contraction(
    state::Node,
    equation::Equation,
    time_step::Integer,
    ims::Dict{IMID, <:SketchyIM},
    kernels::Dict{KernelID, <:Node},
    one_qubit_gate::Node,
    sketch_im::SketchyIM,
)
    for id in reverse(equation)
        if isa(id, IMID)
            ker = ims[id].kernels[time_step]
            state = ker[:pout, :bout] * state[:inp, id.time_position]
            change_id!(state, :binp, id.time_position)
            change_id!(state, :pinp, :inp)
        else
            sketch_ker = sketch_im.kernels[time_step]
            state = sketch_ker[:bout] * state[:sketch_inp]
            change_id!(state, :binp, :sketch_inp)
            ker = kernels[id]
            state = ker[:second_inp, :second_out, :first_out] * state[:pinp, :pout, :inp]
            change_id!(state, :first_inp, :inp)
        end
    end
    state = one_qubit_gate[:pout] * state[:inp]
    change_id!(state, :pinp, :inp)
    n = norm(state)
    state ./ n
    state
end

# indices(sketch) = (im_ids..., :inp, :sketch_inp)
function _build_sketches(
    equation::Equation,
    ims::Dict{IMID, SketchyIM{A}},
    kernels::Dict{KernelID, <:Node},
    one_qubit_gate::Node,
    rank::Integer,
    rng,
) where {A<:AbstractArray}
    time_steps = get_time_steps_number(ims)
    axes = map(x -> x.time_position, filter(x -> isa(x, IMID), equation))
    new_im = _random_im(SketchyIM{A}, rng, rank, time_steps)
    rhs_states = []
    rhs_state = split_axis(
        merge_axes(identity_from_array_type(A, 2, :bra, :ket), :out, :bra, :ket),
        :out,
        (:inp, 4),
        (:sketch_inp, 1),
        map(x -> (x, 1), axes)...,
    )
    push!(rhs_states, rhs_state)
    for ts in time_steps:-1:2
        rhs_state = _bwd_contraction(
            rhs_state, equation, ts, ims, kernels, one_qubit_gate, new_im,
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
    one_qubit_gate::Node,
    ims::Dict{IMID, <:SketchyIM},
    kernels::Dict{KernelID, <:Node},
)
    msg = one_qubit_gate[:pinp] * msg[:out]
    change_id!(msg, :pout, :out)
    for id in equation
        if isa(id, IMID)
            ker = ims[id].kernels[time_step]
            msg = msg[:out, id.time_position] * ker[:pinp, :binp]
            change_id!(msg, :bout, id.time_position)
            change_id!(msg, :pout, :out)
        else
            ker = kernels[id]
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
    kernels::Dict{KernelID, <:Node},
    one_qubit_gate::Node,
    initial_state::Node,
    rank_or_eps::Union{Integer, AbstractFloat, Nothing};
    kwargs...,
) where {A<:AbstractArray}
    if !isa(rank_or_eps, Integer)
        error("Only truncation by fixed rank is supported for this type of IM")
    end
    time_steps = get_time_steps_number(ims)
    sketches = _build_sketches(equation, ims, kernels, one_qubit_gate, rank_or_eps, kwargs[:rng])
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
    for ts in 1:time_steps
        msg_ker = _apply_msg(
            msg,
            equation,
            ts,
            one_qubit_gate,
            ims,
            kernels,
        )
        new_ker, _ = qr((msg_ker[:out, axes...] * sketches[ts][:inp, axes...])[:sketch_inp], :bout)
        push!(new_kernels, new_ker)
        if ts != time_steps
            conj_new_ker = conj.(new_ker)
            msg = conj_new_ker[:binp, :pinp, :pout] * msg_ker[:binp, :pinp, :pout]
            change_id!(msg, :bout, :binp)
        end
    end
    SketchyIM(new_kernels), nothing
end

function log_fidelity(lhs::SketchyIM{A}, rhs::SketchyIM{A}) where {T<:Number, A<:AbstractArray{T}}
    msg = identity_from_array_type(A, 1, :bra, :ket)
    log_fid = zero(T)
    for (ker_lhs, ker_rhs) in zip(lhs.kernels, rhs.kernels)
        ker_rhs = conj.(ker_rhs)
        aux = msg[:bra] * ker_lhs[:binp]
        change_id!(aux, :bout, :bra)
        msg = aux[:ket, :pinp, :pout] * ker_rhs[:binp, :pinp, :pout]
        change_id!(msg, :bout, :ket)
        log_fid += _log_norm!(msg)
    end
    2 * real(log_fid)
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
    one_qubit_gate::Node,
    ims::Dict{IMID, <:SketchyIM},
)
    state = one_qubit_gate[:pinp] * state[:pout]
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
    initial_state::AbstractArray,
) where {T<:Number, A<:AbstractArray{T}}
    initial_state = Node(reshape(initial_state, :), :pout) 
    equation = equations.marginal_eqs[node_id]
    one_qubit_gate = equations.one_qubit_gates[node_id]
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
        push!(dynamics, _get_dens(lhs_state, rhs_state))
        lhs_state = _fwd_evolution(lhs_state, equation, time_step, one_qubit_gate, ims)
    end
    rhs_state = split_axis(
        merge_axes(identity_from_array_type(A, 1, :bra, :ket), :out, :bra, :ket),
        :out,
        map(x -> (x, 1), axes)...,
    )
    push!(dynamics, _get_dens(lhs_state, rhs_state))
    dynamics
end
