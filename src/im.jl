function trace_out(node::Node{A}, id) where {A<:AbstractArray}
    iden = identity_from_array_type(A, 2, :bra, :ket)
    iden = merge_axes(iden, id, :bra, :ket)
    node[id] * iden[id]
end

function _check_dims_consistency(kernels::Vector{<:Node})
    kernels_number = length(kernels)
    first_ker_shape = shape(kernels[1])
    if first_ker_shape[:pinp] != first_ker_shape[:pout]
        error("Input and output dimensions of the kernel #1 in the IM are not equal")
    end
    if first_ker_shape[:binp] != 1
        error("Input bond dimension of the IM is not equal to 1")
    else
        for (pos, kers) in enumerate(zip(kernels[1:(kernels_number - 1)], kernels[2:kernels_number]))
            lhs_ker_shape = shape(kers[1])
            rhs_ker_shape = shape(kers[2])
            if lhs_ker_shape[:bout] != rhs_ker_shape[:binp]
                error("Mismatch of bond dimension between $pos and $(pos + 1) kernals")
            end
            if lhs_ker_shape[:pout] != rhs_ker_shape[:pinp]
                error("Output dimension of the kernel #$pos does not match the input dimensions of the kernel #$(pos + 1)")
            end
            if rhs_ker_shape[:pinp] != rhs_ker_shape[:pout]
                error("Input and output dimensions of the kernel #$(pos + 1) in the IM are not equal")
            end
        end
        if shape(kernels[kernels_number])[:bout] != 1
            error("Output bond dimension of the IM is not equal to 1")
        end
    end
end

function _push_right(msg::Node, ker::Node)
    msg_ker = msg[:bout] * ker[:binp]
    new_ker, new_msg = qr(msg_ker[:bout], :tmp)
    change_id!(new_ker, :tmp, :bout)
    change_id!(new_msg, :tmp, :binp)
    n = norm(new_msg)
    new_ker, new_msg ./ n
end

function _push_left_truncate(
    ker::Node,
    msg::Node,
    rank_or_eps::Union{Integer, AbstractFloat, Nothing},
)
    ker_msg = ker[:bout] * msg[:binp]
    full_norm = norm(ker_msg)
    u, s, new_ker = svd(ker_msg[:pinp, :pout, :bout], :tmp, rank_or_eps)
    err = sqrt(abs(full_norm^2 - norm(s)^2)) / full_norm
    new_msg = u .* s
    change_id!(new_ker, :tmp, :binp)
    change_id!(new_msg, :tmp, :bout)
    n = norm(new_msg)
    new_msg ./ n, new_ker, err
end

function _log_norm!(node::Node)
    #TODO: fix SimpleTN api in order not ot access the private field here
    arr = node.arr
    n = norm(arr)
    arr[:] /= n
    log(n)
end

mutable struct IM{A<:AbstractArray} <: AbstractIM
    kernels::Vector{Node{A}}
    function IM(kernels::Vector{Node{A}}) where {A<:AbstractArray}
        _check_dims_consistency(kernels)
        new{A}(kernels)
    end
end

array_type(::Type{IM{Node{A}}}) where {A<:AbstractArray} = A
get_time_steps_number(im::IM) = length(im.kernels)

function _set_to_left_canonical!(im::IM{A}) where {A<:AbstractArray}
    msg = identity_from_array_type(A, 1, :bout, :binp)
    for (pos, ker) in enumerate(im.kernels)
        im.kernels[pos], msg = _push_right(msg, ker)
    end
    _check_dims_consistency(im.kernels)
end

function _truncate_left_canonical!(
    im::IM{A},
    rank_or_eps::Union{Integer, AbstractFloat, Nothing},
) where {A<:AbstractArray}
    msg = identity_from_array_type(A, 1, :bout, :binp)
    ker_num = get_time_steps_number(im)
    err = zero(real(eltype(A)))
    for (pos, ker) in enumerate(reverse(im.kernels))
        msg, im.kernels[ker_num + 1 - pos], new_err = _push_left_truncate(ker, msg, rank_or_eps)
        err = max(err, new_err)
    end
    _check_dims_consistency(im.kernels)
    err
end

function truncate!(im::IM, rank_or_eps::Union{Integer, AbstractFloat, Nothing})
    _set_to_left_canonical!(im)
    _truncate_left_canonical!(im, rank_or_eps)
end

function perfect_dissipator_kernel(::Type{A}) where {A<:AbstractArray}
    inp_iden = split_axis(
        merge_axes(
            identity_from_array_type(A, 2, :ket_inp, :bra_inp),
            :inp, :ket_inp, :bra_inp,
        ),
        :inp, (:pinp, 4), (:binp, 1))
    out_iden = split_axis(
        merge_axes(
            identity_from_array_type(A, 2, :ket_out, :bra_out),
            :out, :ket_out, :bra_out,
        ),
        :out, (:pout, 4), (:bout, 1))
    (inp_iden[] * out_iden[]) ./ 2
end

function _random_im(
    ::Type{IM{A}},
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
    IM(kernels)
end

function get_perfect_dissipator_im(
    ::Type{IM{A}},
    time_steps_number::Integer,
) where {A<:AbstractArray}
    im = IM(Node{A}[perfect_dissipator_kernel(A) for _ in 1:time_steps_number])
    im
end

function _build_ith_kernel(
    equation::Equation,
    one_qubit_gate::Node,
    initial_state::Node,
    time_step::Integer,
    ims::Dict{IMID, IM{A}},
    kernels::Dict{KernelID, <:Node},
) where {A<:AbstractArray}
    time_steps = get_time_steps_number(ims)
    msg = if time_step > 1
        split_axis(
            split_axis(
                identity_from_array_type(A, 4, :out, :inp),
                :out, (:bout_agr, 1), (:pout, 4), (:new_pout, 1),
            ),
            :inp, (:binp_agr, 1), (:pinp, 4), (:new_pinp, 1),
        )
    else
        split_axis(initial_state, :pout, (:binp_agr, 1), (:pinp, 1), (:new_pinp, 1), (:bout_agr, 1), (:pout, 4), (:new_pout, 1))
    end
    msg = msg[:pout] * one_qubit_gate[:pinp]
    for elem in equation
        if isa(elem, IMID)
            ker = ims[elem].kernels[time_step]
            msg = msg[:pout] * ker[:pinp]
            msg = merge_axes(msg, :bout_agr, :bout_agr, :bout)
            msg = merge_axes(msg, :binp_agr, :binp_agr, :binp)
        else
            ker = kernels[elem]
            msg = msg[:pout] * ker[:first_inp]
            msg = merge_axes(msg, :new_pinp, :new_pinp, :second_inp)
            msg = merge_axes(msg, :new_pout, :new_pout, :second_out)
            change_id!(msg, :first_out, :pout)
        end
    end
    msg = merge_axes(msg, :bout, :bout_agr, :pout)
    msg = merge_axes(msg, :binp, :binp_agr, :pinp)
    change_id!(msg, :new_pinp, :pinp)
    change_id!(msg, :new_pout, :pout)
    if time_step == time_steps
        msg = split_axis(msg, :bout, (:bra, 2), (:ket, 2))
        tr = identity_from_array_type(A, 2, :bra, :ket)
        tr = split_axis(tr, :bra, (:bra, 2), (:bout, 1))
        msg[:bra, :ket] * tr[:bra, :ket]
    else
        msg
    end
end

function contract(
    equation::Equation,
    ims::Dict{IMID, IM{A}},
    kernels::Dict{KernelID, <:Node},
    one_qubit_gate::Node,
    initial_state::Node,
    rank_or_eps::Union{Integer, AbstractFloat},
) where {A<:AbstractArray}
    time_steps = get_time_steps_number(ims)
    new_kernels = Node{A}[]
    for i in 1:time_steps
        push!(new_kernels, _build_ith_kernel(equation, one_qubit_gate, initial_state, i, ims, kernels))
    end
    new_im = IM(new_kernels)
    trunc_err = truncate!(new_im, rank_or_eps)
    new_im, trunc_err
end

function log_fidelity(lhs::IM{A}, rhs::IM{A}) where {T<:Number, A<:AbstractArray{T}}
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

function get_bond_dimensions(im::IM)
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
    ims::Dict{IMID, <:IM},
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
    ims::Dict{IMID, <:IM},
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

function _get_dens(lhs_state::Node, rhs_state::Node)
    # TODO: add dimension function to the public api of SimpleTN to avoid referencing private field below
    dim = length(size(rhs_state.arr))
    axes = 1:dim
    dens = lhs_state[axes...] * rhs_state[axes...]
    dens = get_array(dens, :pout)
    sqrt_dim = round(Int, sqrt(length(dens)))
    @assert sqrt_dim * sqrt_dim == length(dens)
    dens = reshape(dens, (sqrt_dim, sqrt_dim))
    dens /= tr(dens)
    dens
end

function simulate_dynamics(
    node_id::Integer,
    equations::Equations,
    ims::Dict{IMID, IM{A}},
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
