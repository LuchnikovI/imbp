function _check_dims_consistency(kernels::Vector{<:AbstractArray})
    kernels_number = length(kernels)
    first_ker_shape = size(kernels[1])
    if first_ker_shape[2] != first_ker_shape[3]
        error("Input and output dimensions of the kernel #1 in the IM are not equal")
    end
    if first_ker_shape[1] != 1
        error("Input bond dimension of the IM is not equal to 1")
    else
        for (pos, kers) in enumerate(zip(kernels[1:(kernels_number - 1)], kernels[2:kernels_number]))
            lhs_ker_shape = size(kers[1])
            rhs_ker_shape = size(kers[2])
            if lhs_ker_shape[4] != rhs_ker_shape[1]
                error("Mismatch of bond dimension between $pos and $(pos + 1) kernals")
            end
            if lhs_ker_shape[3] != rhs_ker_shape[2]
                error("Output dimension of the kernel #$pos does not match the input dimensions of the kernel #$(pos + 1)")
            end
            if rhs_ker_shape[2] != rhs_ker_shape[3]
                error("Input and output dimensions of the kernel #$(pos + 1) in the IM are not equal")
            end
        end
        if size(kernels[kernels_number])[4] != 1
            error("Output bond dimension of the IM is not equal to 1")
        end
    end
end

function _push_right(msg::A, ker::AbstractArray{T}) where {T<:Number, A<:AbstractArray{T}}
    old_ker_shape = size(ker)
    left_bond = size(msg)[1]
    msgker = tensordot(msg, ker, 1)
    msgker_resh = reshape(msgker, (:, old_ker_shape[4]))
    fac = qr(msgker_resh)
    new_ker_resh = A(fac.Q)
    new_msg = A(fac.R)
    new_ker = reshape(new_ker_resh, (left_bond, old_ker_shape[2:3]..., :))
    normalize!(new_msg)
    new_ker, new_msg
end

function find_rank(lmbd::AbstractArray, eps::AbstractFloat)
    lmbd_norm = norm(lmbd)
    rank = length(lmbd) - sum(sqrt.(cumsum(map(l -> (real(l) / lmbd_norm)^2, reverse(lmbd)))) .< eps)
    rank
end

function _truncated_svd(ker::AbstractArray, rank::Integer)
    n, m = size(ker)
    rank = min(rank, n, m)
    fac = svd(ker)
    u, lmbd, vt = fac.U, fac.S, fac.Vt
    u[:, 1:rank], lmbd[1:rank], vt[1:rank, :]
end

function _truncated_svd(ker::AbstractArray, eps::AbstractFloat)
    fac = svd(ker)
    u, lmbd, vt = fac.U, fac.S, fac.Vt
    rank = find_rank(lmbd, eps)
    u[:, 1:rank], lmbd[1:rank], vt[1:rank, :]
end

function _push_left_truncate(
    ker::AbstractArray,
    msg::AbstractArray,
    rank_or_eps::Union{Integer, AbstractFloat},
)
    old_ker_shape = size(ker)
    right_bond = size(msg)[2]
    kermsg = tensordot(ker, msg, 1)
    kermsg_resh = reshape(kermsg, (old_ker_shape[1], :))
    @assert(size(kermsg_resh) == (old_ker_shape[1], 16 * right_bond))
    u, lmbd, vt = _truncated_svd(kermsg_resh, rank_or_eps)
    full_norm = norm(kermsg_resh)
    err = sqrt(abs(full_norm^2 - norm(lmbd)^2)) / full_norm
    new_msg = u .* reshape(lmbd, 1, :)
    normalize!(new_msg)
    new_ker = reshape(vt, (:, old_ker_shape[2:3]..., right_bond))
    new_msg, new_ker, err
end

function _log_norm!(a::AbstractArray)
    n = norm(a)
    a ./= n
    log(n)
end

mutable struct IM{A<:AbstractArray} <: AbstractIM
    kernels::Vector{A}
    function IM(kernels::Vector{A}) where {A<:AbstractArray}
        _check_dims_consistency(kernels)
        new{A}(kernels)
    end
end

get_time_steps_number(im::IM) = length(im.kernels)

function _set_to_left_canonical!(im::IM{A}) where {T<:Number, A<:AbstractArray{T}}
    msg = A(undef, 1, 1)
    fill!(msg, one(T))
    for (pos, ker) in enumerate(im.kernels)
        im.kernels[pos], msg = _push_right(msg, ker)
    end
    _check_dims_consistency(im.kernels)
end

function _set_to_right_canonical!(im::IM{A}) where {T<:Number, A<:AbstractArray{T}}
    msg = A(undef, 1, 1)
    fill!(msg, one(T))
    im_len = get_time_steps_number(im)
    for (pos, ker) in enumerate(reverse(im.kernels))
        new_ker, msg = _push_right(msg, permutedims(ker, (4, 3, 2, 1)))
        im.kernels[im_len - pos + 1] = permutedims(new_ker, (4, 3, 2, 1))
    end
    _check_dims_consistency(im.kernels)
end

function _truncate_left_canonical!(
    im::IM{A},
    rank_or_eps::Union{Integer, AbstractFloat},
) where {T<:Number, A<:AbstractArray{T}}
    msg = A(undef, 1, 1)
    fill!(msg, one(T))
    ker_num = get_time_steps_number(im)
    err = zero(real(T))
    for (pos, ker) in enumerate(reverse(im.kernels))
        msg, im.kernels[ker_num + 1 - pos], new_err = _push_left_truncate(ker, msg, rank_or_eps)
        err = max(err, new_err)
    end
    _check_dims_consistency(im.kernels)
    err
end

function _truncate_right_canonical!(
    im::IM{A},
    rank_or_eps::Union{Int, AbstractFloat},
) where {T<:Number, A<:AbstractArray{T}}
    msg = A(undef, 1, 1)
    fill!(msg, one(T))
    err = zero(real(T))
    for (pos, ker) in enumerate(im.kernels)
        msg, new_ker, new_err = _push_left_truncate(permutedims(ker, (4, 3, 2, 1)), msg, rank_or_eps)
        im.kernels[pos] = permutedims(new_ker, (4, 3, 2, 1))
        err = max(err, new_err)
    end
    _check_dims_consistency(im.kernels)
    err
end

function truncate!(im::IM, rank_or_eps::Union{Integer, AbstractFloat})
    _set_to_left_canonical!(im)
    _truncate_left_canonical!(im, rank_or_eps)
    #_set_to_right_canonical!(im)
    #_truncate_right_canonical!(im, rank_or_eps)
end

function perfect_dissipator_kernel(dispatch_arr::AbstractArray, hilbert_space_size::Integer)
    dens_matrix_size = hilbert_space_size * hilbert_space_size
    eye = get_similar_identity(dispatch_arr, 2)
    diss = permutedims(reshape(kron(eye, eye), (2, 2, 2, 2)), (1, 3, 2, 4))
    reshape(diss, (1, dens_matrix_size, dens_matrix_size, 1))
end

function _random_im(
    ::Type{T},
    rng,
    bond_dim::Integer,
    time_steps_number::Integer,
) where {T<:Number}
    kernels = Array{T}[]
    push!(kernels, randn(rng, T, (1, 4, 4, bond_dim)))
    for _ in 1:(time_steps_number - 2)
        push!(kernels, randn(rng, T, (bond_dim, 4, 4, bond_dim)))
    end
    push!(kernels, randn(rng, T, (bond_dim, 4, 4, 1)))
    IM(kernels)
end

function get_perfect_dissipator_im(
    ::Type{IM{A}},
    dispatch_arr::AbstractArray,
    time_steps_number::Integer,
) where {A<:AbstractArray}
    im = IM(A[perfect_dissipator_kernel(dispatch_arr, 2) for _ in 1:time_steps_number])
    truncate!(im, 1)
    im
end

function _update_kernel(
    ker::AbstractArray,
    kernels::Dict{KernelID, <:AbstractArray},
    ::Dict{IMID, <:IM},
    id::KernelID,
    ::Integer,
)
    rhs = kernels[id]
    ker_shape = size(ker)
    rhs_shape = size(rhs)
    aux = tensordot(ker, rhs, 5, 3)
    aux = permutedims(aux, (1, 2, 3, 8, 4, 7, 6, 5))
    #@tensor aux[i1, i2, i3, i4, o3, o4, o2, o1] := ker[i1, i2, i3, o3, oi, o1] * rhs[o2, o4, oi, i4]
    reshape(aux, (
        ker_shape[1],
        ker_shape[2],
        ker_shape[3] * rhs_shape[4],
        ker_shape[4] * rhs_shape[2],
        rhs_shape[1],
        ker_shape[6],
    ))
end

function _update_kernel(
    ker::AbstractArray,
    ::Dict{KernelID,<:AbstractArray},
    ims::Dict{IMID, <:IM},
    id::IMID,
    time_position::Integer,
)
    rhs = ims[id].kernels[time_position]
    ker_shape = size(ker)
    rhs_shape = size(rhs)
    #@tensor aux[i0, i1, i2, i3, o3, o2, o0, o1] := ker[i1, i2, i3, o3, oi, o1] * rhs[i0, oi, o2, o0]
    aux = tensordot(ker, rhs, 5, 2)
    aux = permutedims(aux, (6, 1, 2, 3, 4, 7, 8, 5))
    reshape(aux, (
        rhs_shape[1] * ker_shape[1],
        ker_shape[2],
        ker_shape[3],
        ker_shape[4],
        rhs_shape[3],
        rhs_shape[4] * ker_shape[6],
    ))
end

function _build_ith_kernel(
    equation::Equation,
    ims::Dict{IMID, <:IM},
    kernels::Dict{KernelID, <:AbstractArray},
    one_qubit_gate::AbstractArray,
    initial_state::AbstractArray,
    i::Integer,
)
    time_steps = get_time_steps_number(ims)
    state_dim = size(initial_state)[1]
    ker = if i == 1
        reshape(initial_state, (1, 1, 1, 1, state_dim, 1))
    else
        reshape(get_similar_identity(initial_state, state_dim), (1, state_dim, 1, 1, state_dim, 1))
    end
    #@tensor ker[i, j, k, l, m, n] := one_qubit_gate[m, o] * ker[i, j, k, l, o, n]
    #ker = tensordot(one_qubit_gate, ker, 2, 5)
    #ker = permutedims(ker, (2, 3, 4, 5, 1, 6))
    ker = apply_to_position(ker, one_qubit_gate, 5)
    for id in equation
        ker = _update_kernel(ker, kernels, ims, id, i)
    end
    if i == time_steps
        hs_dim = _hs_dim_from_dens_dim(state_dim)
        tr = reshape(get_similar_identity(ker, hs_dim), (1, state_dim))
        ker = apply_to_position(ker, tr, 5)
        #@tensor ker[i, j, k, l, m, n] := ker[i, j, k, l, q, n] * tr[q, m]
    end
    ker_shape = size(ker)
    ker = permutedims(ker, (1, 2, 3, 4, 6, 5))
    reshape(ker, (ker_shape[1] * ker_shape[2], ker_shape[3], ker_shape[4], :))
end

function contract(
    equation::Equation,
    ims::Dict{IMID, IM{A}},
    kernels::Dict{KernelID, <:AbstractArray},
    one_qubit_gate::AbstractArray,
    initial_state::AbstractArray,
    rank_or_eps::Union{Integer,AbstractFloat},
) where {A<:AbstractArray}
    time_steps = get_time_steps_number(ims)
    new_kernels = A[]
    for i in 1:time_steps
        push!(new_kernels, _build_ith_kernel(equation, ims, kernels, one_qubit_gate, initial_state, i))
    end
    new_im = IM(new_kernels)
    trunc_err = truncate!(new_im, rank_or_eps)
    new_im, trunc_err
end

function log_fidelity(lhs::IM{<:AbstractArray{T}}, rhs::IM{<:AbstractArray{T}}) where {T<:Number}
    msg = similar(first(lhs.kernels), 1, 1)
    ArrayInterface.allowed_setindex!(msg, one(T), 1, 1)
    log_fid = zero(T)
    for (ker_lhs, ker_rhs) in zip(lhs.kernels, rhs.kernels)
        ker_rhs = conj(ker_rhs)
        aux = tensordot(msg, ker_lhs, 1, 1)
        msg = tensordot(aux, ker_rhs, (1, 2, 3), (1, 2, 3))
        #@tensor begin
        #    aux[j, k, l, q] := msg[i, j] * ker_lhs[i, k, l, q]
        #    msg[q, p] := aux[j, k, l, q] * ker_rhs[j, k, l, p]
        #end
        log_fid += _log_norm!(msg)
    end
    2 * real(log_fid)
end

function get_bond_dimensions(im::IM)
    bond_dims_vec = Int[]
    push!(bond_dims_vec, size(im.kernels[1])[1])
    for ker in im.kernels
        push!(bond_dims_vec, size(ker)[4])
    end
    bond_dims_vec
end

function _fwd_evolution(
    state::AbstractArray,
    ker::AbstractArray,
)
    ker_shape = size(ker)
    state_shape = size(state)
    resh_state = reshape(state, (ker_shape[1], :, state_shape[2]))
    new_state = tensordot(resh_state, ker, (1, 3), (1, 2))
    new_state = permutedims(new_state, (1, 3, 2))
    #@tensor new_state[j, m, l] := resh_state[i, j, k] * ker[i, k, l, m]
    new_state = reshape(new_state, (:, ker_shape[3]))
    normalize!(new_state)
    new_state
end

function _fwd_evolution(
    state::AbstractArray,
    equation::Equation,
    time_step::Integer,
    one_qubit_gate::AbstractArray,
    ims::Dict{IMID, <:IM},
)
    #@tensor state[i, j] := state[i, k] * one_qubit_gate[j, k]
    state = state * transpose(one_qubit_gate)
    for id in equation
        @assert isa(id, IMID)
        ker = ims[id].kernels[time_step]
        state = _fwd_evolution(state, ker)
    end
    state
end

function _bwd_evolution(
    state::AbstractArray,
    ker::AbstractArray,
)
    ker_shape = size(ker)
    sqrt_inp_tr_dim = round(Int, sqrt(ker_shape[2]))
    @assert sqrt_inp_tr_dim * sqrt_inp_tr_dim == ker_shape[2]
    sqrt_out_tr_dim = round(Int, sqrt(ker_shape[3]))
    @assert sqrt_out_tr_dim * sqrt_out_tr_dim == ker_shape[3]
    #resh_ker = reshape(
    #    ker,
    #    ker_shape[1],
    #    sqrt_inp_tr_dim,
    #    sqrt_inp_tr_dim,
    #    sqrt_out_tr_dim,
    #    sqrt_out_tr_dim,
    #    ker_shape[4],
    #) 
    #@tensor reduced_ker[i, j] := resh_ker[i, q, q, p, p, j]
    resh_state = reshape(state, (:, ker_shape[4]))
    A = typeof(resh_state)
    tr_inp = reshape(get_similar_identity(state, sqrt_inp_tr_dim), (1, :))
    tr_out = reshape(get_similar_identity(state, sqrt_out_tr_dim), (1, :))
    reduced_ker = apply_to_position(apply_to_position(ker, tr_inp, 2), tr_out, 3)[:, 1, 1, :]
    #@tensor new_state[i, k] := reduced_ker[i, j] * resh_state[k, j]
    new_state = reduced_ker * transpose(resh_state)
    new_state = reshape(new_state, (:,))
    normalize!(new_state)
    new_state
end

function _bwd_evolution(
    state::AbstractArray,
    equation::Equation,
    time_step::Integer,
    ims::Dict{IMID, <:IM},
)
    for id in reverse(equation)
        @assert isa(id, IMID)
        ker = ims[id].kernels[time_step]
        state = _bwd_evolution(state, ker)
    end
    state
end

function _get_dens(lhs_state::AbstractArray, rhs_state::AbstractArray)
    dens = transpose(lhs_state) * rhs_state
    #@tensor dens[j] := lhs_state[i, j] * rhs_state[i]
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
    equation = equations.marginal_eqs[node_id]
    one_qubit_gate = equations.one_qubit_gates[node_id]
    time_steps = get_time_steps_number(ims)
    rhs_state = A(undef, 1)
    rhs_state[1] = one(T)
    rhs_states = AbstractArray{T}[]
    for time_step in reverse(1:time_steps)
        rhs_state = _bwd_evolution(rhs_state, equation, time_step, ims)
        push!(rhs_states, rhs_state)
    end
    reverse!(rhs_states)
    lhs_state = reshape(initial_state, (1, :))
    dynamics = AbstractArray{T}[]
    for (time_step, rhs_state) in enumerate(rhs_states)
        push!(dynamics, _get_dens(lhs_state, rhs_state))
        lhs_state = _fwd_evolution(lhs_state, equation, time_step, one_qubit_gate, ims)
    end
    rhs_state = A(undef, 1)
    rhs_state[1] = one(T)
    push!(dynamics, _get_dens(lhs_state, rhs_state))
    dynamics
end
