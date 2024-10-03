using LinearAlgebra
using TensorOperations

function _check_dims_consistency(kernels::Vector{Array{N, 4}}) where {N}
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

function _push_right(msg::Array{N, 2}, ker::Array{N, 4}) where {N<:Number}
    old_ker_shape = size(ker)
    left_bond = size(msg)[1]
    @tensor msgker[i, k, l, q] := msg[i, j] * ker[j, k, l, q]
    msgker_resh = reshape(msgker, (:, old_ker_shape[4]))
    fac = qr(msgker_resh)
    new_ker_resh = Matrix(fac.Q)
    new_msg = Matrix(fac.R)
    new_ker = reshape(new_ker_resh, (left_bond, old_ker_shape[2:3]..., :))
    normalize!(new_msg)
    new_ker, new_msg
end

find_rank(lmbd::Array{N, 1}, rank::Int64) where {N<:Number} = min(size(lmbd)[1], rank)

function find_rank(lmbd::Array{N, 1}, eps::AbstractFloat) where {N<:Number}
    lmbd_norm = norm(lmbd)
    rank = length(lmbd) - sum(sqrt.(cumsum(map(l -> (real(l) / lmbd_norm)^2, reverse(lmbd)))) .< eps)
    rank
end

function _push_left_truncate(ker::Array{N, 4}, msg::Array{N, 2}, rank_or_eps::Union{Int64, AbstractFloat}) where {N<:Number}
    old_ker_shape = size(ker)
    right_bond = size(msg)[2]
    @tensor kermsg[i, k, l, q] := ker[i, k, l, p] * msg[p, q]
    kermsg_resh = reshape(kermsg, (old_ker_shape[1], :))
    @assert(size(kermsg_resh) == (old_ker_shape[1], 16 * right_bond))
    fac = svd(kermsg_resh)
    u, lmbd, vt = fac.U, fac.S, fac.Vt
    rank = find_rank(lmbd, rank_or_eps)
    err = sqrt(reduce((x, y) -> real(x)^2 + real(y)^2, lmbd[(rank + 1):length(lmbd)]; init=zero(real(N)))) / norm(lmbd)
    lmbd = Diagonal(lmbd[1:rank])
    u = u[:, 1:rank]
    vt = vt[1:rank, :]
    new_msg = u * lmbd
    normalize!(new_msg)
    new_ker = reshape(vt, (:, old_ker_shape[2:3]..., right_bond))
    new_msg, new_ker, err
end

function _log_norm!(a::AbstractArray)
    n = norm(a)
    a ./= n
    log(n)
end

mutable struct IM{N<:Number} <: AbstractIM
    kernels::Vector{Array{N, 4}}
    function IM(kernels::Vector{Array{N, 4}}) where {N<:Number}
        _check_dims_consistency(kernels)
        new{N}(kernels)
    end
end

get_time_steps_number(im::IM) = length(im.kernels)

function perfect_dissipator_kernel(::Type{N}, hilbert_space_size::Int) where {N<:Number}
    dens_matrix_size = hilbert_space_size * hilbert_space_size
    eye = Matrix{N}(I, hilbert_space_size, hilbert_space_size)
    @tensor diss[i, j, k, l] := eye[i, j] * eye[k, l]
    reshape(diss, (1, dens_matrix_size, dens_matrix_size, 1))
end

function _random_im(::Type{N}, rng, bond_dim::Int64, time_steps_number::Int64) where {N<:Number}
    kernels = Array{N, 4}[]
    push!(kernels, randn(rng, N, (1, 4, 4, bond_dim)))
    for _ in 1:(time_steps_number - 2)
        push!(kernels, randn(rng, N, (bond_dim, 4, 4, bond_dim)))
    end
    push!(kernels, randn(rng, N, (bond_dim, 4, 4, 1)))
    IM(kernels)
end

function get_perfect_dissipator_im(::Type{IM{N}}, time_steps_number::Int64) where {N<:Number}
    IM([perfect_dissipator_kernel(N, 2) for _ in 1:time_steps_number])
end

function _set_to_left_canonical!(im::IM{N}) where {N<:Number}
    msg = ones(N, 1, 1)
    for (pos, ker) in enumerate(im.kernels)
        im.kernels[pos], msg = _push_right(msg, ker)
    end
    _check_dims_consistency(im.kernels)
end

function _truncate_left_canonical!(im::IM{N}, rank_or_eps::Union{Int64, AbstractFloat}) where {N<:Number}
    msg = ones(N, 1, 1)
    ker_num = get_time_steps_number(im)
    err = zero(real(N))
    for (pos, ker) in enumerate(reverse(im.kernels))
        msg, im.kernels[ker_num + 1 - pos], new_err = _push_left_truncate(ker, msg, rank_or_eps)
        err = max(err, new_err)
    end
    _check_dims_consistency(im.kernels)
    err
end

function truncate!(im::IM, rank_or_eps::Union{Int64, AbstractFloat})
    _set_to_left_canonical!(im)
    _truncate_left_canonical!(im, rank_or_eps)
end

function _hs_dim_from_dens_dim(dens_dim::Int64)
    sqrt_state_dim = round(Int64, sqrt(dens_dim))
    @assert sqrt_state_dim * sqrt_state_dim == dens_dim
    sqrt_state_dim
end

function _update_kernel(
    ker::Array{N, 6},
    kernels::Dict{KernelID, Array{N, 4}},
    ::Dict{IMID, IM{N}},
    id::KernelID,
    ::Int,
) where {N<:Number}
    rhs = kernels[id]
    ker_shape = size(ker)
    rhs_shape = size(rhs)
    @tensor aux[i1, i2, i3, i4, o3, o4, o2, o1] := ker[i1, i2, i3, o3, oi, o1] * rhs[o2, o4, oi, i4]
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
    ker::Array{N, 6},
    ::Dict{KernelID, Array{N, 4}},
    ims::Dict{IMID, IM{N}},
    id::IMID,
    time_position::Int,
) where {N<:Number}
    rhs = ims[id].kernels[time_position]
    ker_shape = size(ker)
    rhs_shape = size(rhs)
    @tensor aux[i0, i1, i2, i3, o3, o2, o0, o1] := ker[i1, i2, i3, o3, oi, o1] * rhs[i0, oi, o2, o0]
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
    ims::Dict{IMID, IM{N}},
    kernels::Dict{KernelID, Array{N, 4}},
    initial_state::Vector{N},
    i::Int64,
) where {N<:Number}
    time_steps = get_time_steps_number(ims)
    state_dim = size(initial_state)[1]
    ker = if i == 1
        reshape(initial_state, (1, 1, 1, 1, state_dim, 1))
    else
        reshape(Matrix{N}(I, state_dim, state_dim), (1, state_dim, 1, 1, state_dim, 1))
    end
    for id in equation
        ker = _update_kernel(ker, kernels, ims, id, i)
    end
    if i == time_steps
        hs_dim = _hs_dim_from_dens_dim(state_dim)
        tr = reshape(Matrix{N}(I, hs_dim, hs_dim), (state_dim, 1))
        @tensor ker[i, j, k, l, m, n] := ker[i, j, k, l, q, n] * tr[q, m]
    end
    ker_shape = size(ker)
    ker = permutedims(ker, (1, 2, 3, 4, 6, 5))
    reshape(ker, (ker_shape[1] * ker_shape[2], ker_shape[3], ker_shape[4], :))
end

function contract(
    equation::Equation,
    ims::Dict{IMID, IM{N}},
    kernels::Dict{KernelID, Array{N, 4}},
    initial_state::Vector{N},
) where {N<:Number}
    time_steps = get_time_steps_number(ims)
    new_kernels = Array{N, 4}[]
    for i in 1:time_steps
        push!(new_kernels, _build_ith_kernel(equation, ims, kernels, initial_state, i))
    end
    IM(new_kernels)
end

function log_fidelity(lhs::IM{N}, rhs::IM{N}) where {N<:Number}
    msg = ones(N, 1, 1)
    log_fid = zero(N)
    for (ker_lhs, ker_rhs) in zip(lhs.kernels, rhs.kernels)
        ker_rhs = conj(ker_rhs)
        @tensor begin
            aux[j, k, l, q] := msg[i, j] * ker_lhs[i, k, l, q]
            msg[q, p] := aux[j, k, l, q] * ker_rhs[j, k, l, p]
        end
        log_fid += _log_norm!(msg)
    end
    2 * real(log_fid)
end

function get_bond_dimensions(im::IM)
    bond_dims_vec = Int64[]
    push!(bond_dims_vec, size(im.kernels[1])[1])
    for ker in im.kernels
        push!(bond_dims_vec, size(ker)[4])
    end
    bond_dims_vec
end

function simulate_dynamics(
    equation::Equation,
    ims::Dict{IMID, I},
    initial_state::Array{N, 2},
) where {N<:Number}
    error("Not Yet Implemented")
end