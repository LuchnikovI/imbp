# only for small matrices
function get_similar_identity(arr::AbstractArray, size::Integer)
    @assert size > 0
    T = eltype(arr)
    m = similar(arr, size, size)
    for idx1 in 1:size
        for idx2 in 1:size
            val = idx1 == idx2 ? one(T) : zero(T)
            ArrayInterface.allowed_setindex!(m, val, idx1, idx2)
        end
    end
    m
end

function get_id_matrix(::Type{A}, size::Integer) where {N, A<:AbstractArray{<:Number, N}}
    @assert size > 0
    @assert N > 0
    m = reshape(A(undef, (size * size, (1 for _ in 1:(N-1))...)...), size, size)
    for i in 1:size
        for j in 1:size
            ArrayInterface.allowed_setindex!(m, i == j ? one(N) : zero(N), i, j)
        end
    end
    m
end

function partial_trace(arr::AbstractArray, pos1::Integer, pos2::Integer)
    # subsystems that are being traced out are small, so we can afford O(n^2)
    dim = length(size(arr))
    @assert pos1 > 0
    @assert pos2 > 0
    @assert pos1 <= dim
    @assert pos2 <= dim
    @assert pos1 != pos2
    size1 = size(arr)[pos1]
    size2 = size(arr)[pos2]
    @assert size1 == size2
    shape_id_broadcasted = Tuple(i != pos1 && i != pos2 ? 1 : size1 for i in 1:dim)
    id_matrix = get_similar_identity(arr, size1)
    sum(arr .* reshape(id_matrix, shape_id_broadcasted), dims=(pos1, pos2))
end

function dist_to_id(m::AbstractMatrix)
    T = real(eltype(m))
    dist = zero(T)
    shape = size(m)
    @assert shape[1] == shape[2]
    sz = shape[1]
    for idx1 in 1:sz
        for idx2 in 1:sz
            val = ArrayInterface.allowed_getindex(m, idx1, idx2)
            diff = idx1 == idx2 ? val - one(T) : val
            dist += real(sqrt(diff * conj(diff)))
        end
    end
    dist
end

function check_density(dens::AbstractArray)
    dens = Array(dens)
    dens_shape = size(dens)
    if dens_shape != (2, 2)
        error("Initial density matrix must be a matrix of shape (2, 2), got shape $dens_shape")
    end
    trace = tr(dens)
    un_unit_traceness = norm(trace - 1)
    inhermicity = norm(dens .- conj(transpose(dens)))
    evals = eigvals(dens)
    if un_unit_traceness > 1e-10
        @warn "Non-unit trace initial state" node_number inhermicity evals tr
    end
    if inhermicity > 1e-10
        @warn "Non-hermitian initial state"  node_number inhermicity evals tr
    end
    if !all(x -> real(x) + 1e-10 > 0, evals)
        @warn "Negative eigenvalues of initial state"  node_number inhermicity evals tr
    end
end

function check_channel(node1::Integer, node2::Integer, channel::AbstractArray)
    channel = Array(channel)
    choi = permutedims(reshape(channel, (4, 4, 4, 4)), (1, 3, 2, 4))
    trace = partial_trace(choi, 1, 3)[1, :, 1, :]
    un_identity_traceness = dist_to_id(trace)
    choi = reshape(choi, (16, 16))
    inhermicity = norm(choi .- conj(transpose(choi)))
    evals = eigvals(choi)
    if un_identity_traceness > 1e-10
        @warn "Non-identity trace of quantum channel" node1 node2 trace inhermicity evals
    end
    if inhermicity > 1e-10
        @warn "Non-hermitian choi matrix" node1 node2 trace inhermicity evals
    end
    if !all(x -> real(x) + 1e-10 > 0, evals)
        @warn "Negative eigenvalues of choi matrix" node1 node2 trace inhermicity evals
    end
end

function trace_dist(lhs::A, rhs::A) where {A<:AbstractArray{<:Number, 2}}
    fac = svd(lhs - rhs)
    0.5 * sum(fac.S)
end

function random_unitary(::Type{A}, rng, size::Integer, out_id, inp_id) where {A<:AbstractArray}
    m = randn(rng, eltype(A), size, size)
    fac = qr(m)
    q = typeof(m)(fac.Q)
    Node(q, out_id, inp_id)
end

function random_pure_dens(::Type{A}, rng, size::Integer, lhs_id, rhs_id) where {A<:AbstractArray}
    psi = randn(rng, eltype(A), size)
    normalize!(psi)
    Node(psi, lhs_id)[] * Node(conj(psi), rhs_id)[]
end
