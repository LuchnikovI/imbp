using  ArrayInterface

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
    #TODO: fix density matrix checking
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

function tensordot(t1::AbstractArray, t2::AbstractArray, axes_number::Integer)
    shape1 = size(t1)
    indices_num1 = length(shape1)
    prefix1 = shape1[1:(indices_num1 - axes_number)]
    suffix1 = shape1[(indices_num1 - axes_number + 1):indices_num1]
    shape2 = size(t2)
    indices_num2 = length(shape2)
    prefix2 = shape2[1:axes_number]
    suffix2 = shape2[(axes_number + 1):indices_num2]
    @assert suffix1 == prefix2
    new_shape = (prefix1..., suffix2...)
    lhs = reshape(t1, prod(prefix1), prod(suffix1))
    rhs = reshape(t2, prod(prefix2), prod(suffix2))
    reshape(lhs * rhs, new_shape)
end

function tensordot(t1::AbstractArray, t2::AbstractArray, axes1::Tuple{Vararg{<:Integer}}, axes2::Tuple{Vararg{<:Integer}})
    @assert length(axes1) == length(axes2)
    new_order1 = (filter(x -> !(x in axes1), 1:length(size(t1)))..., axes1...)
    new_order2 = (axes2..., filter(x -> !(x in axes2), 1:length(size(t2)))...)
    tensordot(permutedims(t1, new_order1), permutedims(t2, new_order2), length(axes1))
end

function tensordot(t1::AbstractArray, t2::AbstractArray, axes1::Integer, axes2::Integer)
    new_order1 = (filter(x -> x != axes1, 1:length(size(t1)))..., axes1)
    new_order2 = (axes2, filter(x -> x != axes2, 1:length(size(t2)))...)
    tensordot(permutedims(t1, new_order1), permutedims(t2, new_order2), 1)
end

function apply_to_position(dst::AbstractArray, m::AbstractMatrix, pos::Integer)
    dot_result = tensordot(m, dst, 2, pos)
    len = length(size(dot_result))
    permutedims(dot_result, ((i for i in 2:pos)..., 1, (i for i in (pos + 1):len)...))
end

function _hs_dim_from_dens_dim(dens_dim::Integer)
    sqrt_state_dim = round(Int, sqrt(dens_dim))
    @assert sqrt_state_dim * sqrt_state_dim == dens_dim
    sqrt_state_dim
end
