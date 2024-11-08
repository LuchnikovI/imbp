function no_repetition_check(idxs::Tuple{Vararg{T}}) where {T<:Integer}
    visited = Set{T}()
    for idx in idxs
        @assert !(idx in visited)
        push!(visited, idx)
    end
    visited
end
no_repetition_check(idxs::Integer...) = no_repetition_check(idxs)

function no_overlap_check(idxs1::Tuple{Vararg{Integer}}, idxs2::Tuple{Vararg{Integer}})
    visited = no_repetition_check(idxs2)
    for idx in idxs1
        @assert !(idx in visited)
        push!(visited, idx) 
    end
end

function bounds_check(max_val::Integer, idxs::Tuple{Vararg{Integer}})
    for idx in idxs
        @assert idx > 0
        @assert max_val >= idx
    end
end
bounds_check(max_val::Integer, idxs::Integer...) = bounds_check(max_val, idxs)

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

function tensordot(t1::AbstractArray, t2::AbstractArray, axes1::Tuple{Vararg{Integer}}, axes2::Tuple{Vararg{Integer}})
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

struct IndexID
    tensor_id::Int
    index_pos::Int
end

function evolve_by_tensor_perm_order(
    state_size::Integer,
    tensor_size::Integer,
    state_axes::Tuple{Vararg{Integer}},
    tensor_output_axes::Tuple{Vararg{Integer}},
)
    bounds_check(state_size, state_axes)
    bounds_check(tensor_size, tensor_output_axes)
    @assert length(state_axes) == length(tensor_output_axes)
    @assert tensor_size == 2 * length(tensor_output_axes)
    order_after_tensordot = (
        map(x -> IndexID(1, x), filter(x -> !(x in state_axes), 1:state_size))...,
        map(x -> IndexID(2, x), sort([i for i in tensor_output_axes]))...,
        0
    )
    correct_order = (
        map(x -> begin
            pos = findfirst(y -> y == x, state_axes)
            if pos === nothing
                IndexID(1, x)
            else
                IndexID(2, tensor_output_axes[pos])
            end
        end, 1:state_size)
    )
    perm_order = Int[]
    for idx in correct_order
        pos = findfirst(x -> x == idx, order_after_tensordot)
        if pos === nothing
            error("Bug in the code")
        else
            push!(perm_order, pos)
        end
    end
    perm_order
end

function evolve_by_tensor(
    state::AbstractArray,
    tensor::AbstractArray,
    state_axes::Tuple{Vararg{Integer}},
    tensor_input_axes::Tuple{Vararg{Integer}},
    tensor_output_axes::Tuple{Vararg{Integer}},
)
    tensor_size = length(size(tensor))
    state_size = length(size(state))
    no_overlap_check(tensor_input_axes, tensor_output_axes)
    bounds_check(tensor_size, tensor_input_axes)
    @assert length(tensor_input_axes) == length(tensor_output_axes)
    new_indices_order = evolve_by_tensor_perm_order(state_size, tensor_size, state_axes, tensor_output_axes)
    new_state = tensordot(state, tensor, state_axes, tensor_input_axes)
    permutedims(new_state, new_indices_order)
end

function apply_to_position(dst::AbstractArray, m::AbstractMatrix, pos::Integer)
    evolve_by_tensor(dst, m, (pos,), (2,), (1,))
end

function _hs_dim_from_dens_dim(dens_dim::Integer)
    sqrt_state_dim = round(Int, sqrt(dens_dim))
    @assert sqrt_state_dim * sqrt_state_dim == dens_dim
    sqrt_state_dim
end
