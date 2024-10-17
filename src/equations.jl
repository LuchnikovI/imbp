abstract type ElementID end

struct KernelID <: ElementID
    time_position::Int64
    is_forward::Bool  # Forwrad for moving from nodes[1] to nodes[2]
    nodes::Tuple{Int64, Int64}  # connected nodes by a kernel (gate)
end

struct IMID <: ElementID
    time_position::Int64
    is_forward::Bool # dirrection of the influende matrix
    nodes::Tuple{Int64, Int64}  # connected nodes by an influence matrix
end

IMID(kernel_id::KernelID) = IMID(kernel_id.time_position, kernel_id.is_forward, kernel_id.nodes)

const Equation = Vector{ElementID}

struct Equations{T<:Number}
    self_consistency_eqs::Vector{Vector{Equation}}
    marginal_eqs::Vector{Equation}
    kernels::Dict{KernelID, AbstractArray{T, 4}}
    one_qubit_gates::Vector{AbstractMatrix{T}}
    initial_states::Vector{AbstractVector{T}}
end
