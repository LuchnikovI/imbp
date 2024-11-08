abstract type ElementID end

struct KernelID <: ElementID
    time_position::Int
    is_forward::Bool  # Forwrad for moving from nodes[1] to nodes[2]
    nodes::Tuple{Int, Int}  # connected nodes by a kernel (gate)
end

struct IMID <: ElementID
    time_position::Int
    is_forward::Bool # dirrection of the influende matrix
    nodes::Tuple{Int, Int}  # connected nodes by an influence matrix
end

IMID(kernel_id::KernelID) = IMID(kernel_id.time_position, kernel_id.is_forward, kernel_id.nodes)

const Equation = Vector{ElementID}

struct Equations{
    N<:Number,
    TQG<:Node{<:AbstractArray{N, 4}},
    IS<:Node{<:AbstractVector{N}},
    OQG<:Node{<:AbstractMatrix{N}},
}
    self_consistency_eqs::Vector{Vector{Equation}}
    marginal_eqs::Vector{Equation}
    kernels::Dict{KernelID, TQG}
    one_qubit_gates::Vector{OQG}
    initial_states::Vector{IS}
end
