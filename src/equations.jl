abstract type ElementID end

struct KernelID <: ElementID
    time_position::Int
    is_forward::Bool  # Forwrad for moving from nodes[1] to nodes[2]
    nodes::Tuple{Int, Int}  # connected nodes by a kernel (gate)
end

function Base.show(io::IO, id::KernelID)
    (st, ed) = id.is_forward ? (id.nodes[1], id.nodes[2]) : (id.nodes[2], id.nodes[1])
    println(io, "Kernel(", st, " -- ", ed, "), time position: ", id.time_position)
end

struct IMID <: ElementID
    time_position::Int
    is_forward::Bool # dirrection of the influende matrix
    nodes::Tuple{Int, Int}  # connected nodes by an influence matrix
end

function Base.show(io::IO, id::ElementID)
    (st, ed) = id.is_forward ? (id.nodes[1], id.nodes[2]) : (id.nodes[2], id.nodes[1])
    print(io, "IM(", st, " -> ", ed, "), time position: ", id.time_position)
end

IMID(kernel_id::KernelID) = IMID(kernel_id.time_position, kernel_id.is_forward, kernel_id.nodes)

const Equation = Vector{ElementID}

struct Equations{
    N<:Number,
    TQG<:Node{<:AbstractArray{N, 4}},
    IS<:Node{<:AbstractVector{N}},
    OQG<:Node{<:AbstractMatrix{N}},
}
    self_consistency_eqs::Vector{Equation}
    marginal_eqs::Vector{Equation}
    kernels::Dict{KernelID, Vector{TQG}}
    one_qubit_gates::Vector{Vector{OQG}}
    initial_states::Vector{IS}
    time_steps::Int
end
