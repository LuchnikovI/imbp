abstract type AbstractIM end

# Abstract IM interface
init_im(
    ::Type{<:AbstractIM},
) = error("Not Yet Implemented")

contract(
    equation::Equation,
    ims::Dict{IMID, <:AbstractIM},
    kernels::Dict{KernelID, Vector{N}} where {N<:Node},
    one_qubit_gate::Vector{<:Node},
    initial_state::Node,
    rank_or_eps::Union{Integer, AbstractFloat, Nothing};
    time_steps::Int,
    kwargs...,
) = error("Not Yet Implemented")

im_distance(lhs::AbstractIM, rhs::AbstractIM; kwargs...) = error("Not Yet Implemented")

get_bond_dimensions(im::AbstractIM) = error("Not Yet Implemented")

get_time_steps_number(im::AbstractIM) = error("Not Yet Implemented")

simulate_dynamics(
    node_id::Integer,
    equations::Equations,
    ims::Dict{IMID, <:AbstractIM},
    initial_state::Union{<:AbstractArray, Nothing},
) = error("Not Yet Implemented")

# ----------------------------------------------------------------------------------------------------------------------

check_two_time_steps(::Nothing, rhs::Integer) = rhs

check_two_time_steps(lhs::Integer, rhs::Integer) = (@assert(lhs == rhs); rhs)

function get_time_steps_number(ims::Dict{IMID, <:AbstractIM})
    time_steps::Union{Integer, Nothing} = nothing
    for (_, val) in ims
        time_steps = check_two_time_steps(time_steps, get_time_steps_number(val))
    end
    time_steps
end
