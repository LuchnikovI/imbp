abstract type AbstractIM end

# Abstract IM interface
get_perfect_dissipator_im(
    ::Type{<:AbstractIM},
    time_steps_number::Integer,
) = error("Not Yet Implemented")

contract(
    equation::Equation,
    ims::Dict{IMID, <:AbstractIM},
    kernels::Dict{KernelID, <:AbstractArray},
    one_qubit_gate::AbstractArray,
    initial_state::AbstractArray,
    rank_or_eps::Union{Integer, AbstractFloat, Nothing};
    kwargs...,
) = error("Not Yet Implemented")

log_fidelity(lhs::AbstractIM, rhs::AbstractIM) = error("Not Yet Implemented")

get_bond_dimensions(im::AbstractIM) = error("Not Yet Implemented")

get_time_steps_number(im::AbstractIM) = error("Not Yet Implemented")

simulate_dynamics(
    node_id::Int,
    equations::Equations,
    ims::Dict{IMID, <:AbstractIM},
    one_qubit_gate::AbstractArray,
    initial_state::AbstractArray,
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
