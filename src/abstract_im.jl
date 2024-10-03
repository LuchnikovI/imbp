abstract type AbstractIM end

# Abstract IM interface
get_perfect_dissipator_im(::Type{I}, time_steps_number::Int64) where {I<:AbstractIM} = error("Not Yet Implemented")

truncate!(im::AbstractIM, rank_or_eps::Union{Int64, AbstractFloat}) = error("Not Yet Implemented")

contract(
    equation::Equation,
    ims::Dict{IMID, I},
    kernels::Dict{KernelID, A},
    initial_states::Vector{A},
) where {I<:AbstractIM, A<:AbstractArray} = error("Not Yet Implemented")

log_fidelity(lhs::AbstractIM, rhs::AbstractIM) = error("Not Yet Implemented")

get_bond_dimensions(im::AbstractIM) = error("Not Yet Implemented")

get_time_steps_number(im::AbstractIM) = error("Not Yet Implemented")

simulate_dynamics(
    equation::Equation,
    ims::Dict{IMID, I},
    initial_state::Array{N, 2},
) where {N<:Number, I<:AbstractIM} = error("Not Yet Implemented")

# ----------------------------------------------------------------------------------------------------------------------

check_two_time_steps(::Nothing, rhs::Int64) = rhs

check_two_time_steps(lhs::Int64, rhs::Int64) = (@assert(lhs == rhs); rhs)

function get_time_steps_number(ims::Dict{IMID, I}) where {I<:AbstractIM}
    time_steps::Union{Int64, Nothing} = nothing
    for (_, val) in ims
        time_steps = check_two_time_steps(time_steps, get_time_steps_number(val))
    end
    time_steps
end
