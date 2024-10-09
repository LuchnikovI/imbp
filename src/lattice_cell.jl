using Logging
using LinearAlgebra
using TensorOperations

function check_density(initial_states::Vector{Matrix{N}}) where {N<:Number}
    for (node_number, in_st) in enumerate(initial_states)
        trace = tr(in_st)
        un_unit_traceness = norm(trace - 1)
        inhermicity = norm(in_st .- conj(transpose(in_st)))
        evals = eigvals(in_st)
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
end

function check_channel(node1::Int64, node2::Int64, channel::Matrix{N}) where {N<:Number}
    choi = permutedims(reshape(channel, (4, 4, 4, 4)), (1, 3, 2, 4))
    @tensor trace[j, k] := choi[i, j, i, k]
    un_identity_traceness = norm(trace - Matrix{N}(I, 4, 4))
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

mutable struct LatticeCell{N<:Number}
    two_qubit_gates_seq::Vector{Tuple{Int64, Int64}}
    two_qubit_gates::Dict{KernelID, Array{N, 4}}
    initial_states::Vector{Vector{N}}
    one_qubit_gates::Vector{Array{N, 2}}
    function LatticeCell(::Type{N}, initial_states::Vector{Matrix{N}}) where {N<:Number}
        one_qubit_gates = [Matrix{N}(I, 4, 4) for _ in 1:length(initial_states)]
        check_density(initial_states)
        initial_states = map(dens -> begin
            dens_shape = size(dens)
            if dens_shape != (2, 2)
                error("Initial density matrix must be a matrix of shape (2, 2), got shape $dens_shape")
            end
            reshape(dens, (4,))
        end, initial_states)
        new{N}(Tuple{Int64, Int64}[], Dict{KernelID, Array{N, 4}}(), initial_states, one_qubit_gates)
    end
end

get_nodes_number(lattice_cell::LatticeCell) = length(lattice_cell.initial_states)

get_gates_number(lattice_cell::LatticeCell) = length(lattice_cell.two_qubit_gates_seq)

Base.in(time_pos::Int64, lattice_cell::LatticeCell) = get_nodes_number(lattice_cell) >= time_pos ? true : false

function Base.getindex(lattice_cell::LatticeCell, time_pos::Int64)
    time_pos in lattice_cell ? lattice_cell.two_qubit_gates_seq[time_pos] : error("There is not a gate with time position $gate_pos")
end

function add_two_qubit_gate!(lattice_cell::LatticeCell{N}, node1::Int64, node2::Int64, gate::Matrix{N}) where {N<:Number}
    if node1 == 0 || node2 == 0
        error("Node ID must be > 0, got IDs $node1 and $node2")
    elseif node1 == node2
        error("Gate must act on different nodes, but it acts on $node1 and $node2")
    elseif !(node1 in lattice_cell)
        error("There is not a gate with number $node1 in the lattice cell")
    elseif !(node2 in lattice_cell)
        error("There is not a gate with number $node2 in the lattice cell")
    end
    gate_shape = size(gate)
    if gate_shape != (16, 16)
        error("Gate shape must be equal to (16, 16), but got an array of shape $gate_shape")
    end
    check_channel(node1, node2, gate)
    gate = reshape(permutedims(reshape(gate, (2, 2, 2, 2, 2, 2, 2, 2)), (3, 1, 4, 2, 7, 5, 8, 6)), (4, 4, 4, 4))
    push!(lattice_cell.two_qubit_gates_seq, (node1, node2))
    ker_id = length(lattice_cell.two_qubit_gates_seq)
    lattice_cell.two_qubit_gates[KernelID(ker_id, true, (node1, node2))] = permutedims(gate, (2, 1, 4, 3))
    lattice_cell.two_qubit_gates[KernelID(ker_id, false, (node1, node2))] = gate
end

function add_one_qubit_gate!(lattice_cell::LatticeCell{N}, node::Int64, gate::Matrix{N}) where {N<:Number}
    if node == 0
        error("Node ID must be > 0, got ID $node")
    elseif !(node in lattice_cell)
        error("There is not a gate with number $node in the lattice cell")
    end
    gate_shape = size(gate)
    if gate_shape != (4, 4)
        error("Gate shape must be equal to (4, 4), but got an array of shape $gate_shape")
    end
    # TODO: add CPTP properties validation
    gate = reshape(permutedims(reshape(gate, (2, 2, 2, 2)), (2, 1, 4, 3)), (4, 4))
    lattice_cell.one_qubit_gates[node] = gate
end

function get_equations(lattice_cell::LatticeCell{N})::Equations where{N<:Number}
    self_consistency_eqs = [Equation[] for _ in 1:get_nodes_number(lattice_cell)]
    marginal_eqs = [Equation() for _ in 1:get_nodes_number(lattice_cell)]
    for (time_position, (left_node, right_node)) in enumerate(lattice_cell.two_qubit_gates_seq)
        left2right_im = IMID(time_position, true, (left_node, right_node))
        right2left_im = IMID(time_position, false, (left_node, right_node))
        left2right_ker = KernelID(time_position, true, (left_node, right_node))
        right2left_ker = KernelID(time_position, false, (left_node, right_node))
        for equation in self_consistency_eqs[left_node]
            push!(equation, right2left_im)
        end
        for equation in self_consistency_eqs[right_node]
            push!(equation, left2right_im)
        end
        new_eq = copy(marginal_eqs[left_node])
        push!(new_eq, left2right_ker)
        push!(self_consistency_eqs[left_node], new_eq)
        new_eq = copy(marginal_eqs[right_node])
        push!(new_eq, right2left_ker)
        push!(self_consistency_eqs[right_node], new_eq)
        push!(marginal_eqs[left_node], right2left_im)
        push!(marginal_eqs[right_node], left2right_im)
    end
    Equations(
        self_consistency_eqs,
        marginal_eqs,
        lattice_cell.two_qubit_gates,
        lattice_cell.one_qubit_gates,
        lattice_cell.initial_states,
    )
end

function initialize_ims_by_perfect_dissipators(
    ::Type{I},
    lattice_cell::LatticeCell,
    time_steps_number::Int64,
)::Dict{IMID, I} where {I<:AbstractIM}
    ims = Dict{IMID, I}()
    for (time_position, (left_node, right_node)) in enumerate(lattice_cell.two_qubit_gates_seq)
        forward_id = IMID(time_position, true, (left_node, right_node))
        perf_diss = get_perfect_dissipator_im(I, time_steps_number)
        truncate!(perf_diss, 1)
        ims[forward_id] = perf_diss
        backward_id = IMID(time_position, false, (left_node, right_node))
        ims[backward_id] = perf_diss
    end
    ims
end

function _single_equation_iter(
    equation::Equation,
    ims::Dict{IMID, I},
    kernels::Dict{KernelID, A},
    one_qubit_gate::AbstractArray,
    initial_state::AbstractArray,
) where {I<:AbstractIM, A<:AbstractArray}
    kernel_ids = findall(id -> isa(id, KernelID), equation)
    @assert(length(kernel_ids) == 1)
    kernel_id = equation[kernel_ids[1]]
    @assert(isa(kernel_id, KernelID))
    new_im = contract(equation, ims, kernels, one_qubit_gate, initial_state)
    new_im, IMID(kernel_id)
end

function _single_iter!(
    eqs::Equations,
    ims::Dict{IMID, I},
    rank_or_eps::Union{Int64, F}
) where {F<:AbstractFloat, I<:AbstractIM}
    min_log_fid = 0.
    for (node, node_eqs) in enumerate(eqs.self_consistency_eqs)
        for equation in node_eqs
            (new_im, imid) = _single_equation_iter(equation, ims, eqs.kernels, eqs.one_qubit_gates[node], eqs.initial_states[node])
            trunc_err = truncate!(new_im, rank_or_eps)
            direction = imid.is_forward ? imid.nodes : (imid.nodes[2], imid.nodes[1])
            log_fid = log_fidelity(new_im, ims[imid])
            min_log_fid = min(log_fid, min_log_fid)
            final_bond_dims = get_bond_dimensions(new_im)
            @debug "IM recomputed:" direction log_fid trunc_err final_bond_dims
            ims[imid] = new_im
        end
    end
    min_log_fid
end

function iterate_equations!(
    eqs::Equations,
    ims::Dict{IMID, I},
    rank_or_eps::Union{Int64, N},
    max_iter::Int64 = 100,
    log_fid::AbstractFloat = 1e-10,
) where {N<:AbstractFloat, I<:AbstractIM}
    min_log_fid = 0.
    for iter_num in 1:convert(UInt64, max_iter)
        min_log_fid = _single_iter!(eqs, ims, rank_or_eps)
        @info "Iteration" iter_num min_log_fid
        if log_fid < min_log_fid
            return min_log_fid
        end
    end
    min_log_fid
end
