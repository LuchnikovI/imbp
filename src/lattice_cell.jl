include("convergence_info.jl")

using .ConvergenceInfo

mutable struct LatticeCell{
        N<:Number,
        TQG<:Node{<:AbstractArray{N, 4}},
        IS<:Node{<:AbstractVector{N}},
        OQG<:Node{<:AbstractMatrix{N}},
    }
    two_qubit_gates_seq::Vector{Tuple{Int, Int}}
    two_qubit_gates::Dict{KernelID, Vector{TQG}}
    initial_states::Vector{IS}
    one_qubit_gates::Vector{Vector{OQG}}
    time_steps::Union{Nothing, Int}

    function LatticeCell(initial_states::Vector{<:AbstractMatrix{<:Number}})
        dispatch_arr = first(initial_states)
        ET = eltype(first(initial_states))
        U = typeof(dispatch_arr)
        IS = Node{U.name.wrapper{ET, 1}}
        TQG = Node{U.name.wrapper{ET, 4}}
        OQG = Node{U.name.wrapper{ET, 2}}
        one_qubit_gates = [Vector{OQG}[] for _ in 1:length(initial_states)]
        initial_states = map(dens -> begin
            check_density(dens)
            Node(reshape(dens, (4,)), :pout)
        end, initial_states)
        new{ET, TQG, IS, OQG}(Tuple{Int, Int}[], Dict{KernelID, TQG}(), initial_states, one_qubit_gates)
    end
end

get_nodes_number(lattice_cell::LatticeCell) = length(lattice_cell.initial_states)

get_interaction_terms_number(lattice_cell::LatticeCell) = length(lattice_cell.two_qubit_gates_seq)

Base.in(time_pos::Integer, lattice_cell::LatticeCell) = get_nodes_number(lattice_cell) >= time_pos ? true : false

function Base.getindex(lattice_cell::LatticeCell, time_pos::Integer)
    time_pos in lattice_cell ? lattice_cell.two_qubit_gates_seq[time_pos] : error("There is not a gate with time position $gate_pos")
end

update_time_steps(::Nothing, given_time_steps::Int) = given_time_steps
function update_time_steps(time_steps::Int, given_time_steps::Int)
    if time_steps == given_time_steps
        time_steps
    else
        error("Number of time steps in the circuit is $time_steps, but the number of gates given in a row is $given_time_steps")
    end
end

function add_two_qubit_gates!(
    lattice_cell::LatticeCell,
    node1::Integer,
    node2::Integer,
    gates::Vector{<:AbstractMatrix},
)
    if node1 == 0 || node2 == 0
        error("Node ID must be > 0, got IDs $node1 and $node2")
    elseif node1 == node2
        error("Gate must act on different nodes, but it acts on $node1 and $node2")
    elseif !(node1 in lattice_cell)
        error("There is not a gate with number $node1 in the lattice cell")
    elseif !(node2 in lattice_cell)
        error("There is not a gate with number $node2 in the lattice cell")
    end
    for gate in gates
        gate_shape = size(gate)
        if gate_shape != (16, 16)
            error("Gate shape must be equal to (16, 16), but got an array of shape $gate_shape")
        end
        check_channel(node1, node2, gate)
    end
    given_time_steps = length(gates)
    lattice_cell.time_steps = update_time_steps(lattice_cell.time_steps, given_time_steps)
    push!(lattice_cell.two_qubit_gates_seq, (node1, node2))
    ker_id = length(lattice_cell.two_qubit_gates_seq)
    lattice_cell.two_qubit_gates[KernelID(ker_id, true, (node1, node2))] = map(gate -> begin
        gate = reshape(permutedims(reshape(gate, (2, 2, 2, 2, 2, 2, 2, 2)), (3, 1, 4, 2, 7, 5, 8, 6)), (4, 4, 4, 4))
        Node(
            gate,
            :second_out,
            :first_out,
            :second_inp,
            :first_inp,
        )
    end, gates)
    lattice_cell.two_qubit_gates[KernelID(ker_id, false, (node1, node2))] = map(gate -> begin
        gate = reshape(permutedims(reshape(gate, (2, 2, 2, 2, 2, 2, 2, 2)), (3, 1, 4, 2, 7, 5, 8, 6)), (4, 4, 4, 4))
        Node(
            gate,
            :first_out,
            :second_out,
            :first_inp,
            :second_inp,
        )
    end, gates)
end

function add_one_qubit_gates!(
    lattice_cell::LatticeCell,
    node::Integer,
    gates::Vector{<:AbstractMatrix},
)
    if node == 0
        error("Node ID must be > 0, got ID $node")
    elseif !(node in lattice_cell)
        error("There is not a gate with number $node in the lattice cell")
    end
    for gate in gates
        gate_shape = size(gate)
        if gate_shape != (4, 4)
            error("Gate shape must be equal to (4, 4), but got an array of shape $gate_shape")
        end
    end
    given_time_steps = length(gates)
    lattice_cell.time_steps = update_time_steps(lattice_cell.time_steps, given_time_steps)
    lattice_cell.one_qubit_gates[node] = map(gate -> begin
        gate = reshape(permutedims(reshape(gate, (2, 2, 2, 2)), (2, 1, 4, 3)), (4, 4))
        Node(gate, :pout, :pinp)
    end, gates)
end

function get_equations(lattice_cell::LatticeCell)
    dispatch_arr = get_array(last(lattice_cell.initial_states))
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
    flatten_self_consistency_eqs = Equation[]
    for node_eqs in self_consistency_eqs
        for equation in node_eqs
            push!(flatten_self_consistency_eqs, equation)
        end
    end
    sort!(flatten_self_consistency_eqs, by = eqs -> begin
        pos = findfirst(map(x -> isa(x, KernelID), eqs))
        (eqs[pos].time_position, eqs[pos].is_forward)
    end)
    one_qubit_gates = map(x -> begin
        if isempty(x)
            [Node(get_similar_identity(dispatch_arr, 4), :pinp, :pout) for _ in 1:lattice_cell.time_steps]
        else
            x
        end
    end, lattice_cell.one_qubit_gates)
    Equations(
        flatten_self_consistency_eqs,
        marginal_eqs,
        lattice_cell.two_qubit_gates,
        one_qubit_gates,
        lattice_cell.initial_states,
        lattice_cell.time_steps,
    )
end

function initialize_ims(
    ::Type{I},
    lattice_cell::LatticeCell,
) where {I<:AbstractIM}
    ims = Dict{IMID, I}()
    for (time_position, (left_node, right_node)) in enumerate(lattice_cell.two_qubit_gates_seq)
        forward_id = IMID(time_position, true, (left_node, right_node))
        im = initialize_im(I)
        ims[forward_id] = im
        backward_id = IMID(time_position, false, (left_node, right_node))
        ims[backward_id] = im
    end
    ims
end

function _single_equation_iter(
    equation::Equation,
    ims::Dict{IMID,<:AbstractIM},
    kernels::Dict{KernelID, Vector{N}} where {N<:Node},
    one_qubit_gates::Vector{<:Node},
    initial_state::Node,
    rank_or_eps::Union{Integer,AbstractFloat},
    iter_num::Int;
    kwargs...,
)
    kernel_ids = findall(id -> isa(id, KernelID), equation)
    @assert(length(kernel_ids) == 1)
    kernel_id = equation[kernel_ids[1]]
    @assert(isa(kernel_id, KernelID))
    new_im, trunc_err = contract(
        equation,
        ims,
        kernels,
        one_qubit_gates,
        initial_state,
        rank_or_eps,
        iter_num;
        kwargs...
    )
    new_im, IMID(kernel_id), trunc_err
end

function _single_iter!(
    eqs::Equations,
    ims::Dict{IMID,<:AbstractIM},
    rank_or_eps::Union{Integer,F},
    iter_num::Int,
    information::Vector{<:InfoCell{F}};
    kwargs...,
) where {F<:AbstractFloat}
    info_cell = InfoCell(F)
    for equation in eqs.self_consistency_eqs
        # TODO: too ugly
        node = isa(equation[1], KernelID) ? equation[1].is_forward ? equation[1].nodes[1] : equation[1].nodes[2] : equation[1].is_forward ? equation[1].nodes[2] : equation[1].nodes[1]
        (new_im, imid, trunc_err) = _single_equation_iter(
            equation,
            ims,
            eqs.kernels,
            eqs.one_qubit_gates[node],
            eqs.initial_states[node],
            rank_or_eps,
            iter_num;
            kwargs...,
        )
        direction = imid.is_forward ? imid.nodes : (imid.nodes[2], imid.nodes[1])
        add_point!(info_cell, trunc_err, im_distance(new_im, ims[imid]; kwargs...))
        final_bond_dims = get_bond_dimensions(new_im)
        @debug "IM recomputed:" direction info_cell final_bond_dims
        ims[imid] = new_im
    end
    push!(information, info_cell)
end

function iterate_equations!(
    eqs::Equations,
    ims::Dict{IMID, <:AbstractIM},
    rank_or_eps::Union{Integer, F},
    callback=nothing;
    kwargs...,
) where {F<:AbstractFloat}
    information = InfoCell{Float64}[]
    time_steps = eqs.time_steps
    for iter_num in 1:time_steps
        _single_iter!(eqs, ims, rank_or_eps, iter_num, information; kwargs...)
        current_information = information[end]
        if !isnothing(callback)
            callback(iter_num, current_information, ims)
        end
    end
    information
end
