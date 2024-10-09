module Utils

export QSim, apply_two_qubit_gate!, apply_one_qubit_gate!, get_one_qubit_dens, random_unitary, random_state

using LinearAlgebra
using Random

@inline insert_zero(index::UInt64, mask::UInt64, flipped_mask::UInt64)::UInt64 = ((index & mask) << 1) | (index & flipped_mask)

@inline get_mask(pos::UInt64)::UInt64 = typemax(UInt64) << (pos - 1)

@inline flip_mask(mask::UInt64)::UInt64 = ~mask

@inline get_stride(pos::UInt64)::UInt64 = 1 << (pos - 1)

mutable struct QSim{T<:Real}
    state::Vector{Complex{T}}
    qubits_number::UInt64
    function QSim(::Type{T}, qubits_number::Int) where {T<:Real}
        elems_number = 2^qubits_number
        state = zeros(Complex{T}, elems_number)
        state[1] = one(Complex{T})
        new{T}(state, convert(UInt64, qubits_number))
    end
end

function apply_two_qubit_gate!(qsim::QSim{T}, gate::Matrix{Complex{T}}, pos1::Int, pos2::Int) where {T<:Real}
    @assert size(gate) == (4, 4)
    @assert pos1 != pos2
    @assert pos1 <= qsim.qubits_number
    @assert pos1 >= 1
    @assert pos2 <= qsim.qubits_number
    @assert pos2 >= 1
    apply_two_qubit_gate!(qsim, gate, convert(UInt64, pos1), convert(UInt64, pos2))
end

function apply_two_qubit_gate!(
    qsim::QSim{T},
    gate::Matrix{Complex{T}},
    pos2::UInt64,
    pos1::UInt64,
) where {T<:Real}
    mask1 = get_mask(pos1)
    mask2 = get_mask(pos2)
    min_mask = min(mask1, mask2)
    max_mask = max(mask1, mask2)
    flipped_max_mask = flip_mask(max_mask)
    flipped_min_mask = flip_mask(min_mask)
    stride1 = get_stride(pos1)
    stride2 = get_stride(pos2)
    size = 1 << (qsim.qubits_number - 2)
    state = qsim.state
    @inbounds for i in zero(UInt64):UInt64(size - 1)
        bi = insert_zero(insert_zero(i, max_mask, flipped_max_mask), min_mask, flipped_min_mask)
        tmp1 = (
              gate[begin, begin    ] * state[begin + bi]
            + gate[begin, begin + 1] * state[begin + bi + stride1]
            + gate[begin, begin + 2] * state[begin + bi + stride2]
            + gate[begin, begin + 3] * state[begin + bi + stride1 + stride2]
        )
        tmp2 = (
              gate[begin + 1, begin    ] * state[begin + bi]
            + gate[begin + 1, begin + 1] * state[begin + bi + stride1]
            + gate[begin + 1, begin + 2] * state[begin + bi + stride2]
            + gate[begin + 1, begin + 3] * state[begin + bi + stride1 + stride2]
        )
        tmp3 = (
              gate[begin + 2, begin    ] * state[begin + bi]
            + gate[begin + 2, begin + 1] * state[begin + bi + stride1]
            + gate[begin + 2, begin + 2] * state[begin + bi + stride2]
            + gate[begin + 2, begin + 3] * state[begin + bi + stride1 + stride2]
        )
        state[begin + bi + stride1 + stride2] = (
              gate[begin + 3, begin    ] * state[begin + bi]
            + gate[begin + 3, begin + 1] * state[begin + bi + stride1]
            + gate[begin + 3, begin + 2] * state[begin + bi + stride2]
            + gate[begin + 3, begin + 3] * state[begin + bi + stride1 + stride2]
        )
        state[begin + bi] = tmp1
        state[begin + bi + stride1] = tmp2
        state[begin + bi + stride2] = tmp3
    end
end

function apply_one_qubit_gate!(qsim::QSim{T}, gate::Matrix{Complex{T}}, pos::Int) where {T<:Real}
    @assert size(gate) == (2, 2)
    @assert pos <= qsim.qubits_number
    @assert pos >= 1
    apply_one_qubit_gate!(qsim, gate, convert(UInt64, pos))
end

function apply_one_qubit_gate!(qsim::QSim{T}, gate::Matrix{Complex{T}}, pos::UInt64) where {T<:Real}
    mask = get_mask(pos)
    flipped_mask = flip_mask(mask)
    stride = get_stride(pos)
    size = 1 << (qsim.qubits_number - 1)
    state = qsim.state
    @inbounds for i in zero(UInt64):UInt64(size - 1)
        bi = insert_zero(i, mask, flipped_mask)
        tmp = gate[begin, begin] * state[begin + bi] + gate[begin, begin + 1] * state[begin + bi + stride]
        state[begin + bi + stride] = gate[begin + 1, begin] * state[begin + bi] + gate[begin + 1, begin + 1] * state[begin + bi + stride]
        state[begin + bi] = tmp
    end
end

function get_one_qubit_dens(qsim::QSim{T}, pos::Int) where {T<:Real}
    @assert pos <= qsim.qubits_number
    @assert pos >= 1
    get_one_qubit_dens(qsim, convert(UInt64, pos))
end

function get_one_qubit_dens(qsim::QSim{T}, pos::UInt64) where {T<:Real}
    dens = zeros(Complex{T}, 2, 2)
    mask = get_mask(pos)
    flipped_mask = flip_mask(mask)
    stride = get_stride(pos)
    size = 1 << (qsim.qubits_number - 1)
    state = qsim.state
    @inbounds for i in zero(UInt64):UInt64(size - 1)
        bi = insert_zero(i, mask, flipped_mask)
        dens[begin,     begin    ] += state[begin + bi         ] * conj(state[begin + bi])
        dens[begin,     begin + 1] += state[begin + bi         ] * conj(state[begin + bi + stride])
        dens[begin + 1, begin    ] += state[begin + bi + stride] * conj(state[begin + bi])
        dens[begin + 1, begin + 1] += state[begin + bi + stride] * conj(state[begin + bi + stride])
    end
    dens
end

function random_unitary(::Type{T}, rng, size::Int) where {T<:Real}
    m = randn(rng, Complex{T}, size, size)
    Matrix(qr(m).Q)
end

function random_state(::Type{T}, rng, size::Int) where {T<:Real}
    m = randn(rng, Complex{T}, size, size)
    m = m * conj(transpose(m))
    m / tr(m)
end

end