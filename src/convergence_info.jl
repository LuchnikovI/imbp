module ConvergenceInfo
    export InfoCell, add_point!

    mutable struct InfoCell{F<:AbstractFloat}
        max_truncation_error::Union{F, Nothing}
        min_truncation_error::Union{F, Nothing}
        mean_truncation_error::Union{F, Nothing}
        mean_sq_truncation_error::Union{F, Nothing}
        std_truncation_error::Union{F, Nothing}
        max_discrepancy::F
        min_discrepancy::F
        mean_discrepancy::F
        mean_sq_discrepancy::F
        std_discrepancy::F
        statistics_size::UInt

        InfoCell(::Type{F}) where {F<:AbstractFloat} = new{F}(zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), 0)
    end
    update_mean(old_mean::F, new_val::F, stat_size::UInt) where {F<:AbstractFloat} = (old_mean * stat_size + new_val) / (stat_size + 1)
    update_mean_sq(old_mean_sq::F, new_val::F, stat_size::UInt) where {F<:AbstractFloat} = (old_mean_sq * stat_size + new_val^2) / (stat_size + 1)
    get_std(mean::F, mean_sq::F)  where {F<:AbstractFloat} = sqrt(abs(mean_sq - mean^2))

    function add_point!(info_cell::InfoCell{F}, truncation_error::Union{F, Nothing}, infidelity::F) where {F<:AbstractFloat}
        if truncation_error == nothing
            info_cell.max_truncation_error = nothing
            info_cell.min_truncation_error = nothing
            info_cell.mean_truncation_error = nothing
            info_cell.mean_sq_truncation_error = nothing
            info_cell.std_truncation_error = nothing
        else
            info_cell.max_truncation_error = max(truncation_error, info_cell.max_truncation_error)
            info_cell.min_truncation_error = min(truncation_error, info_cell.min_truncation_error)
            info_cell.mean_truncation_error = update_mean(info_cell.mean_truncation_error, truncation_error, info_cell.statistics_size)
            info_cell.mean_sq_truncation_error = update_mean_sq(info_cell.mean_sq_truncation_error, truncation_error, info_cell.statistics_size)
            info_cell.std_truncation_error = get_std(info_cell.mean_truncation_error, info_cell.mean_sq_truncation_error)
        end
        info_cell.max_discrepancy = max(infidelity, info_cell.max_discrepancy)
        info_cell.min_discrepancy = min(infidelity, info_cell.min_discrepancy)
        info_cell.mean_discrepancy = update_mean(info_cell.mean_discrepancy, infidelity, info_cell.statistics_size)
        info_cell.mean_sq_discrepancy = update_mean_sq(info_cell.mean_sq_discrepancy, infidelity, info_cell.statistics_size)
        info_cell.std_discrepancy = get_std(info_cell.mean_discrepancy, info_cell.mean_sq_discrepancy)
        info_cell.statistics_size += 1
    end

    function Base.show(io::IO, info_cell::InfoCell)
        print(
            io,
            "maximal truncation error: ",
            info_cell.max_truncation_error, '\n',
            "minimal truncation error: ",
            info_cell.min_truncation_error, '\n',
            "mean truncation error: ",
            info_cell.mean_truncation_error, '\n',
            "truncation error standart deviation: ",
            info_cell.std_truncation_error, '\n',
            "maximal discrepancy: ",
            info_cell.max_discrepancy, '\n',
            "minimal discrepancy: ",
            info_cell.min_discrepancy, '\n',
            "mean discrepancy: ",
            info_cell.mean_discrepancy, '\n',
            "discrepancy standart deviation: ",
            info_cell.std_discrepancy,
        )
    end
end