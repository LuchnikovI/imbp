function get_or_default(kwargs, key, default)
    try
        kwargs[key]
    catch
        @warn "Parameter $key has not been set, falling into default value $default"
        default
    end
end