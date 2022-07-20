function metric_to_dataframe(metric, name, scope)
    if !isa(metric, AbstractVector)
        metric = [metric]
    end
    metric = DataFrame(NamedTuple{(:value, :p_value, :class)}.(metric))
    metric.metric .= name
    metric.scope .= scope
    return metric
end