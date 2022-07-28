using Gadfly
import Plots: plot
using DataFrames

function plot(results::ExperimentResults, variable::Symbol=:mmd; kwargs...)
    
    df = results.output
    @assert variable in unique(df.name) "Not a valid variable."

    gdf = groupby(df, [:generator, :model, :n, :name, :scope])
    df_plot = combine(gdf, :value => (x -> [(mean(x),mean(x)+std(x),mean(x)-std(x))]) => [:mean, :ymax, :ymin])


    # Plot validity:
    plt = Gadfly.plot(
        df_plot[df_plot.name.==variable,:], 
        ygroup=:scope, xgroup=:model, x=:n, y=:mean, ymin=:ymin, ymax=:ymax, color=:generator,
        Geom.subplot_grid(Geom.line, Geom.ribbon, free_y_axis=true),
        kwargs...
    )

    return plt

end

