using AlgorithmicRecourseDynamics: is_logging
using CSV
using DataFrames
using Images
using Plots
using ProgressMeter: Progress, next!
using RCall
using Statistics

function run_bootstrap(
    results::Dict{Symbol, ExperimentResults}, n_bootstrap::Int=1000; 
    filename::String="bootstrapped_results.csv", show_progress=!is_logging(stderr)
)
    df = DataFrame()
    for (key, val) in results
        n_folds = length(val.experiment.recourse_systems)
        p_fold = Progress(n_folds; desc="Progress on folds:", showspeed=true, enabled=show_progress, output = stderr)
        for fold in 1:n_folds
            N = length(val.experiment.system_identifiers)
            p_sys = Progress(N; desc="Progress on systems:", showspeed=true, enabled=show_progress, output = stderr)
            Threads.@threads for i in 1:N
                rec_sys = val.experiment.recourse_systems[fold][i]
                model_name, gen_name = collect(val.experiment.system_identifiers)[i]
                df_ = evaluate_system(rec_sys, val.experiment; n=n_bootstrap)
                df_.data .= key
                df_.model .= model_name
                df_.generator .= gen_name
                df_.fold .= fold
                df = vcat(df, df_)
                next!(p_sys, showvalues = [(:Model, model_name), (:Generator, gen_name)])
            end
            next!(p_fold)
        end
    end
    df = mapcols(x -> typeof(x) == Vector{Symbol} ? string.(x) : x, df)
    CSV.write(filename, df)
    return df
end

function Plots.plot(results::ExperimentResults, variable::Symbol=:mmd, scope::Symbol=:model; size=3, title=nothing, kwargs...)

    df = results.output
    @assert variable in unique(df.name) "Not a valid variable."

    gdf = groupby(df, [:generator, :model, :n, :name, :scope])
    df_plot = combine(gdf, :value => (x -> [(mean(x), mean(x) + std(x), mean(x) - std(x))]) => [:mean, :ymax, :ymin])
    df_plot = mapcols(x -> typeof(x) == Vector{Symbol} ? string.(x) : x, df_plot)
    df_plot.name .= [r[:name] == "mmd" ? "$(r[:name])_$(r[:scope])" : r[:name] for r in eachrow(df_plot)]
    select!(df_plot, Not(:scope))

    ncol = length(unique(df_plot.model))
    nrow = length(unique(df_plot.name))
    R"""
    library(ggplot2)
    plt <- ggplot($df_plot,aes(x=n, y=mean, ymin=ymin, ymax=ymax, color=generator)) +
        geom_ribbon(aes(fill=generator), alpha=0.5) +
        geom_line() +
        facet_wrap(name ~ model, scale="free_y", ncol=$ncol) +
        labs(
            x = "Round",
            y = "Value",
            title = $title
        )
    temp_path <- file.path(tempdir(), "plot.png")
    ggsave(temp_path,width=$ncol * $size,height=$nrow * $size * 0.8)
    """

    img = Images.load(rcopy(R"temp_path"))
    return img

end

function Plots.plot(results::ExperimentResults, n::Int, variable::Symbol=:mmd, scope::Symbol=:model; size=3, title=nothing, kwargs...)

    df = results.output
    @assert variable in unique(df.name) "Not a valid variable."
    @assert n in unique(df.n) "No results for round `n`."
    df = df[df.n.==n, :]

    gdf = groupby(df, [:generator, :model, :n, :name, :scope])
    df_plot = combine(gdf, :value => (x -> [(mean(x), mean(x) + std(x), mean(x) - std(x))]) => [:mean, :ymax, :ymin])
    df_plot = mapcols(x -> typeof(x) == Vector{Symbol} ? string.(x) : x, df_plot)
    df_plot.name .= [r[:name] == "mmd" ? "$(r[:name])_$(r[:scope])" : r[:name] for r in eachrow(df_plot)]
    select!(df_plot, Not(:scope))

    ncol = length(unique(df_plot.model))
    nrow = length(unique(df_plot.name))
    R"""
    library(ggplot2)
    plt <- ggplot($df_plot) +
        geom_bar(aes(x=n, y=mean, fill=generator), stat="identity", alpha=0.5, position="dodge") +
        geom_pointrange( aes(x=n, y=mean, ymin=ymin, ymax=ymax, colour=generator), alpha=0.9, position=position_dodge(width=0.9), size=1.0) +
        facet_wrap(name ~ model, scale="free_y", ncol=$ncol) +
        labs(
            x = "Round",
            y = "Value",
            title = $title
        ) + 
        theme(
            axis.title.x=element_blank(),
            axis.text.x=element_blank(),
            axis.ticks.x=element_blank()
        )
    temp_path <- file.path(tempdir(), "plot.png")
    ggsave(temp_path,width=$ncol * $size,height=$nrow * $size * 0.8) 
    """

    img = Images.load(rcopy(R"temp_path"))
    return img

end

function kable(result::ExperimentResults, n::Vector{Int}; format="latex")
    df = deepcopy(result.output)
    mapcols!(x -> eltype(x) == Symbol ? string.(x) : x, df)
    R"""
    library(data.table)
    dt <- data.table($df)
    n_ <- $n
    dt <- dt[n %in% n_]
    dt[,name:=ifelse(name=="mmd",paste0(name,scope),name)][,scope:=NULL]
    dt <- dt[,.(value=mean(value,na.rm=TRUE),sd=sd(value)),by=.(model,generator,name,n)]
    dt[,text:=sprintf("%0.3f (%0.3f)",value,sd)][,value:=NULL][,sd:=NULL]
    dt <- dcast(dt, ... ~ name, value.var="text")
    library(kableExtra)
    dt <- dt[order(n)]
    setcolorder(dt, c("n"))
    ktab <- kbl(dt, booktabs = T, align = "c", format=$format) %>%
        column_spec(1, bold = T, width = "5em") %>%
        collapse_rows(columns = 1:3, latex_hline = "major", valign = "middle")
    """
    return println(rcopy(R"ktab"))
end

using DataFrames
function kable(
    results::Dict{Symbol,ExperimentResults},
    n::Vector{Int};
    format="latex",
    exclude_metric::Vector{Symbol}=[:mmd_grid]
)
    df = DataFrame()
    for (key, val) in results
        df_ = deepcopy(val.output)
        df_.dataset .= key
        df = vcat(df, df_)
    end
    mapcols!(x -> eltype(x) == Symbol ? string.(x) : x, df)
    R"""
    library(data.table)
    dt <- data.table($df)
    n_ <- $n
    dt <- dt[n %in% n_]
    dt[,name:=ifelse(name=="mmd",paste0(name,"_",scope),name)][,scope:=NULL]
    dt <- dt[,.(value=mean(value,na.rm=TRUE),sd=sd(value)),by=.(dataset,model,generator,name,n)]
    dt[,text:=sprintf("%0.3f (%0.3f)",value,sd)][,value:=NULL][,sd:=NULL]
    dt <- dcast(dt, ... ~ name, value.var="text")
    library(kableExtra)
    dt <- dt[order(-dataset,n)]
    setcolorder(dt, c("dataset","n"))
    ktab <- kbl(dt, booktabs = T, align = "c", format=$format) %>%
        column_spec(1, bold = T, width = "5em") %>%
        collapse_rows(columns = 1:4, latex_hline = "major", valign = "middle")
    """
    return println(rcopy(R"ktab"))
end