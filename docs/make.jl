using AlgorithmicRecourseDynamics
using Documenter

DocMeta.setdocmeta!(AlgorithmicRecourseDynamics, :DocTestSetup, :(using AlgorithmicRecourseDynamics); recursive=true)

makedocs(;
    modules=[AlgorithmicRecourseDynamics],
    authors="Anonymous",
    repo="https://github.com/pat-alt/AlgorithmicRecourseDynamics.jl/blob/{commit}{path}#{line}",
    sitename="AlgorithmicRecourseDynamics.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://pat-alt.github.io/AlgorithmicRecourseDynamics.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/pat-alt/AlgorithmicRecourseDynamics.jl",
    devbranch="main",
)