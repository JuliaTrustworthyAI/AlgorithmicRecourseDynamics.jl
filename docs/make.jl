using AlgorithmicRecourseDynamics
using Documenter

ex_meta = quote
    # Import module(s):
    using AlgorithmicRecourseDynamics
end

DocMeta.setdocmeta!(AlgorithmicRecourseDynamics, :DocTestSetup, ex_meta; recursive = true)

makedocs(;
    modules = [AlgorithmicRecourseDynamics],
    authors = "Patrick Altmeyer",
    repo = "https://github.com/juliatrustworthyai/AlgorithmicRecourseDynamics.jl/blob/{commit}{path}#{line}",
    sitename = "AlgorithmicRecourseDynamics.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://juliatrustworthyai.github.io/AlgorithmicRecourseDynamics.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "🏠 Home" => "index.md",
        "🧐 Reference" => "_reference.md",
    ],
)

deploydocs(; repo = "github.com/juliatrustworthyai/AlgorithmicRecourseDynamics.jl", devbranch = "main")
