using Documenter
using AlgorithmicRecourseDynamics

makedocs(
    sitename = "AlgorithmicRecourseDynamics",
    format = Documenter.HTML(),
    modules = [AlgorithmicRecourseDynamics]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
