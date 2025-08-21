using Documenter
using RunningStats

makedocs(;
    modules=[RunningStats],
    authors="Anthony Meza <ameza@mit.edu>",
    repo="https://github.com/anthony-meza/RunningStats.jl/blob/{commit}{path}#{line}",
    sitename="RunningStats.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://anthony-meza.github.io/RunningStats.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Mathematical Background" => "mathematical_background.md",
        "API Reference" => "api.md",
        "Examples" => "examples.md",
    ],
)