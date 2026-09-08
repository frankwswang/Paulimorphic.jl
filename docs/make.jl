using Documenter: DocMeta, HTML, makedocs, deploydocs
using Paulimorphic

DocMeta.setdocmeta!(Paulimorphic, :DocTestSetup, :(using Paulimorphic); recursive=true)

makedocs(
    sitename = "Paulimorphic", 
    authors = "Weishi Wang and contributors", 
    modules = [Paulimorphic], 
    format = HTML(
        canonical = "https://frankwswang.github.io/Paulimorphic.jl", 
        edit_link = "dev", 
    ), 
    pages = [
        "Home" => "index.md", 
        "Core Types" => "types.md", 
        "Core Functions" => "functions.md", 
    ], 
    checkdocs = :all, 
)

deploydocs(
    repo = "github.com/frankwswang/Paulimorphic.jl.git", 
    devbranch = "dev", 
)
