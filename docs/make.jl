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
        assets = ["assets/custom.css"], 
    ), 
    pages = [
        "Home" => "index.md", 
        "Examples" => "examples.md", 
        "Types" => "types.md", 
        "Functions" => "functions.md", 
        "Development Guide" => "development.md", 
    ], 
    checkdocs = :public, 
)

deploydocs(
    repo = "github.com/frankwswang/Paulimorphic.jl.git", 
    devbranch = "dev", 
)
