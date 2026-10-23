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
        collapselevel = 1, 
    ), 
    pages = [
        "Home" => "index.md", 
        "Examples" => [
            "Transverse-field Ising model" => "ising.md",
        ], 
        "Types" => [
            "Operators" => "types/operators.md", 
            "Collections" => "types/collections.md", 
            "Labels" => "types/labels.md", 
        ], 
        "Functions" => [
            "Manipulations" => "functions/manipulations.md", 
            "Inspections" => "functions/inspections.md", 
            "Representations" => "functions/representations.md", 
            "Extensions" => "functions/extensions.md", 
        ], 
        "Development Guide" => "development.md", 
    ], 
    checkdocs = :public, 
)

deploydocs(
    repo = "github.com/frankwswang/Paulimorphic.jl.git", 
    devbranch = "dev", 
)
