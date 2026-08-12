using Documenter: DocMeta, makedocs
using Paulimorphic

DocMeta.setdocmeta!(Paulimorphic, :DocTestSetup, :(using Paulimorphic); recursive=true)

makedocs(
    sitename = "Paulimorphic",
    format = Documenter.HTML(),
    modules = [Paulimorphic], 
    checkdocs = :all,
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
