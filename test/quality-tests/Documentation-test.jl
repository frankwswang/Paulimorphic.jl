using Test
using Documenter: DocMeta, doctest
using LinearAlgebra
using Paulimorphic
using Paulimorphic: Interface


@testset "Documentation Test" begin

@testset "`Docs.hasdoc` behavior check" begin
    @test !Docs.hasdoc(Paulimorphic, :Interface)
    @test !Docs.hasdoc(Paulimorphic, :MissingOr)
    @test !Docs.hasdoc(Paulimorphic, :BitUInteger)
    @test !Docs.hasdoc(Paulimorphic, :SameTypePair)
    @test Docs.hasdoc(LinearAlgebra, :AbstractVecOrMat)
end

@assert (DataType <: Type) && (UnionAll <: Type)

function appendHiddenAPI!(buffer::AbstractVector{<:Pair{<:AbstractString, <:Type}}, 
                          source::Module, exempt::IdSet{Interface}, 
                          prefix::AbstractString=(string∘nameof)(source))
    sourceName = nameof(source) |> string
    prefix == sourceName || (prefix *= '.' * sourceName)
    scope = names(source, all=true, imported=false, usings=false) |> unique!

    for objSym in scope
        ((objSym in (:eval, :include)) || startswith(string(objSym), '#')) && continue
        isdefined(source, objSym) || continue #> Defensive filter to avoid undefined object
        obj = getfield(source, objSym)
        objTyp = typeof(obj)

        if objTyp <: Interface && !(obj in exempt)
            if objTyp <: Module
                if parentmodule(obj) === source !== obj && nameof(obj) === objSym
                    #> This filters out module aliases, non-sub modules, and self module
                    appendHiddenAPI!(buffer, obj, exempt, prefix)
                end
            else
                if obj isa Union{Function, DataType}
                    #> This filters out the aliases that may inherit docstrings
                    nameof(obj) == objSym || continue
                end

                #> `Base.ispublic` correctly identifies whether the alias itself is public
                #> `Docs.hasdoc` correctly fetches the docstring attached to the alias
                if !Base.ispublic(source, objSym) && Docs.hasdoc(source, objSym)
                    push!(buffer, prefix * ".$objSym" => objTyp)
                end
            end
        end
    end

    nothing
end

function checkHiddenAPI(mod::Module, exempt::AbstractVector=[])
    exemptSet = IdSet{Interface}()

    for (i, item) in enumerate(exempt)
        if item isa Interface
            push!(exemptSet, item)
        else
            throw(ArgumentError("The $i-th element of `exempt` should be a $Interface."))
        end
    end

    objNameBuffer = Pair{String, Type}[]
    appendHiddenAPI!(objNameBuffer, mod, exemptSet)
    nHidden = length(objNameBuffer)

    if nHidden > 0
        sort!(objNameBuffer, by=first)
        message = "Found $nHidden documented object$(nHidden > 1 ? "s" : "") in $mod " * 
                  "that are neither exported nor explicitly marked by keyword `public`:\n"
        for pair in objNameBuffer
            str, type = pair
            message *= '[' * if type <: Function
                "Function"
            elseif type <: DataType
                "DataType"
            elseif type <: UnionAll
                "UnionAll"
            else
                "  Type  "
            end * ']' * " " * str * '\n'
        end

        throw(ErrorException(message))
    end

    true
end

@testset "Hidden API check" begin
    exempt = [Paulimorphic.absorbPhases!, Paulimorphic.getFrustrationInfoCore!, 
              Paulimorphic.pasteBits!, Paulimorphic.shiftBits!, Paulimorphic.stampBits!]
    @test checkHiddenAPI(Paulimorphic, exempt)
end

@testset "Docstring-based est" begin
    DocMeta.setdocmeta!(Paulimorphic, :DocTestSetup, :(using Paulimorphic); recursive=true)
    doctest(Paulimorphic)
end

end
