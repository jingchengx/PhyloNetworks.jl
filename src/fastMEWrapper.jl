using DelimitedFiles

# TODO: check names agree with fastME spec; add distance matrix check
function writeDistanceMatrix!(s::IO, D::Matrix{<:Real}, names::AbstractVector{String})
    write(s, string(size(D, 1)), "\n")
    for (i, col) in enumerate(eachcol(D))
        write(s, names[i], "\t")
        writedlm(s, transpose(col))
    end
end

function fastme(D::Matrix{<:Real}, names::AbstractVector{String})
    (path, io) = mktemp()
    tpath = tempname()
    writeDistanceMatrix!(io, D, names)
    close(io)
    fastmecmd = `fastme -m OLSME -n -i $path -o $tpath`
    run(fastmecmd)
    tree = readTopology(tpath)
end
