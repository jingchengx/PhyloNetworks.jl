using DelimitedFiles

function writeDistanceMatrix!(s::IO, D::Matrix{<:Real}, names::AbstractVector{String})
    check_distance_matrix(D)
    write(s, string(size(D, 1)), "\n")
    for (i, col) in enumerate(eachcol(D))
        write(s, names[i], "\t")
        writedlm(s, transpose(col))
    end
end

function fastME(D::Matrix{<:Real}, names::AbstractVector{String})
    global fastme
    (path, io) = mktemp()
    tpath = tempname()
    writeDistanceMatrix!(io, D, names)
    close(io)
    fastmecmd = `$fastme -m OLSME -n -i $path -o $tpath`
    run(fastmecmd)
    tree = readTopology(tpath)
end
