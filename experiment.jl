using PhyloNetworks
using PhyloPlots
using Symbolics
P = PhyloNetworks

using Debugger

function initsymbols!(net::HybridNetwork)
    for e in net.edge
        if e.hybrid
            e.length = 0
        else
            s = Symbol("t$(e.number)")
            e.length = (@variables $s)[1]
        end
    end
    for v in net.hybrid
        edges = v.edge
        major = P.getMajorParentEdge(v)
        s = Symbol("g$(major.number)")
        sg = (@variables $s)[1]
        major.gamma = sg
        minor = P.getMinorParentEdge(v)
        minor.gamma = 1 - sg
    end
end

function initparams!(net::HybridNetwork)
    for e in net.edge
        if e.hybrid
            e.length = 0
        else
            e.length = 1.
        end
    end
    for v in net.hybrid
        edges = v.edge
        major = P.getMajorParentEdge(v)
        sg = 0.5
        major.gamma = sg
        minor = P.getMinorParentEdge(v)
        minor.gamma = 1 - sg
    end
end

function fit_caterpillar(M::Matrix{Real})
    n = size(M, 1)
    r(i) = mod(i, 1:n)
    pendants = [ (M[i, r(i+1)] + M[i, r(i-1)] - M[r(i-1), r(i+1)]) / 2 for i = 1:n ]
    internals = [ (M[i, i+2] + M[i+1, i+3] - M[i, i+1] - M[i+2, i+3]) / 2 for i = 1:(n-3) ]
    return (pendants, internals)
end

function fit5sunlet(M::Matrix{Real})
    # M ordering {a,b,c,d,h}: start with parent of hybrid edge with gamma, go around, end with hybrid node
    pd, it = fit_caterpillar(M[1:4, 1:4])
    gamma = ((M[5,3] - M[5,2] - pd[3] + pd[2]) / it[1] + 1) / 2
    lh = (M[1,5] + M[4,5] - M[1,4]) / 2
    i12 = (M[1,4] + M[2,5] - M[1,5] - M[2,4]) / (2 * gamma)
    i34 = (M[1,4] + M[3,5] - M[1,3] - M[4,5]) / (2 * (1-gamma))
    return (gamma, lh, i12, it[1], i34, pd[1] - i12, pd[2], pd[3], pd[4] - i34)
end

# # test fit_caterpillar
# caterpillar = P.readTopology("((a:1,b:1):10,c:2,(d:1,e:1):1);")
# m = pairwiseTaxonDistanceMatrix(caterpillar)
# fit_caterpillar(m)

net = P.readTopology("(a,(b,((c,(d,#H2)),(#H1)#H2)),(h)#H1);")

initparams!(net)
M = pairwiseTaxonDistanceMatrix(net) 
r = fit5sunlet(M)

sunlet = P.readTopology("(a,(b,(c,(d,#H1))),(h)#H1);")
initparams!(sunlet)
for (i, e) in enumerate(sunlet.edge[[9,8,7,6,1,2,3,4]])
    e.length = r[i+1]
end
P.setGamma!(sunlet.edge[10], r[1])
d = pairwiseTaxonDistanceMatrix(sunlet)

M == d

# initsymbols!(net)
# M = pairwiseTaxonDistanceMatrix(net) 
# r = fit5sunlet(M)
# rs = simplify.(r, expand=true)

# simplify((M[5,3] - M[5,2]) / M[2,3], expand=true)
# simplify((M[1,4] + M[3,5] - M[1,3] - M[4,5]), expand=true)

