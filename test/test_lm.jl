## Test of PhyloNetworklm

using PhyloNetworks
using GLM
include("../src/types.jl")
include("../src/functions.jl")
include("../src/traits.jl")

tree= "(A,((B,#H1),(C,(D)#H1)));"
net=readTopologyLevel1(tree)
printEdges(net)

# Re-root the tree so that it matches my example
root!(net, "A")
printEdges(net)
directEdges!(net) ## I am forced to do thi step here, because root sends a net with net.isRooted = false. Expected behavior ? 
preorder!(net)

# Data : simulate function
params = paramsBM(10, 0.9)
sim = simulate(net, params)
Y = extractSimulateTips(sim, net)
X = ones(4, 1)

# "Ancestral state reconstruction"
fit = phyloNetworklm(Y, X, net)
coef(fit.lm)
sum(residuals(fit.lm).^2)

fitNaive = phyloNetworklmNaive(Y, X, net)

