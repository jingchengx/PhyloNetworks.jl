const SSM = StatisticalSubstitutionModel

# Collect the relevant likelihoods for calculation: in this case,
# across all sites
# returns an array such that array[treeidx,rateidx][siteidx] = (gs,f,rateidx)
function collect_liks(obj::SSM, edgenum::Integer, t::Integer,
                      ri::Integer) # rate index
    ns = obj.nsites
    k = nstates(obj.model)
    tree = obj.displayedtree[t]
    preorder!(tree)
    # important that nodes and edges from the tree, not the
    # network: in the network we may have sister edges that are not
    # present in the tree
    b = tree.edge[edgenum]; v = getParent(b); u = getChild(b);
    # let edge be b := v -> u, and sibling edge be d
    # log(backlik) + sum(log(dirlik)) for node v and out edges of v across sites
    liks = Vector{Tuple{Vector{Float64},Vector{Float64},Int}}(undef, ns)
    # forwardlik for node u, across sites
    # avoid repated memory allocation
    forwardliktemp = Array{Float64}(undef, k, obj.net.numNodes)
    dirliktemp = Array{Float64}(undef, k, obj.net.numEdges)
    backwardliktemp = Array{Float64}(undef, k, obj.net.numNodes)

    for si in 1:ns
        # set up forward/directional likelihoood
        discrete_corelikelihood_trait!(obj, t, si, ri,
                                       forwardliktemp, dirliktemp)
        # set up backward likelihood
        discrete_backwardlikelihood_tree!(obj, t, si, ri,
                                          backwardliktemp)
        for e in v.edge
            if e != b && v == getParent(e) # e is sister edg of b
                @views backwardliktemp[:,v.number] .+= dirliktemp[:, e.number]
            end
        end
        @views liks[si] = (forwardliktemp[ :, u.number],
                           backwardliktemp[ :, v.number],
                           ri)
    end

    return liks
end

# PRECONDITIONS: logtrans updated, edges directed
function optimize_single_branch(obj::SSM, edgenum::Integer)
    # TODO: sanity checks, edgnum valid, etc.

    ntrees = length(obj.displayedtree)
    nrates = length(obj.ratemodel.ratemultiplier)
    ns = obj.nsites
    k = nstates(obj.model)
    liks = [ collect_liks(obj, edgenum, it, ir)
             for it=1:ntrees, ir=1:nrates ]

    # construct objective function, gradient, Hessian (possibly)
    function loglik(t::Float64)
        # lp[ri] = transition prob matrix for ri-th rate
        lp = map(rate -> log.(P(obj.model, rate * t)),
                 obj.ratemodel.ratemultiplier)

        # map each site to the loglik
        trslglks = map(slktp ->
                        map((gs,f,ri)::Tuple -> logsumexp(lp[ri] .+ (gs .+ f')),slktp),
                        liks)
        # add siteweight to each (tree,rate) combination
        ismissing(obj.siteweight) ||
            (slks -> slks .* obj.siteweight).(trslglks)
        # trslglks[tree,rate][site] = loglik for site
        trlglks = map(sum, trslglks)
        # trlglks[tree,rate] = loglik over all sites, for fiex tree and rate
        tlglks = mapslices(rlglks -> reduce(logaddexp, rlglks, init=-Inf) - log(nrates),
                           trlglks, dims = [2])
        dropdims(tlglks, dims = 2)
        # tlglks[tree] = loglik for tree
        # aggregate over trees
        loglik = logsumexp(tlglks .+ obj.priorltw)

        return loglik
    end

    return loglik
end
