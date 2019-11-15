const SSM = StatisticalSubstitutionModel

# Collect the relevant likelihoods for calculation: in this case,
# across all sites
# returns an array such that array[treeidx,ri(rateidx)][siteidx] = (f,gs,ri)
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
    ftemp = Array{Float64}(undef, k, obj.net.numNodes)
    stemp = Array{Float64}(undef, k, obj.net.numEdges)
    gstemp = Array{Float64}(undef, k, obj.net.numNodes)

    for si in 1:ns
        # set up forward/directional likelihoood
        discrete_corelikelihood_trait!(obj, t, si, ri, ftemp, stemp)
        # set up backward likelihood
        discrete_backwardlikelihood_tree!(obj, t, si, ri, gstemp)
        for e in v.edge
            if e != b && v == getParent(e) # e is sister edg of b
                @views gstemp[:,v.number] .+= stemp[:, e.number]
            end
        end
        @views let f = ftemp[ :, u.number], gs=gstemp[ :, v.number]
            liks[si] = (f, gs, ri)
        end
    end
    return liks
end

# PRECONDITIONS: logtrans updated, edges directed
# return a loglik(t) and derivative at t where t is the branch length
function single_branch_loglik_objective(obj::SSM, edgenum::Integer)
    # TODO: sanity checks, edgnum valid, etc.

    ntrees = length(obj.displayedtree)
    rates = obj.ratemodel.ratemultiplier
    nrates = length(rates)
    ns = obj.nsites
    k = nstates(obj.model)
    qmat = Q(obj.model)
    liks = [ collect_liks(obj, edgenum, it, ir)
             for it=1:ntrees, ir=1:nrates ]

    # loglikelihood objective function
    function objective(t::Float64)
        # lp[ri] = transition prob matrix for ri-th rate
        lp = map(rate -> log.(P(obj.model, rate * t)),
                 obj.ratemodel.ratemultiplier)
        pq = map(rate -> rate * qmat * P(obj.model, rate * t),
                 obj.ratemodel.ratemultiplier)

        # taking care of siteweight
        wsum = sum
        if !ismissing(obj.siteweight)
            wsum = (vals -> sum(vals .* obj.siteweight))
        end

        # sll: array of site loglikelihood (for fixed tree & rate)
        # slls: array of sll's
        # i.e. slls[tree,rate][site]=loglik for site
        # stup: array of (f,gs,ri) tuples
        slls = map(stup -> map((f,gs,ri)::Tuple ->
                           logsumexp(lp[ri] .+ (gs .+ f')),
                           stup), liks)
        # tll: loglikelihood for a (tree,rate) pair
        # tlls: array of tll's
        # tlls[tree,rate]=loglik for tree and rate
        tlls = map(wsum, slls)
        # multiply in the mixutre probabilities
        tlls .+= obj.priorltw
        tlls .-= log(nrates)
        loglik = reduce(logaddexp, tlls, init=-Inf)

        # gradient
        # grads[tree, rate](t) = gradient of tlls[tree,rate](t)
        # TODO cache exp.(gs) and exp.(f)
        # slikd[tree,rate][site]=deriv of likelihood (NOT loglik)
        slikd = map(stup -> map((gs,f,ri)::Tuple ->
                        exp.(gs)' * pq[ri] * exp.(f),
                        stup), liks)
        grads = map((sld, sll)::Tuple ->
                    wsum(sld ./ exp.(sll)),
                    zip(slikd,slls))
        grads .*= exp.(tlls)
        grads .*= exp.(obj.priorltw)
        grads ./= nrates
        grad = sum(grads) / exp(loglik)

        return (loglik,grad)
    end

    return objective
end
