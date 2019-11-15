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

        # map each site to the loglik
        # trslglks[tree,rate][site]=loglik for site
        trslglks = map(sitevectup ->
                       map((f,gs,ri)::Tuple ->
                           logsumexp(lp[ri] .+ (gs .+ f')),
                           sitevectup),
                        liks)
        # add siteweight to each (tree,rate) combination
        ismissing(obj.siteweight) ||
            (slks -> slks .* obj.siteweight).(trslglks)
        # trlglks[tree,rate]=loglik for tree and rate
        trlglks = map(sum, trslglks)
        # multiply in the mixutre probabilities
        trlglks .+= obj.priorltw
        trlglks .-= log(nrates)
        loglik = reduce(logaddexp, trlglks, init=-Inf)

        # gradient
        # grads[tree, rate](t) = gradient of trlglks[tree,rate](t)
        # TODO cache exp.(gs) and exp.(f)
        grads_site_numerator = map(sitevectup ->
                                   map((gs,f,ri)::Tuple ->
                                       exp.(gs)' * pq[ri] * exp.(f),
                                       sitevectup),
                                   liks)
        ismissing(obj.siteweight) ||
            (slks -> slks .* obj.siteweight).(grads_site_numerator)
        grads = map((gsnum, siteveclik)::Tuple ->
                    sum(gsnum ./ exp.(siteveclik)),
                    zip(grads_site_numerator,trslglks))
        grads .*= exp.(trlglks)
        grads .*= exp.(obj.priorltw)
        grads ./= nrates
        grad = sum(grads) / exp(loglik)

        return (loglik,grad)
    end

    return objective
end
