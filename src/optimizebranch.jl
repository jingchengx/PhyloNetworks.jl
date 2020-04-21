const SSM = StatisticalSubstitutionModel

# Collect the relevant likelihoods for calculation: in this case,
# across all sites
# returns an tuple (ri, arr, lik)
# when edgenum is in tree, then lik = nothing, and arr[site] = (f,gs)
# where f is the forwardlik at site (of size k), gs is the product of
# backwardlik and directlik
# when edgenum is not in the tree, then ri = 0, arr = nothing, lik is the per
# site likelihoods
function collect_liks(obj::SSM, edgenum::Integer, t::Integer,
                      ri::Integer) # rate index
    ns = obj.nsites
    k = nstates(obj.model)
    tree = obj.displayedtree[t]

    # let edge be b := v -> u, and sibling edge be d
    result = Tuple{Int,
        Union{Vector{Tuple{Vector{Float64}, Vector{Float64}}},Nothing},
        Union{Vector{Float64},Nothing}}

    # TODO optimize this part: check if edge in tree?
    ind = findfirst(x -> x.number == edgenum, tree.edge)
    if isnothing(ind)
        siteliks = map(si -> discrete_corelikelihood_trait!(obj,t,si,ri)[1],
                       1:ns)
        return (0,nothing,siteliks)
    end

    # important that nodes and edges from the tree, not the
    # network: in the network we may have sister edges that are not
    # present in the tree
    preorder!(tree)
    b = tree.edge[ind]; v = getParent(b); u = getChild(b);

    liks = Vector{Tuple{Vector{Float64}, Vector{Float64}}}(undef, ns)
    for si in 1:ns
        fwdlik = Array{Float64}(undef, k, obj.net.numNodes)
        dirlik = Array{Float64}(undef, k, obj.net.numEdges)
        bkwlik = Array{Float64}(undef, k, obj.net.numNodes)
        discrete_corelikelihood_trait!(obj, t, si, ri, fwdlik, dirlik)
        discrete_backwardlikelihood_trait!(obj, t, si, ri, bkwlik, dirlik)
        f = copy(fwdlik[ :, u.number])
        gs = copy(bkwlik[ :, v.number])
        for e in v.edge
            if e != b && v == getParent(e) # e is sister edg of b
                gs .+= dirlik[:, e.number]
            end
        end
        liks[si] = (f, gs)
    end
    return (ri,liks,nothing)
end

# PRECONDITIONS: logtrans updated, edges directed
# return a loglik(t) and derivative at t where t is the branch length
function single_branch_loglik_objective(obj::SSM, edgenum::Integer)
    # TODO: sanity checks, edgnum valid, etc.

    # DEBUG: update logtrans and direct trees in this function, to be
    # removed in final version
    # directEdges!(obj.net)
    # update_logtrans(obj)

    ntrees = length(obj.displayedtree)
    rates = obj.ratemodel.ratemultiplier
    nrates = length(rates)
    qmat = Q(obj.model)

    liks = [ collect_liks(obj, edgenum, it, ir)
             for it=1:ntrees, ir=1:nrates ]

    function objective(t::Float64)
        # lp[ri] = transition prob matrix for ri-th rate
        lp = map(rate -> log.(P(obj.model, rate * t)),
                 obj.ratemodel.ratemultiplier)
        pq = map(rate -> rate * qmat * P(obj.model, rate * t),
                 obj.ratemodel.ratemultiplier)
        pqq = map(rate -> rate^2 * qmat^2 * P(obj.model, rate * t),
                  obj.ratemodel.ratemultiplier)

        # taking care of siteweight
        wsum = sum
        if !isnothing(obj.siteweight)
            wsum = (vals -> sum(vals .* obj.siteweight))
        end

        # aggregate a [tree,rate] array with the prior prob
        function mix(ls::Array{Float64, 2})
            ls = copy(ls)
            ls .*= exp.(obj.priorltw)
            ls ./= nrates
            sum(ls)
        end

        # same as mix, but on log scale
        function lmix(lls::Array{Float64, 2})
            lls = copy(lls)
            lls .+= obj.priorltw
            lls .-= log(nrates)
            logsumexp(lls)
        end

        # sll: array of site loglikelihood (for fixed tree & rate)
        # slls: array of sll's
        # i.e. slls[tree,rate][site]=loglik for site
        # stup: array of (f,gs,ri) tuples
        slls = map((ri,stup,lik)::Tuple ->
                   ri == 0 ? lik :
                   map((f,gs)::Tuple ->
                       logsumexp(lp[ri] .+ (gs .+ f')),
                       stup), liks)
        # since we need to integrate over tree and rate first, need to "transpose" slls, strlls = [Site][Tree, Rate] -> LogLikS
        # TODO this copying operation could be wasteful
        strlls = [ [slls[tree,rate][site] for tree = 1:ntrees, rate = 1:nrates] for site = 1:obj.nsites]
        # slls[site]=loglik for site (integrated over tree and rates)
        slls = map(lmix, strlls)
        loglik = wsum(slls)

        # TODO cache exp.(gs) and exp.(f)
        # gradient
        # slikd[tree,rate][site]=deriv of likelihood (NOT loglik)
        slikd = map((ri,stup,lik)::Tuple ->
                    ri == 0 ? zeros(length(lik)) :
                    map((f,gs)::Tuple ->
                        exp.(gs)' * pq[ri] * exp.(f),
                        stup), liks)
        # similarly, now transpose slikd
        # strdls = [site][tree, rate] derivative of *likelihoods* (not loglik)
        strdls = [ [slikd[tree,rate][site] for tree = 1:ntrees, rate = 1:nrates] for site = 1:obj.nsites]
        # sdlls = [site] derivative of logliks
        sdlls = map(mix, strdls) ./ exp.(slls)
        grad = wsum(sdlls)

        # # Hessian, the following implementation is wrong
        # slikdd = map((ri,stup,lik)::Tuple ->
        #              ri == 0 ? zeros(length(lik)) :
        #              map((f,gs)::Tuple ->
        #                  exp.(gs)' * pqq[ri] * exp.(f),
        #                  stup), liks)
        # tlldd = map((sldd,sld,sll)::Tuple ->
        #             wsum((sldd ./ exp.(sll)) - (sld .^ 2 ./ exp.(2 .* sll))),
        #             zip(slikdd,slikd,slls))
        # hessian = mix((tlldd .+ tlld .^2) .* exp.(tll .- loglik)) - grad^2

        # return (loglik,grad,hessian)

        return (loglik,grad)
    end

    return objective
end

function optimizeSBL_LiNC!(obj::SSM, edge::Edge, ftolRel::Float64,
                         ftolAbs::Float64, xtolRel::Float64, xtolAbs::Float64,
                         maxeval=1000::Int)
    len = edge.length
    startlik = discrete_corelikelihood!(obj)
    fun = single_branch_loglik_objective(obj, edge.number)
    function wrapper(t::Vector, grad::Vector)
        (l,g) = fun(t[1])
        length(grad) > 0 && (grad[1] = g)
        # println("g = ", g)
        return l
    end
    @debug "start len : $len, start loglik : $startlik"

    optBL = NLopt.Opt(:LD_SLSQP, 1)
    NLopt.ftol_rel!(optBL,ftolRel) # relative criterion
    NLopt.ftol_abs!(optBL,ftolAbs) # absolute criterion
    NLopt.xtol_rel!(optBL,xtolRel)
    NLopt.xtol_abs!(optBL,xtolAbs)
    NLopt.maxeval!(optBL, maxeval) # max number of iterations
    NLopt.max_objective!(optBL, wrapper)
    optBL.lower_bounds = 0
    fmax, xmax, ret = NLopt.optimize(optBL, [len]) # get lengths in order of edges vector
    @debug "BL: got $(round(fmax, digits=5)) at BL = $(round.(xmax, digits=5)) after $(optBL.numevals) iterations (return code $(ret))"
    if startlik < fmax
        @debug "taking a step"
        setLength!(edge, xmax[1])
        obj.loglik = fmax
    end
    return edge
end

# function optimize_branch_newton(obj::SSM, edgenum::Int)
#     fun = single_branch_loglik_objective(obj, edgenum)
#     f = x::Vector -> -fun(x[1])[1]
#     function g!(g,x::Vector)
#         g = [-fun(x[1])[2]]
#     end
#     function h!(h,x::Vector)
#         h = [-fun(x[1])[3]]
#     end

#     len = getEdge(edgenum, obj.net).length
#     x0 = [len]
#     df = TwiceDifferentiable(f, g!, h!, x0)

#     lx = [0.]; ux = [Inf];
#     dfc = TwiceDifferentiableConstraints(lx, ux)

#     res = Optim.optimize(df, dfc, x0, IPNewton())
# end
