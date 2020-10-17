module GeodesicLearning
export mean, mean_energy, initialize_pop, algorithm_step, entropy_invert


using Statistics
using Random
using Roots


struct Population
    points::Array{Vector} # vector of θ equal to the size of the population
    energies::Array{Float64} # vector of ε
    log_weights::Array{Float64} # a vector of log weights
end

# initialize Population
Population(points, F::Function) = Population(points, F.(points), zeros(length(points)))


# the partition function
Z(pop::Population) = sum(exp(lw) for lw in pop.log_weights)
# Return the mean θ vector
mean(pop::Population; z = Z(pop)) = 
    sum(exp(lw) .* p for (p,lw) in zip(pop.points, pop.log_weights)) ./ z
# Mean energy
mean_energy(pop::Population; z = Z(pop)) = 
    sum(exp(lw) .* e for (e,lw) in zip(pop.energies, pop.log_weights)) ./ z
# Variance matrix of θ
var(pop::Population;  z = Z(pop), m = mean(pop, z = z)) = 
    sum(exp(lw) .* kron( (p - m) , (p - m)')  for  (p,lw) in zip(pop.points, pop.log_weights)) ./ z
#Variance of free energy
var_free_energy(pop::Population, x; z = Z(pop), m = mean(pop), me = mean_energy(pop,x)) = 
    sum(exp(lw) * ( x'*(p .- m) - (e-me) )^2  for  (p,e,lw) in zip(pop.points, pop.energies, pop.log_weights)) ./ z
# cov(φ_i, θ_i)
covar_energy(pop::Population; 
    z = Z(pop), m = mean(pop, z = z), me = mean_energy(pop::Population; z = z))  = 
    sum(exp(lw) .* (p .- m) .* (e - me) for (p,e,lw) in zip(pop.points, pop.energies, pop.log_weights)) ./ z


function initialize_pop(F::Function, n::Int64; dim = 1, mean = zeros(dim), var = 1.0)
    points = [randn(dim) for _ in 1:n]
    energies = F.(points)
    log_weights = zeros(n)
    Population(points,energies,log_weights)
end

function least_squares(pop; z = Z(pop), m = mean(pop; z = z))
    var(pop, z = z, m = m) \ covar_energy(pop, z = z, m = m)
end

function generate_δ(pop; z= Z(pop), m = mean(pop; z = z))
    sum( randn() .* (p .- m) * exp(lw/2)  for (p,lw) in zip(pop.points,pop.log_weights)) ./ sqrt(z)
end

function algorithm_step(pop::Population, F::Function; new_points = 20, s0 = log(new_points)/2)
    # generational algorithm
    x = least_squares(pop)
    β0 = entropy_invert(pop, s0, -x) # doubles the distance traveled by mean
    evolve!(pop, β0, -x) 
# mean and variance are now correct except for factor related to variance inflation
    z = Z(pop)
    m = mean(pop,z = z)
    new_points = [m .+ generate_δ(pop, z = z; m = m) for ii in 1:new_points]
# new points generated
    x = least_squares(pop, z = z, m = m)
    β1 = entropy_invert(pop ,s1, x) # shrinks to collect variance information
    qop = evolve(pop, β1, x) 
    y = (var(qop) * x) ./ β1
# estimate of minimum achieved
    
# add the extra variance
    Population(new_points, F)

end

function free_energies(pop::Population, x)
    [x'*p - e for (p,e) in zip(pop.points,pop.energies)]
end

function entropy_fun(log_weights, free_energies) # p = exp(l) - log z
    function s(β::Float64)
        z = 0.0
        s = 0.0
        for (lw,e) in zip(log_weights,free_energies)
            s += exp(lw - β * e) * (lw - β * e)
            z += exp(lw - β * e)
        end
        log(z) - s/z #-p logp#
    end
    return s
end

function find_bracket(s, s0) # s is a monotonically decreasing function of β
    β0 = 0.0
    β1 = 1.0
    while s(β1) > s0
        β0 = β1
        β1 = 2*β1
    end
    (β0,β1)
end

function entropy_invert(p::Population, s0::Float64, x)
    #free_energies = free_energies(p, x)
    free_energies = [x'*p - e for (p,e) in zip(p.points,p.energies)]
    s = entropy_fun(p.log_weights,free_energies)
    start = find_bracket(s, s0)
    find_zero(β -> exp(s(β)) - exp(s0), start)
end

## ``Natural gradient" step

function evolve!(pop::Population, β::Float64, x) where dim
    for (ii,p) in enumerate(pop.points)
        pop.log_weights[ii] += β * (x'*pop.position- pop.energy)
    end
end

function evolve(pop::Population, β::Float64, x) where dim
    qop = pop
    evolve!(qop, β, x)
    return qop
end


function recombine_id!(pop, number ; new_log_weight = 0, old_log_weight = -10)
end

function get_gradient(pop)
end


end # module
