using GeodesicLearning

## energy function
f(x) = sum( 3*(x .- 5).^2 )

pop = initialize_pop(f, 20, dim = 3) # initialize_pop with 20 points 
##
x = GeodesicLearning.least_squares(pop)
entropy_invert(pop, log(20)/100, x)

