from egglog import union

from egga.geometric_algebra import GeometricAlgebra
from egga.utils import simplify

# Pass eq_solve=True to enable the equation solving rules.
# Add a cost to variable to it gets rewritten to something else.
ga = GeometricAlgebra(signature=[1.0, 1.0], eq_solve=True, costs={"variable": 1_000})

e_0, e_1 = ga.basis_vectors
e_01 = e_0 * e_1

# Solve e_01 * x * ~e_01 = e_0 for x
x = ga.expr_cls.variable("x")
lhs = e_01 * x * ~e_01
rhs = -e_0

# Make LHS equal to RHS
ga.egraph.register(union(lhs).with_(rhs))

solved = simplify(ga, x)
print("X:", solved)
assert str(solved) == str(ga.expr_cls.e("0"))
