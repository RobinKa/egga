from egglog import union

from egga.geometric_algebra import GeometricAlgebra
from egga.utils import simplify

# Pass eq_solve=True to enable the equation solving rules
ga = GeometricAlgebra(
    signature=[1.0, 1.0],
    eq_solve=True,
)

e_0, e_1 = ga.basis_vectors
e_01 = e_0 * e_1

# Solve e_01 * x * ~e_01 = e_0 for x
x = ga.expr_cls.variable("x")
lhs = e_01 * x * ~e_01
rhs = -e_0

# Make LHS equal to RHS
ga.egraph.register(union(lhs).with_(rhs))

assert str(simplify(ga, x)) == str(ga.expr_cls.e("0"))
