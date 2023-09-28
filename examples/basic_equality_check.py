from egga.geometric_algebra import GeometricAlgebra
from egga.utils import check_equality

ga = GeometricAlgebra(signature=[1.0, 1.0])
e_0, e_1 = ga.basis_vectors
e_01 = e_0 * e_1

# Build an lhs to check for equality to an rhs
lhs = e_01 * e_01
rhs = ga.expr_cls.scalar_literal(-1.0)

assert check_equality(ga, lhs, rhs)
