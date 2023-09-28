from egga.geometric_algebra import GeometricAlgebra
from egga.utils import simplify

ga = GeometricAlgebra(signature=[1.0, 1.0])
e_0, e_1 = ga.basis_vectors
e_01 = e_0 * e_1

# Build an expression to simplify
expr = e_01 * e_0 * ~e_01

# Prints Simplified: -e("0")
print("Simplified:", simplify(ga, expr))
