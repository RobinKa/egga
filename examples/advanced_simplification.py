from egga.geometric_algebra import GeometricAlgebra
from egga.utils import RunScheduledOptions, simplify

# Assign a large cost to diff expressions so they will get simplified
# to something else.
ga = GeometricAlgebra(signature=[-1.0, 1.0], costs={"diff": 1_000})
e_0, e_1 = ga.basis_vectors
e_01 = e_0 * e_1

E = ga.expr_cls

# Build an expression to simplify:
# d/dphi exp(-phi/2 e_01)
phi = E.scalar_variable("phi")
rotor = E.rotor(e_01, phi)
expr = E.diff(rotor, phi)


# print the current simplificiation at each step, and optionally wait for user
# input and render the egraph to pdf.
# Here are the unique outputs this will print over time:
# diff(rotor(e2("0", "1"), scalar_variable("phi")), scalar_variable("phi"))
# rotor(e2("0", "1"), variable("phi")) * (
#     ((scalar_literal(0.0) * e("1")) * (variable("phi") * scalar_literal(-0.5)))
#     + (e2("0", "1") * ((scalar_literal(1.0) * scalar_literal(-0.5)) + (variable("phi") * scalar_literal(0.0))))
# )
# (rotor(e2("0", "1"), variable("phi")) * scalar_literal(0.5)) * e2("1", "0")
def on_step():
    print("Simplified:", ga.egraph.extract(expr))
    # ga.egraph.graphviz.render("render", quiet=True)
    # input()


# With 6 steps this hangs, with 5 it seems okay.
# Options are figured out by trial and error:
# - limits (how many iterations to run)
# - fast_count: how many fast ruleset steps to run per iteration
# - full_interval: run the full ruleset every full_interval iterations,
#                  otherwise the medium ruleset will run which is a subset
simplified = simplify(
    ga,
    expr,
    options=RunScheduledOptions(on_step=on_step, limit=5),
)

print(
    "Final simplified:",
    simplified,
)
