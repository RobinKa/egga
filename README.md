# E-Graph Geometric Algebra (EGGA)

[![PyPI](https://badge.fury.io/py/egga.svg)](https://badge.fury.io/py/egga)

Symbolic [Geometric Algebra](https://en.wikipedia.org/wiki/Geometric_algebra) with [E-Graphs](https://egraphs-good.github.io/)

Things you can do with this library

- Simplify expressions
- Prove equalities
- Solve for variables

Things that are supported

- Any signature
- Arbitrary number of basis vectors
- Symplectic Geometric Algebra (aka Weyl Algebras)
- Derivatives
- Add your own expression types and rules (with egglog)

Based on the [Python bindings](https://github.com/metadsl/egglog-python) for [egglog](https://github.com/egraphs-good/egglog)

## Setup

Supports Python 3.8 and higher.

`pip install egga`

## Usage

The first step is to create a `GeometricAlgebra` object with a given signature.
You can then use its basis vectors as well as functions exposed by it. Use the utility methods provided to do things like simplification and
equation solving. In some cases you might need to interface with egglog directly. Below are
some examples for common use-cases.

Simplification

```python
from egga.geometric_algebra import GeometricAlgebra
from egga.utils import simplify

ga = GeometricAlgebra(signature=[1.0, 1.0])
e_0, e_1 = ga.basis_vectors
e_01 = e_0 * e_1

# Build an expression to simplify
expr = e_01 * e_0 * ~e_01

# Prints Simplified: -e("0")
print("Simplified:", simplify(ga, expr))
```

Equation solving

```python
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

assert str(simplify(ga, x)) == str(ga.expr_cls.e("0"))
```

Equality check

```python
from egga.geometric_algebra import GeometricAlgebra
from egga.utils import check_equality

ga = GeometricAlgebra(signature=[1.0, 1.0])
e_0, e_1 = ga.basis_vectors
e_01 = e_0 * e_1

# Build an lhs to check for equality to an rhs
lhs = e_01 * e_01
rhs = ga.expr_cls.scalar_literal(-1.0)

assert check_equality(ga, lhs, rhs)
```

The [/examples](examples) as well as the [/tests](tests) directories contain more examples.

## List of expressions

### Operators

| Code         | Description                                                                           |
| ------------ | ------------------------------------------------------------------------------------- |
| `x_1 + x_2`  | Addition of x_1 and x_2                                                               |
| `x_1 - x_2`  | Subtraction of x_1 and x_2                                                            |
| `x_1 * x_2`  | Multiplication of x_1 and x_2 (aka the Geometric Product)                             |
| `x_1 ^ x_2`  | Wedge / exterior / outer product of x_1 and x_2                                       |
| `x_1 \| x_2` | Inner ("fat dot") product of x_1 and x_2                                              |
| `-x_1`       | Negation of x_1                                                                       |
| `~x_1`       | Reversion of x_1                                                                      |
| `x_1 ** x_2` | x_1 to the power of x_2                                                               |
| `x_1 / x_2`  | x_1 divided by x_2 (more generally, x_1 right-multiplied by the right inverse of x_2) |

### Functions

| Code                     | Description                                                                                          |
| ------------------------ | ---------------------------------------------------------------------------------------------------- |
| `inverse(x)`             | Right-multiplicative inverse of x                                                                    |
| `scalar(x)`              | Mark x as a scalar                                                                                   |
| `scalar_literal(f)`      | Create a scalar constant                                                                             |
| `scalar_variable(s)`     | Create a scalar variable                                                                             |
| `e(s)`                   | Basis vector                                                                                         |
| `e2(s_1, s_2)`           | Basis bivector                                                                                       |
| `e3(s_1, s_2, s_3)`      | Basis trivector                                                                                      |
| `variable(s)`            | Create a variable                                                                                    |
| `cos(x)`                 | Cos of x                                                                                             |
| `sin(x)`                 | Sin of x                                                                                             |
| `cosh(x)`                | Cosh of x                                                                                            |
| `sinh(x)`                | Sinh of x                                                                                            |
| `exp(x)`                 | Exponential function of x                                                                            |
| `grade(x)`               | Grade of x                                                                                           |
| `mix_grades(x_1, x_2)`   | Represents the mixture of two grades. If x_1 and x_2 are the same, this will be simplified to `x_1`. |
| `select_grade(x_1, x_2)` | Selects the grade x_2 part of x_1                                                                    |
| `abs(x)`                 | Absolute value of x                                                                                  |
| `rotor(x_1, x_2)`        | Shorthand for `exp(scalar_literal(-0.5) * scalar(x_2) * x_1)`                                        |
| `sandwich(x_1, x_2)`     | Shorthand for `x_1 * x_2 * ~x_1`                                                                     |
| `diff(x_1, x_2)`         | Derivative of x_1 with respect to x_2                                                                |

### Unsupported but exists, might or might not work

| Code                 | Description                    |
| -------------------- | ------------------------------ |
| `boolean(x)`         | Mark x as a boolean            |
| `x_1.equal(x_2)`     | Whether x_1 equals x_2         |
| `x_1.not_equal(x_2)` | Whether x_1 does not equal x_2 |

## Caveats

- Egraphs are bad with associativity (combined with commutativity?) so things can blow up
- Most operations aren't "fully" implemented (eg. `pow` only supports powers of two right now)

## Contributing

Code contributions as well as suggestions and comments about things that don't work yet are appreciated.
You can reach me by email at `tora@warlock.ai` or in the [Bivector Discord](https://discord.gg/vGY6pPk).
