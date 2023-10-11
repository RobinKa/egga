from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Type

from egglog import (
    EGraph,
    Expr,
    Fact,
    String,
    StringLike,
    egraph,
    eq,
    f64,
    f64Like,
    i64,  # noqa: F401
    i64Like,
    rewrite,
    rule,
    union,
    var,
    vars_,
)
from egglog import (
    egraph as egglog_egraph,
)

from egga.expression import Expression

birewrite = egraph.birewrite


def _close(a: f64Like, b: f64Like, eps: float = 1e-3):
    diff = a - b
    return diff * diff < eps * eps


def _not_close(a: f64Like, b: f64Like, eps: float = 1e-3):
    diff = a - b
    return diff * diff >= eps * eps


@dataclass
class GeometricAlgebraRulesets:
    full: egglog_egraph.Ruleset
    medium: egglog_egraph.Ruleset
    fast: egglog_egraph.Ruleset


def _maybe_inverse(
    m: type[Expression], x: Expression, dims: int
) -> Tuple[List[Expression], List[Fact], Expression]:
    clifford_conjugation = m.clifford_conjugation
    grade_involution = m.grade_involution
    scalar_literal = m.scalar_literal
    select_grade = m.select_grade

    x_conj = clifford_conjugation(x)
    x_x_conj = x * x_conj
    x_conj_rev_x_x_conj = x_conj * ~x_x_conj
    x_x_conj_rev_x_x_conj = x * x_conj_rev_x_x_conj

    if dims == 0:
        numerator = scalar_literal(1.0)
    elif dims == 1:
        numerator = grade_involution(x)
    elif dims == 2:
        numerator = x_conj
    elif dims == 3:
        numerator = x_conj * ~x_x_conj
    elif dims == 4:
        numerator = x_conj * (
            x_x_conj
            - scalar_literal(2.0)
            # Invert sign of grade 3 and 4 parts
            # TODO: is there a more efficient way?
            * (
                select_grade(x_x_conj, scalar_literal(3.0))
                + select_grade(x_x_conj, scalar_literal(4.0))
            )
        )
    elif dims == 5:
        numerator = x_conj_rev_x_x_conj * (
            x_x_conj_rev_x_x_conj
            - scalar_literal(2.0)
            # Invert sign of grade 1 and 4 parts
            # TODO: is there a more efficient way?
            * (
                select_grade(
                    x_x_conj_rev_x_x_conj,
                    scalar_literal(1.0),
                )
                + select_grade(
                    x_x_conj_rev_x_x_conj,
                    scalar_literal(4.0),
                )
            )
        )
    else:
        raise NotImplementedError("Unreachable")

    denominator = x * numerator
    dependencies = [denominator]

    scalar_divisor = var("scalar_divisor", f64)
    has_inverse = [
        eq(denominator).to(scalar_literal(scalar_divisor)),
        _not_close(scalar_divisor, 0.0),
    ]
    inverse_x = numerator * (scalar_literal(f64(1.0) / scalar_divisor))

    return dependencies, has_inverse, inverse_x


@dataclass
class GeometricAlgebra:
    egraph: EGraph
    expr_cls: Type[Expression]
    rulesets: GeometricAlgebraRulesets
    signature: Tuple[float]

    @property
    def basis_vectors(self):
        return [self.expr_cls.e(str(i)) for i in range(len(self.signature))]

    @property
    def symplectic_dual_basis_vectors(self):
        return [self.expr_cls.e(f"{i}*") for i in range(len(self.signature))]

    def __init__(
        self,
        signature: Tuple[float],
        symplectic=False,
        eq_solve=False,
        costs: Optional[Dict[str, int]] = None,
        full_inverse=False,
    ):
        if costs is None:
            costs = {}

        egraph = EGraph()

        @egraph.class_
        class MathExpr(Expr):
            @egraph.method(cost=costs.get("equal"))
            def equal(self, other: MathExpr) -> MathExpr:
                ...

            @egraph.method(cost=costs.get("not_equal"))
            def not_equal(self, other: MathExpr) -> MathExpr:
                ...

            @egraph.method(cost=costs.get("__add__"))
            def __add__(self, other: MathExpr) -> MathExpr:
                ...

            @egraph.method(cost=costs.get("__sub__"))
            def __sub__(self, other: MathExpr) -> MathExpr:
                ...

            @egraph.method(cost=costs.get("__mul__"))
            def __mul__(self, other: MathExpr) -> MathExpr:
                ...

            @egraph.method(cost=costs.get("__neg__"))
            def __neg__(self) -> MathExpr:
                ...

            @egraph.method(cost=costs.get("__invert__"))
            def __invert__(self) -> MathExpr:
                ...

            @egraph.method(cost=costs.get("__pow__"))
            def __pow__(self, other: MathExpr) -> MathExpr:
                ...

            @egraph.method(cost=costs.get("__truediv__"))
            def __truediv__(self, other: MathExpr) -> MathExpr:
                ...

            @egraph.method(cost=costs.get("__xor__"))
            def __xor__(self, other: MathExpr) -> MathExpr:
                ...

            @egraph.method(cost=costs.get("__or__"))
            def __or__(self, other: MathExpr) -> MathExpr:
                ...

        @egraph.function(cost=costs.get("grade_involution"))
        def grade_involution(other: MathExpr) -> MathExpr:
            ...

        @egraph.function(cost=costs.get("clifford_conjugation"))
        def clifford_conjugation(other: MathExpr) -> MathExpr:
            ...

        @egraph.function(cost=costs.get("inverse"))
        def inverse(other: MathExpr) -> MathExpr:
            ...

        @egraph.function(cost=costs.get("boolean"))
        def boolean(b: i64Like) -> MathExpr:
            ...

        @egraph.function(cost=costs.get("scalar"))
        def scalar(value: MathExpr) -> MathExpr:
            ...

        @egraph.function(cost=costs.get("scalar_literal"))
        def scalar_literal(value: f64Like) -> MathExpr:
            ...

        @egraph.function(cost=costs.get("scalar_variable"))
        def scalar_variable(value: StringLike) -> MathExpr:
            ...

        @egraph.function(cost=costs.get("e"))
        def e(s: StringLike) -> MathExpr:
            ...

        @egraph.function(cost=costs.get("e2"))
        def e2(s_1: StringLike, s_2: StringLike) -> MathExpr:
            ...

        @egraph.function(cost=costs.get("e3"))
        def e3(s_1: StringLike, s_2: StringLike, s_3: StringLike) -> MathExpr:
            ...

        @egraph.function(cost=costs.get("variable"))
        def variable(name: StringLike) -> MathExpr:
            ...

        @egraph.function(cost=costs.get("cos"))
        def cos(value: MathExpr) -> MathExpr:
            ...

        @egraph.function(cost=costs.get("sin"))
        def sin(value: MathExpr) -> MathExpr:
            ...

        @egraph.function(cost=costs.get("cosh"))
        def cosh(value: MathExpr) -> MathExpr:
            ...

        @egraph.function(cost=costs.get("sinh"))
        def sinh(value: MathExpr) -> MathExpr:
            ...

        @egraph.function(cost=costs.get("exp"))
        def exp(value: MathExpr) -> MathExpr:
            ...

        @egraph.function(cost=costs.get("grade"))
        def grade(value: MathExpr) -> MathExpr:
            ...

        @egraph.function(cost=costs.get("mix_grades"))
        def mix_grades(a: MathExpr, b: MathExpr) -> MathExpr:
            ...

        @egraph.function(cost=costs.get("select_grade"))
        def select_grade(value: MathExpr, grade: MathExpr) -> MathExpr:
            ...

        @egraph.function(cost=costs.get("abs"))
        def abs_(value: MathExpr) -> MathExpr:
            ...

        @egraph.function(cost=costs.get("rotor"))
        def rotor(basis_blade: MathExpr, angle: MathExpr) -> MathExpr:
            ...

        @egraph.function(cost=costs.get("sandwich"))
        def sandwich(r: MathExpr, x: MathExpr) -> MathExpr:
            ...

        @egraph.function(cost=costs.get("diff"))
        def diff(value: MathExpr, wrt: MathExpr) -> MathExpr:
            ...

        MathExpr.grade_involution = grade_involution
        MathExpr.clifford_conjugation = clifford_conjugation
        MathExpr.inverse = inverse
        MathExpr.boolean = boolean
        MathExpr.scalar = scalar
        MathExpr.scalar_literal = scalar_literal
        MathExpr.scalar_variable = scalar_variable
        MathExpr.e = e
        MathExpr.e2 = e2
        MathExpr.e3 = e3
        MathExpr.variable = variable
        MathExpr.cos = cos
        MathExpr.sin = sin
        MathExpr.cosh = cosh
        MathExpr.sinh = sinh
        MathExpr.exp = exp
        MathExpr.grade = grade
        MathExpr.mix_grades = mix_grades
        MathExpr.select_grade = select_grade
        MathExpr.abs = abs_
        MathExpr.rotor = rotor
        MathExpr.sandwich = sandwich
        MathExpr.diff = diff

        x_1, x_2, x_3 = vars_("x_1 x_2 x_3", MathExpr)
        f_1, f_2 = vars_("f_1 f_2", f64)
        s_1, s_2, s_3 = vars_("s_1 s_2 s_3", String)

        orig_rewrite = rewrite
        orig_birewrite = birewrite
        orig_rule = rule

        def set_active_ruleset(ruleset):
            global rewrite, birewrite, rule
            rewrite = partial(orig_rewrite, ruleset=ruleset)
            birewrite = partial(orig_birewrite, ruleset=ruleset)
            rule = partial(orig_rule, ruleset=ruleset)

        def register_addition(full=True, medium=True):
            egraph.register(
                # Identity
                rewrite(x_1 + scalar_literal(0.0)).to(x_1),
            )
            if medium:
                egraph.register(
                    # Comm
                    rewrite(x_1 + x_2).to(x_2 + x_1),
                )
            if full:
                egraph.register(
                    # Assoc
                    birewrite(x_1 + (x_2 + x_3)).to((x_1 + x_2) + x_3),
                )

            if eq_solve:
                egraph.register(
                    rule(eq(x_3).to(x_1 + x_2)).then(union(x_1).with_(x_3 - x_2)),
                )

        def register_subtraction(medium=True):
            egraph.register(
                # Sub self is zero
                rewrite(x_1 - x_1).to(scalar_literal(0.0)),
                # (a + b) - (a + c) = b - c
                rewrite((x_1 + x_2) - (x_1 + x_3)).to(x_2 - x_3),
            )

            if medium:
                egraph.register(
                    # Add negation is subtraction
                    birewrite(x_1 + -x_2).to(x_1 - x_2),
                )

        def register_negation():
            egraph.register(
                # Involute
                rewrite(--x_1).to(x_1),
                # This makes things blow up
                # birewrite(-x_1).to(scalar_literal(-1.0) * x_1),
            )

        def register_multiplication(full=True, medium=True):
            egraph.register(
                # Identity
                rewrite(scalar_literal(1.0) * x_1).to(x_1),
                # Zero
                rewrite(scalar_literal(0.0) * x_1).to(scalar_literal(0.0)),
                # x + x = 2x
                birewrite(x_1 + x_1).to(scalar_literal(2.0) * x_1),
                # ax + bx = (a+b)x
                birewrite(scalar(x_2) * x_1 + scalar(x_3) * x_1).to(
                    (scalar(x_2) + scalar(x_3)) * x_1
                ),
                # ax + x = (a+1)x
                birewrite(scalar(x_2) * x_1 + x_1).to(
                    (scalar(x_2) + scalar_literal(1.0)) * x_1
                ),
            )

            if medium:
                egraph.register(
                    # Scalar comm
                    birewrite(x_1 * scalar(x_2)).to(scalar(x_2) * x_1),
                    # Assoc
                    birewrite(x_1 * (x_2 * x_3)).to((x_1 * x_2) * x_3),
                    # Left distr
                    birewrite(x_1 * (x_2 + x_3)).to(x_1 * x_2 + x_1 * x_3),
                    # Right distr
                    birewrite((x_1 + x_2) * x_3).to(x_1 * x_3 + x_2 * x_3),
                    # Neg
                    birewrite(-x_1 * x_2).to(-(x_1 * x_2)),
                    birewrite(-x_1 * x_2).to(x_1 * -x_2),
                )
            if full:
                pass

        def register_wedge(medium=True):
            egraph.register(
                birewrite(x_1 ^ x_2).to(
                    select_grade(x_1 * x_2, grade(x_1) + grade(x_2)),
                ),
            )
            if medium:
                egraph.register(
                    # TODO: Can we do without these rules?
                    # Without them, (e1 + e12) ^ e12 = e12 fails.
                    # Assoc
                    birewrite(x_1 ^ (x_2 ^ x_3)).to((x_1 ^ x_2) ^ x_3),
                    # Left distr
                    birewrite(x_1 ^ (x_2 + x_3)).to((x_1 ^ x_2) + (x_1 ^ x_3)),
                    # Right distr
                    birewrite((x_1 + x_2) ^ x_3).to((x_1 ^ x_3) + (x_2 ^ x_3)),
                )

        def register_inner(medium=True):
            egraph.register(
                birewrite(x_1 | x_2).to(
                    select_grade(x_1 * x_2, abs_(grade(x_1) - grade(x_2))),
                ),
            )
            if medium:
                egraph.register(
                    # TODO: Can we do without these rules?
                    # Assoc
                    birewrite(x_1 | (x_2 | x_3)).to((x_1 | x_2) | x_3),
                    # Left distr
                    birewrite(x_1 | (x_2 + x_3)).to((x_1 | x_2) + (x_1 | x_3)),
                    # Right distr
                    birewrite((x_1 + x_2) | x_3).to((x_1 | x_3) + (x_2 | x_3)),
                )

        def register_division(medium=True):
            egraph.register(
                # / is syntactic sugar for multiplication by inverse
                rewrite(x_1 / x_2).to(x_1 * inverse(x_2)),
                # Inverse of non-zero scalar
                rewrite(inverse(scalar_literal(f_1))).to(
                    scalar_literal(f64(1.0) / f_1), _not_close(f_1, 0.0)
                ),
                # Inverse of basis vector
                rewrite(inverse(e(s_1))).to(e(s_1) * inverse(e(s_1) * e(s_1))),
                # Inverse of product is product of inverses in reverse order
                rewrite(inverse(x_1 * x_2)).to(inverse(x_2) * inverse(x_1)),
            )

            if full_inverse:
                for dims in range(len(signature) + 1):
                    deps, x_1_has_inverse, x_1_inverse = _maybe_inverse(
                        m=MathExpr, x=x_1, dims=dims
                    )

                    egraph.register(
                        rule(eq(x_2).to(inverse(x_1))).then(*deps),
                        rewrite(inverse(x_1)).to(x_1_inverse, *x_1_has_inverse),
                    )

            # Multiplicative equation solving with inverses
            if eq_solve:
                for dims in range(len(signature) + 1):
                    deps_1, x_1_has_inverse, x_1_inverse = _maybe_inverse(
                        m=MathExpr, x=x_1, dims=dims
                    )
                    deps_2, x_2_has_inverse, x_2_inverse = _maybe_inverse(
                        m=MathExpr, x=x_2, dims=dims
                    )

                    egraph.register(
                        # x_3 = x_1 * x_2: Figure out if x_1 and x_2 have inverses
                        rule(eq(x_3).to(x_1 * x_2)).then(
                            *deps_1,
                            *deps_2,
                        ),
                        # Left inverse: inv(x_1) * x_3 = x_2
                        rule(eq(x_3).to(x_1 * x_2), *x_1_has_inverse).then(
                            union(x_2).with_(x_1_inverse * x_3),
                        ),
                        # Right inverse: x_3 * inv(x_2) = x_1
                        rule(eq(x_3).to(x_1 * x_2), *x_2_has_inverse).then(
                            union(x_1).with_(x_3 * x_2_inverse),
                        ),
                    )

        def register_pow(medium=True):
            egraph.register(
                # pow zero
                rewrite(x_1 ** scalar_literal(0.0)).to(scalar_literal(1.0)),
                # pow one
                rewrite(x_1 ** scalar_literal(1.0)).to(x_1),
            )
            if medium:
                egraph.register(
                    # expand pow 2 to mul
                    birewrite(x_1 ** scalar_literal(2.0)).to(x_1 * x_1),
                )

        def register_exp(medium=True):
            egraph.register(
                # B^2 = -1
                birewrite(exp(scalar(x_2) * x_1)).to(
                    cos(scalar(x_2)) + x_1 * sin(scalar(x_2)),
                    eq(x_1 * x_1).to(scalar_literal(-1.0)),
                ),
                # B^2 = 0
                birewrite(exp(scalar(x_2) * x_1)).to(
                    scalar_literal(1.0) + x_1 * scalar(x_2),
                    eq(x_1 * x_1).to(scalar_literal(0.0)),
                ),
                # B^2 = +1
                birewrite(exp(scalar(x_2) * x_1)).to(
                    cosh(scalar(x_2)) + x_1 * sinh(scalar(x_2)),
                    eq(x_1 * x_1).to(scalar_literal(1.0)),
                ),
                # Euler's formula etc. require adding B^2 so the rule can get matched.
                rule(eq(x_2).to(exp(x_1))).then(x_1 * x_1),
                # exp(x) ** f is exp(f x)
                birewrite(exp(x_1) ** scalar(x_2)).to(exp(scalar(x_2) * x_1)),
                # exp(x) * exp(y) -> exp(x + y) if [x, y] = 0
                rule(eq(x_3).to(exp(x_1) * exp(x_2))).then(x_1 * x_2, x_2 * x_1),
                rewrite(exp(x_1) * exp(x_2)).to(
                    exp(x_1 + x_2), eq(x_1 * x_2).to(x_2 * x_1)
                ),
                # TODO: Add more general BCH case, might require adding (infinite) sum
                # exp(x) * exp(y) = exp(X + Y + [X, Y]/2 + [X, [X, Y]]/12 + ...)
            )

        def register_scalar(medium=True):
            if medium:
                egraph.register(
                    birewrite(scalar_literal(f_1)).to(scalar(scalar_literal(f_1))),
                    birewrite(scalar_variable(s_1)).to(scalar(scalar_variable(s_1))),
                    birewrite(scalar_variable(s_1)).to(scalar(variable(s_1))),
                    rewrite(scalar_variable(s_1)).to(variable(s_1)),
                    # -0 is 0 (apparently not true for f64)
                    birewrite(scalar_literal(-0.0)).to(scalar_literal(0.0)),
                    birewrite(-scalar_literal(0.0)).to(scalar_literal(0.0)),
                    union(scalar_literal(-0.0)).with_(scalar_literal(0.0)),
                    union(-scalar_literal(0.0)).with_(scalar_literal(0.0)),
                    # Scalar
                    rewrite(scalar(x_1) + scalar(x_2)).to(scalar(x_1 + x_2)),
                    rewrite(scalar(x_1) - scalar(x_2)).to(scalar(x_1 - x_2)),
                    rewrite(-scalar(x_1)).to(scalar(-x_1)),
                    rewrite(scalar(x_1) * scalar(x_2)).to(scalar(x_1 * x_2)),
                    rewrite(scalar(x_1) / scalar(x_2)).to(scalar(x_1 / x_2)),
                    # Scalar literal
                    rewrite(scalar_literal(f_1) + scalar_literal(f_2)).to(
                        scalar_literal(f_1 + f_2)
                    ),
                    rewrite(scalar_literal(f_1) - scalar_literal(f_2)).to(
                        scalar_literal(f_1 - f_2)
                    ),
                    rewrite(-scalar_literal(f_1)).to(scalar_literal(-f_1)),
                    rewrite(scalar_literal(f_1) * scalar_literal(f_2)).to(
                        scalar_literal(f_1 * f_2)
                    ),
                    rewrite(scalar_literal(f_1) / scalar_literal(f_2)).to(
                        scalar_literal(f_1 / f_2), _not_close(f_2, 0.0)
                    ),
                    # Scalar literal - abs
                    rewrite(abs_(scalar_literal(f_1))).to(
                        scalar_literal(f_1), f_1 >= 0.0
                    ),
                    rewrite(abs_(scalar_literal(f_1))).to(
                        scalar_literal(-f_1), f_1 < 0.0
                    ),
                )

        def register_grade():
            # -- Idea:
            # mix_grades(x, y) -> x if x == y
            # grade(s) -> 0
            # grade(e(i)) -> 1
            # grade(s * e(i)) -> 1
            # grade(x + y) -> mix_grades(grade(x), grade(y))
            # -- Example:
            # grade(e1 + e2 + e3) ->
            # mix_grades(grade(e1), grade(e2 + e3)) ->
            # mix_grades(grade(e1), mix_grades(grade(e2), grade(e3)))

            egraph.register(
                # mix_grades(x, y) -> x if x == y
                rewrite(mix_grades(x_1, x_2)).to(x_1, eq(x_1).to(x_2)),
                # mix_grades comm
                rewrite(mix_grades(x_1, x_2)).to(mix_grades(x_2, x_1)),
                # negation doesn't affect grade
                rewrite(grade(-x_1)).to(grade(x_1)),
                # Grade is scalar
                birewrite(grade(x_1)).to(scalar(grade(x_1))),
                # Grade of scalar is 0, if scalar is not zero
                # rule(eq(x_2).to(grade(scalar(x_1)))).then(
                #     x_1 != scalar_literal(0.0)
                # ),
                rewrite(grade(scalar_literal(f_1))).to(
                    scalar_literal(0.0), _not_close(f_1, 0.0)
                ),
                # grade(a + b) -> mix_grades(grade(a), grade(b)), if a + b is not zero
                # rule(eq(x_1).to(grade(x_2 + x_3))).then(
                #     x_2 + x_3 != scalar_literal(0.0)
                # ),
                rewrite(grade(x_1 + x_2)).to(
                    mix_grades(grade(x_1), grade(x_2)),
                    x_1 + x_2 != scalar_literal(0.0),
                ),
                # grade(s * x) -> grade(x) if s != 0
                # With scalar coef, if scalar coef is not zero
                # rule(eq(x_1).to(scalar(x_2) * x_3)).then(
                #     x_2 != scalar_literal(0.0)
                # ),
                rewrite(grade(scalar_literal(f_1) * x_2)).to(
                    grade(x_2), _not_close(f_1, 0.0)
                ),
            )

            # Basis blade grades
            for blade_grade in range(1, len(signature) + 1):
                basis_blade = None
                basis_vector_names = [var(f"s_{i}", String) for i in range(blade_grade)]
                basis_vectors = [e(name) for name in basis_vector_names]
                for basis_vector in basis_vectors:
                    if basis_blade is None:
                        basis_blade = basis_vector
                    else:
                        basis_blade *= basis_vector
                conds = []
                for name_1, name_2 in combinations(basis_vector_names, 2):
                    conds.append(name_1 != name_2)
                egraph.register(
                    rewrite(grade(basis_blade)).to(
                        scalar_literal(float(blade_grade)), *conds
                    ),
                )

            # Select grade
            egraph.register(
                # select_grade(x + y, z) -> select_grade(x, z) + select_grade(y, z)
                rewrite(select_grade(x_1 + x_2, x_3)).to(
                    select_grade(x_1, x_3) + select_grade(x_2, x_3)
                ),
                # select_grade(x, y) -> 0 if grade(x) != y
                rewrite(select_grade(x_1, scalar_literal(f_1))).to(
                    scalar_literal(0.0),
                    eq(grade(x_1)).to(scalar_literal(f_2)),
                    _not_close(f_1, f_2),
                ),
                # select_grade(x, y) -> x if grade(x) == y
                rule(eq(x_3).to(select_grade(x_1, scalar_literal(f_1)))).then(
                    grade(x_1)
                ),
                rewrite(select_grade(x_1, scalar_literal(f_1))).to(
                    x_1, eq(grade(x_1)).to(scalar_literal(f_1))
                ),
            )

        def register_basic_ga(medium=True):
            basis_vectors = [e(str(i)) for i in range(len(signature))]
            egraph.register(
                # e_i^2 = signature[i]
                *map(
                    lambda e, s: rewrite(e * e).to(scalar_literal(s)),
                    basis_vectors,
                    signature,
                ),
                # e_i e_j = e_ij
                birewrite(e(s_1) * e(s_2)).to(e2(s_1, s_2)),
                birewrite(e(s_1) * e(s_2) * e(s_3)).to(e3(s_1, s_2, s_3)),
                # Rotor
                birewrite(exp(x_2 * scalar(x_1) * scalar_literal(-0.5))).to(
                    rotor(x_2, scalar(x_1))
                ),
                # Sandwich
                rewrite(sandwich(x_1, x_2)).to(x_1 * x_2 * ~x_1),
                # # (e_i + e_j)^2 = e_i^2 + e_j^2
                birewrite((e(s_1) + e(s_2)) ** scalar_literal(2.0)).to(
                    e(s_1) ** scalar_literal(2.0) + e(s_2) ** scalar_literal(2.0),
                    s_1 != s_2,
                ),
                # (a + b)^2 = a^2 + ab + ba + b^2
                # birewrite((x_1 + x_2) * (x_1 + x_2)).to(
                #     x_1 * x_1 + x_1 * x_2 + x_2 * x_1 + x_2 * x_2
                # ),
                # rule((x_1 + x_2) * (x_1 + x_2)).then(x_1 * x_2, x_2 * x_1),
                # birewrite((x_1 + x_2) * (x_1 + x_2)).to(
                #     x_1 * x_1 + x_2 * x_2,
                #     eq(x_1 * x_2).to(-x_2 * x_1),
                # ),
            )

            if medium:
                egraph.register(
                    # e_i e_j = -e_j e_i, i != j
                    birewrite(-e(s_1) * e(s_2)).to(e(s_2) * e(s_1), s_1 != s_2),
                    # Sandwich
                    rewrite(x_1 * x_2 * ~x_1).to(sandwich(x_1, x_2)),
                )

        def register_symplectic_ga(medium=True):
            q_vectors = [e(str(i)) for i in range(len(signature))]
            p_vectors = [e(f"{i}*") for i in range(len(signature))]

            if True:
                egraph.register(
                    # q p - p q = 1
                    *map(
                        lambda q, p, s: birewrite(q * p - p * q).to(
                            scalar_literal(2 * s)
                        ),
                        q_vectors,
                        p_vectors,
                        signature,
                    ),
                    # q p = p q + 1
                    *map(
                        lambda q, p, s: birewrite(q * p).to(
                            p * q + scalar_literal(2 * s)
                        ),
                        q_vectors,
                        p_vectors,
                        signature,
                    ),
                    # p q = q p - 1
                    *map(
                        lambda q, p, s: birewrite(p * q).to(
                            q * p - scalar_literal(2 * s)
                        ),
                        q_vectors,
                        p_vectors,
                        signature,
                    ),
                )

            egraph.register(
                # cutoff e_i^n = 0
                rewrite(e(s_1) * e(s_1) * e(s_1) * e(s_1)).to(scalar_literal(0.0)),
                # e_i e_j = e_ij
                birewrite(e(s_1) * e(s_2)).to(e2(s_1, s_2)),
                birewrite(e(s_1) * e(s_2) * e(s_3)).to(e3(s_1, s_2, s_3)),
            )

            # # # q_i p_j = p_j q_i, i != j
            # for i, q in enumerate(q_vectors):
            #     for j, p in enumerate(p_vectors):
            #         if i != j:
            #             egraph.register(birewrite(q * p).to(p * q))
            # # q_i q_j = q_j q_i, i != j
            # for i, q_1 in enumerate(q_vectors):
            #     for j, q_2 in enumerate(q_vectors):
            #         if i != j:
            #             egraph.register(birewrite(q_1 * q_2).to(q_2 * q_1))
            # # p_i p_j = p_j p_i, i != j
            # for i, p_1 in enumerate(p_vectors):
            #     for j, p_2 in enumerate(p_vectors):
            #         if i != j:
            #             egraph.register(birewrite(p_1 * p_2).to(p_2 * p_1))

        def register_reverse(medium=True):
            if not symplectic:
                egraph.register(
                    rewrite(~~x_1).to(x_1),
                    rewrite(~scalar(x_1)).to(scalar(x_1)),
                    rewrite(~e(s_1)).to(e(s_1)),
                )

                if medium:
                    egraph.register(
                        birewrite(~(x_1 * x_2)).to(~x_2 * ~x_1),
                        birewrite(~(x_1 + x_2)).to(~x_1 + ~x_2),
                    )

        def register_grade_involution():
            egraph.register(
                birewrite(grade_involution(x_1 * x_2)).to(
                    grade_involution(x_1) * grade_involution(x_2)
                ),
                birewrite(grade_involution(x_1 + x_2)).to(
                    grade_involution(x_1) + grade_involution(x_2)
                ),
                rewrite(grade_involution(scalar_literal(f_1))).to(scalar_literal(f_1)),
                rewrite(grade_involution(e(s_1))).to(-e(s_1)),
                rewrite(grade_involution(grade_involution(x_1))).to(x_1),
            )

        def register_clifford_conjugation():
            egraph.register(
                birewrite(clifford_conjugation(x_1 * x_2)).to(
                    clifford_conjugation(x_2) * clifford_conjugation(x_1)
                ),
                birewrite(clifford_conjugation(x_1 + x_2)).to(
                    clifford_conjugation(x_1) + clifford_conjugation(x_2)
                ),
                rewrite(clifford_conjugation(scalar_literal(f_1))).to(
                    scalar_literal(f_1)
                ),
                rewrite(clifford_conjugation(e(s_1))).to(-e(s_1)),
                rewrite(clifford_conjugation(clifford_conjugation(x_1))).to(x_1),
            )

        def register_equality():
            egraph.register(
                # Comm
                birewrite(x_1.equal(x_2)).to(x_2.equal(x_1)),
                birewrite(x_1.not_equal(x_2)).to(x_2.not_equal(x_1)),
                # Eq / Ne
                rewrite(x_1.equal(x_1)).to(boolean(1)),
                rewrite(x_1.not_equal(x_1)).to(boolean(0)),
                rewrite(x_1.equal(x_2)).to(boolean(0), x_1 != x_2),
                rewrite(x_1.not_equal(x_2)).to(boolean(1), x_1 != x_2),
            )

        def register_trigonometry():
            egraph.register(
                # scalar
                birewrite(sin(x_1)).to(scalar(sin(x_1))),
                birewrite(cos(x_1)).to(scalar(cos(x_1))),
                birewrite(sinh(x_1)).to(scalar(sinh(x_1))),
                birewrite(cosh(x_1)).to(scalar(cosh(x_1))),
                # sin/cos
                rewrite(cos(scalar_literal(0.0))).to(scalar_literal(1.0)),
                rewrite(sin(scalar_literal(0.0))).to(scalar_literal(0.0)),
                rewrite(cos(x_2) * cos(x_2) + sin(x_1) * sin(x_1)).to(
                    scalar_literal(1.0)
                ),
                rewrite(-sin(-x_1)).to(sin(x_1)),
                rewrite(cos(-x_1)).to(cos(x_1)),
                birewrite(cos(x_1) * cos(x_2)).to(
                    scalar_literal(0.5) * (cos(x_1 - x_2) + cos(x_1 + x_2))
                ),
                birewrite(sin(x_1) * sin(x_2)).to(
                    scalar_literal(0.5) * (cos(x_1 - x_2) - cos(x_1 + x_2))
                ),
                birewrite(sin(x_1) * cos(x_2)).to(
                    scalar_literal(0.5) * (sin(x_1 + x_2) + sin(x_1 - x_2))
                ),
                birewrite(cos(x_1) * sin(x_2)).to(
                    scalar_literal(0.5) * (sin(x_1 + x_2) - sin(x_1 - x_2))
                ),
                # sinh/cosh
                rewrite(cosh(x_2) * cosh(x_2) - sinh(x_1) * sinh(x_1)).to(
                    scalar_literal(1.0)
                ),
                rewrite(-sin(-x_1)).to(sin(x_1)),
                rewrite(cos(-x_1)).to(cos(x_1)),
            )

        def register_diff(medium=True):
            # TODO: maybe add constant() to unify scalar_literal and e?

            if medium:
                # Linearity
                egraph.register(
                    # Addition
                    birewrite(diff(x_1 + x_2, x_3)).to(diff(x_1, x_3) + diff(x_2, x_3)),
                    # Constant multiplication
                    birewrite(diff(scalar_literal(f_1) * x_2, x_3)).to(
                        scalar_literal(f_1) * diff(x_2, x_3)
                    ),
                    birewrite(diff(e(s_1) * x_2, x_3)).to(e(s_1) * diff(x_2, x_3)),
                    birewrite(diff(x_2 * e(s_1), x_3)).to(diff(x_2, x_3) * e(s_1)),
                )

            # Concrete derivatives
            egraph.register(
                # wrt self
                # TODO: Fix not constant condition
                # rewrite(diff(x_1, x_1)).to(
                #     scalar_literal(1.0)#, x_1 != scalar_literal(f_1)
                # ),
                rewrite(diff(variable(s_1), variable(s_1))).to(scalar_literal(1.0)),
                # wrt other
                # rewrite(diff(variable(s_1), variable(s_2))).to(
                #     scalar_literal(0.0), s_1 != s_2
                # ),
                # constant: scalar_literal
                rewrite(diff(scalar_literal(f_1), x_1)).to(scalar_literal(0.0)),
                # constant: e
                rewrite(diff(e(s_1), x_1)).to(scalar_literal(0.0)),
                # x * y
                rewrite(diff(x_1 * x_2, x_3)).to(
                    diff(x_1, x_3) * x_2 + x_1 * diff(x_2, x_3)
                ),
                # sin(x)
                rewrite(diff(sin(x_1), x_2)).to(cos(x_1) * diff(x_1, x_2)),
                # cos(x)
                rewrite(diff(cos(x_1), x_2)).to(-sin(x_1) * diff(x_1, x_2)),
                # sinh(x)
                rewrite(diff(sinh(x_1), x_2)).to(cosh(x_1) * diff(x_1, x_2)),
                # cosh(x)
                rewrite(diff(cosh(x_1), x_2)).to(sinh(x_1) * diff(x_1, x_2)),
                # exp(x)
                rewrite(diff(exp(x_1), x_2)).to(exp(x_1) * diff(x_1, x_2)),
                # reverse
                rewrite(diff(~x_1, x_2)).to(~diff(x_1, x_2)),
                # negative
                rewrite(diff(-x_1, x_2)).to(-diff(x_1, x_2)),
                # square
                rewrite(diff(x_1 * x_1, x_1)).to(scalar_literal(2.0) * x_1),
            )

        full_ruleset = egraph.ruleset("full")
        medium_ruleset = egraph.ruleset("medium")
        fast_ruleset = egraph.ruleset("fast")

        set_active_ruleset(full_ruleset)
        register_addition()
        register_negation()
        register_subtraction()
        register_multiplication()
        register_division()
        register_pow()
        register_trigonometry()
        register_exp()
        register_scalar()
        register_equality()
        register_grade()
        if symplectic:
            register_symplectic_ga()
        else:
            register_basic_ga()
        register_wedge()
        register_inner()
        register_reverse()
        register_grade_involution()
        register_clifford_conjugation()
        register_diff()

        set_active_ruleset(medium_ruleset)
        register_addition(full=False)
        register_negation()
        register_subtraction()
        register_multiplication(full=False)
        register_division()
        register_pow()
        register_trigonometry()
        register_exp()
        register_scalar()
        register_equality()
        register_grade()
        if symplectic:
            register_symplectic_ga()
        else:
            register_basic_ga()
        register_wedge()
        register_inner()
        register_reverse()
        register_grade_involution()
        register_clifford_conjugation()
        register_diff()

        set_active_ruleset(fast_ruleset)
        register_addition(medium=False, full=False)
        register_negation()
        register_subtraction(medium=False)
        register_multiplication(medium=False, full=False)
        register_division(medium=False)
        register_pow(medium=False)
        register_trigonometry()
        register_exp(medium=False)
        register_scalar(medium=False)
        register_equality()
        register_grade()
        if symplectic:
            register_symplectic_ga(medium=False)
        else:
            register_basic_ga(medium=False)
        register_wedge(medium=False)
        register_inner(medium=False)
        register_reverse(medium=False)
        register_grade_involution()
        register_clifford_conjugation()
        register_diff(medium=False)

        self.egraph = egraph
        self.expr_cls = MathExpr
        self.rulesets = GeometricAlgebraRulesets(
            full=full_ruleset,
            medium=medium_ruleset,
            fast=fast_ruleset,
        )
        self.signature = signature
