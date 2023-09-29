from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from itertools import combinations
from typing import Dict, Optional, Tuple, Type

from egglog import (
    BaseExpr,
    EGraph,
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
    set_,
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
    ):
        if costs is None:
            costs = {}

        egraph = EGraph()

        @egraph.class_
        class MathExpr(BaseExpr):
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
                    # Assoc
                    # rewrite(x_1 + (x_2 + x_3)).to((x_1 + x_2) + x_3),
                )
            if full:
                egraph.register(
                    # Assoc
                    # rewrite((x_1 + x_2) + x_3).to(x_1 + (x_2 + x_3)),
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
            # TODO: Left / right inverse

            egraph.register(
                # Divide identity
                rewrite(x_1 / scalar_literal(1.0)).to(x_1),
                # Divide self
                rewrite(x_1 / x_1).to(scalar_literal(1.0), x_1 != scalar_literal(0.0)),
            )

            if medium:
                egraph.register(
                    # 1 / x is inverse of x
                    birewrite(scalar_literal(1.0) / x_1).to(inverse(x_1)),
                    # Division is right multiplication inverse
                    birewrite(x_1 / x_2).to(x_1 * inverse(x_2)),
                    # Inverse basis vector
                    # TODO: Figure out why this is broken x_1 * x_1 != scalar_literal(0.0)
                    rule(eq(x_1).to(inverse(x_2))).then(
                        x_2 * x_2 != scalar_literal(0.0)
                    ),
                    birewrite(inverse(x_1)).to(
                        x_1 / (x_1 * x_1), x_1 * x_1 != scalar_literal(0.0)
                    ),
                )

            if eq_solve:
                egraph.register(
                    rule(eq(x_3).to(x_1 * x_2)).then(
                        x_1 * x_1 != scalar_literal(0.0),
                        x_2 * x_2 != scalar_literal(0.0),
                    ),
                    # Left inverse: x_3 = x_1 * x_2 -> inv(x_1) * x_3 = x_2
                    rule(eq(x_3).to(x_1 * x_2), x_1 * x_1 != scalar_literal(0.0)).then(
                        union(x_2).with_(inverse(x_1) * x_3)
                    ),
                    # Right inverse: x_3 = x_1 * x_2 -> x_3 * inv(x_2) = x_1
                    rule(eq(x_3).to(x_1 * x_2), x_2 * x_2 != scalar_literal(0.0)).then(
                        union(x_1).with_(x_3 * inverse(x_2))
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
            )

        def register_scalar(medium=True):
            if medium:
                egraph.register(
                    birewrite(scalar_literal(f_1)).to(scalar(scalar_literal(f_1))),
                    birewrite(scalar_variable(s_1)).to(scalar(scalar_variable(s_1))),
                    birewrite(scalar_variable(s_1)).to(scalar(variable(s_1))),
                    rewrite(scalar_variable(s_1)).to(variable(s_1)),
                    # -0 is 0 (apparently not true for f64)
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
                    )
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

        def register_equality():
            egraph.register(
                # Comm
                birewrite(x_1.equal(x_2)).to(x_2.equal(x_1)),
                birewrite(x_1.not_equal(x_2)).to(x_2.not_equal(x_1)),
                # Eq / Ne
                rewrite(x_1.equal(x_1)).to(boolean(True)),
                rewrite(x_1.not_equal(x_1)).to(boolean(False)),
                rewrite(x_1.equal(x_2)).to(boolean(False), x_1 != x_2),
                rewrite(x_1.not_equal(x_2)).to(boolean(True), x_1 != x_2),
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
        register_diff(medium=False)

        self.egraph = egraph
        self.expr_cls = MathExpr
        self.rulesets = GeometricAlgebraRulesets(
            full=full_ruleset,
            medium=medium_ruleset,
            fast=fast_ruleset,
        )
        self.signature = signature
