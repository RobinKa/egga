from typing import List, Tuple, Union

import pytest
from egglog import eq

from egga.expression import Expression
from egga.geometric_algebra import GeometricAlgebra
from egga.utils import (
    CheckEqualityOptions,
    RunRulesetOptions,
    check_equality,
    run_ruleset,
)
from itertools import combinations

ga = GeometricAlgebra([0.0, 1.0, 1.0, 1.0, -1.0])
E = ga.expr_cls

e_0, e_1, e_2, e_3, e_4 = ga.basis_vectors

e_12 = e_1 * e_2
e_23 = e_2 * e_3
e_31 = e_3 * e_1
e_123 = e_1 * e_2 * e_3


test_equations: List[
    Union[Tuple[Expression, Expression], Tuple[Expression, Expression, bool]]
] = [
    (-e_1 * e_2, e_2 * e_1),
    (e_1 * e_1, E.scalar_literal(1.0)),
    (-e_1 * e_2 * e_1, e_2),
    (~(e_1 * e_2), -e_1 * e_2),
    (e_12 * e_1 * ~e_12, -e_1),
    (e_23 * e_1 * ~e_23, e_1),
    (~(e_12 + e_23), -e_12 - e_23),
    ((e_12 + e_23) * e_1, -e_2 + e_123),
    (~(e_12 + e_23 + e_31), -e_12 - e_23 - e_31),
    ((e_12 + e_23 + e_31) * e_1, -e_2 + e_123 + e_3),
    (e_1 * ~(e_12 + e_23 + e_31), -e_2 - e_123 + e_3),
    (e_23 ** E.scalar_literal(1.0), e_23),
    (e_23 * e_23, E.scalar_literal(-1.0)),
    (e_1.equal(E.scalar_literal(1.0) * e_1), E.boolean(1)),
    (e_1.not_equal(E.scalar_literal(1.0) * e_1), E.boolean(0)),
    (e_1.equal(-E.scalar_literal(1.0) * e_1), E.boolean(0)),
    (e_1.not_equal(-E.scalar_literal(1.0) * e_1), E.boolean(1)),
    (E.inverse(e_1), e_1),
    (E.inverse(e_12), -e_12),
    (E.scalar_literal(1.0) / e_12, -e_12),
    (E.scalar_literal(5.0) / e_12, -E.scalar_literal(5.0) * e_12),
    (e_123 / e_12, e_3),
    (e_12 / e_3, e_123),
    (e_1 ^ e_2, e_12),
    (e_1 ^ e_1, E.scalar_literal(0.0)),
    (e_1 | e_2, E.scalar_literal(0.0)),
    (e_1 | e_1, E.scalar_literal(1.0)),
    (e_0 * e_0, E.scalar_literal(0.0)),
    (e_4 * e_4, E.scalar_literal(-1.0)),
    (e_1 * e_4 * e_1 * e_4, E.scalar_literal(1.0)),
    (
        E.exp(E.scalar_variable("x") * e_23),
        E.cos(E.scalar_variable("x")) + e_23 * E.sin(E.scalar_variable("x")),
    ),
    (
        E.exp(e_23),
        E.cos(E.scalar_literal(1.0)) + e_23 * E.sin(E.scalar_literal(1.0)),
    ),
    (E.exp(e_0 * e_1), E.scalar_literal(1.0) + e_0 * e_1),
    (E.exp(e_0 * e_4), E.scalar_literal(1.0) + e_0 * e_4),
    (
        E.exp(e_1 * e_4),
        E.cosh(E.scalar_literal(1.0)) + e_1 * e_4 * E.sinh(E.scalar_literal(1.0)),
    ),
    (E.scalar_literal(22.0) * e_1, e_1 * E.scalar_literal(22.0)),
    (
        E.scalar(E.sin(E.scalar_literal(22.0))) * e_1,
        e_1 * E.scalar(E.sin(E.scalar_literal(22.0))),
    ),
    (
        E.sin(E.scalar_literal(1.0)) * e_1,
        e_1 * E.sin(E.scalar_literal(1.0)),
    ),
    ((e_1 + e_12) ^ e_2, e_12),
    (e_123 ^ e_2, E.scalar_literal(0.0)),
    (e_1 ^ e_2, e_123, False),
    (E.grade(e_1), E.scalar_literal(1.0)),
    (E.grade(e_1), E.scalar_literal(0.0), False),
    (
        E.select_grade(e_1 * e_2, E.scalar_literal(2.0)),
        e_1 * e_2,
    ),
    (E.grade(-e_1), E.scalar_literal(1.0)),
    (E.grade(E.scalar_literal(-1.0) * e_1), E.scalar_literal(1.0)),
    (E.grade(e_1 ^ e_2), E.scalar_literal(2.0)),
    (E.grade(e_1 ^ e_2), E.scalar_literal(1.0), False),
    (E.grade(e_1 ^ e_2), E.scalar_literal(0.0), False),
    (e_23 ^ e_1, e_123),
    (E.scalar_literal(1.0), E.scalar_literal(2.0), False),
    (e_1 ^ e_2, E.scalar_literal(0.0), False),
    (
        E.select_grade(e_12, E.scalar_literal(2.0)),
        E.scalar_literal(0.0),
        False,
    ),
    (
        E.select_grade(E.scalar_literal(5.0) * e_1 * e_2, E.scalar_literal(2.0)),
        E.scalar_literal(0.0),
        False,
    ),
    (
        E.grade(E.scalar_literal(5.0) * e_1 * e_2),
        E.scalar_literal(0.0),
        False,
    ),
    (E.grade(E.scalar_literal(5.0) * e_1), E.scalar_literal(1.0)),
    (E.scalar_literal(5.0) * e_1, E.scalar(E.scalar_literal(5.0)) * e_1),
    (E.grade(E.scalar_literal(5.0) * e_1 * e_2), E.scalar_literal(2.0)),
    (E.grade(e_1 * e_2), E.scalar_literal(2.0)),
    (E.grade(e_1 * e_2), E.scalar_literal(0.0), False),
    (E.grade(e_1), E.scalar_literal(2.0), False),
    (
        E.select_grade(e_1, E.scalar_literal(1.0)),
        E.scalar_literal(1.0),
        False,
    ),
    (E.select_grade(e_1, E.scalar_literal(1.0)), e_1),
    (e_1, E.scalar_literal(1.0), False),
    (e_123, E.scalar_literal(0.0), False),
    (
        (e_1 * E.scalar_literal(5.0)) ^ (e_2 * e_3),
        e_123 * E.scalar_literal(5.0),
    ),
    (e_1 ^ (e_2 * e_3), e_123),
    (e_1 ^ (e_2 ^ e_3), e_123),
    (e_1 | e_23, E.scalar_literal(0.0)),
    (e_1 * (e_2 * e_3), (e_2 * e_3) * e_1),
    (e_1 * e_2 * e_3 - e_2 * e_3 * e_1, E.scalar_literal(0.0)),
    (e_1 * e_2 * e_3, e_2 * e_3 * e_1),
    ((E.scalar_literal(5.0) * e_12) ^ e_1, E.scalar_literal(0.0)),
    (
        e_2 * (E.scalar_literal(-5.0) ^ e_1),
        (e_1 * e_2) * E.scalar_literal(5.0),
    ),
    (
        (e_2 * e_1) * (E.scalar_literal(-5.0) ^ e_1),
        E.scalar_literal(-5.0) * e_2,
    ),
    ((E.scalar_literal(5.0) * e_1) ^ e_1, E.scalar_literal(0.0)),
    (E.abs(E.grade(e_1) - E.grade(e_2)), E.scalar_literal(0.0)),
    (
        E.select_grade(e_1 * e_2, E.scalar_literal(0.0)),
        E.scalar_literal(0.0),
    ),
    ((e_1 ^ e_2) + (e_1 | e_2), e_12),
    (e_12, e_1 ^ e_2),
    (e_12, (e_1 ^ e_2) + (e_1 | e_2)),
    (E.grade(e_1 + e_2), E.scalar_literal(1.0)),
    (E.mix_grades(E.grade(e_1), E.grade(e_2)), E.scalar_literal(1.0)),
    (E.grade(e_1 + e_2 + e_3), E.scalar_literal(1.0)),
    (
        E.grade(e_1 + e_23),
        E.mix_grades(E.scalar_literal(1.0), E.scalar_literal(2.0)),
    ),
    (
        E.grade(e_1 + e_23),
        E.mix_grades(E.scalar_literal(2.0), E.scalar_literal(1.0)),
    ),
    (E.grade(E.scalar_literal(3.0)), E.scalar_literal(0.0)),
    (E.select_grade(e_1, E.scalar_literal(2.0)), E.scalar_literal(0.0)),
    (E.select_grade(e_1 + e_23, E.scalar_literal(1.0)), e_1),
    (E.select_grade(e_1 + e_23, E.scalar_literal(2.0)), e_23),
    (
        E.select_grade(E.scalar_literal(5.0) + e_1 + e_23, E.scalar_literal(0.0)),
        E.scalar_literal(5.0),
    ),
    (e_1 ^ e_2 ^ e_3, e_1 * e_2 * e_3),
    (
        e_1 ^ e_2 ^ (E.scalar_literal(5.0) * e_3),
        E.scalar_literal(5.0) * e_1 * e_2 * e_3,
    ),
    (E.abs(-E.scalar_literal(2.0)), E.scalar_literal(2.0)),
    (E.abs(E.grade(e_1) - E.grade(e_23)), E.scalar_literal(1.0)),
    # (E.inverse(e_0), e_0, False),
    (e_1 ^ e_2, E.select_grade(e_1 * e_2, E.grade(e_1) + E.grade(e_2))),
    (e_1 ^ e_2, E.select_grade(e_1 * e_2, E.scalar_literal(2.0))),
    (
        E.select_grade(e_1 * e_2, E.scalar_literal(0.0)),
        e_1 * e_2,
        False,
    ),
    (
        E.diff(E.scalar_literal(5.0), E.variable("x")),
        E.scalar_literal(0.0),
    ),
    (
        E.diff(E.variable("x"), E.scalar_literal(5.0)),
        E.scalar_literal(0.0),
        False,
    ),
    (
        E.diff(E.scalar_literal(5.0), E.scalar_literal(5.0)),
        E.scalar_literal(0.0),
    ),
    (E.diff(E.variable("x"), E.variable("x")), E.scalar_literal(1.0)),
    # (E.diff(E.variable("x"), E.variable("y")), E.scalar_literal(0.0)),
    (
        E.diff(E.variable("x") + E.variable("y"), E.variable("z")),
        E.diff(E.variable("x"), E.variable("z"))
        + E.diff(E.variable("y"), E.variable("z")),
    ),
    (
        E.diff(E.variable("x") * E.variable("y"), E.variable("z")),
        E.diff(E.variable("x"), E.variable("z")) * E.variable("y")
        + E.variable("x") * E.diff(E.variable("y"), E.variable("z")),
    ),
    # (
    #     E.diff(
    #         E.variable("x") * e_1 + E.variable("y") * e_2, E.variable("x")
    #     ),
    #     e_1,
    # ),
    # (
    #     E.diff(
    #         e_1 * E.variable("x") + e_2 * E.variable("y"), E.variable("x")
    #     ),
    #     e_1,
    # ),
    (E.diff(e_12 * E.variable("x"), E.variable("x")), e_12),
    (
        E.diff(
            e_1 * E.variable("x") * e_1,
            E.variable("x"),
        ),
        E.scalar_literal(1.0),
    ),
    (
        E.diff(E.sin(E.variable("x")), E.variable("x")),
        E.cos(E.variable("x")),
    ),
    (
        E.diff(E.cos(E.variable("x")), E.variable("x")),
        -E.sin(E.variable("x")),
    ),
    (
        E.diff(E.sinh(E.variable("x")), E.variable("x")),
        E.cosh(E.variable("x")),
    ),
    (
        E.diff(E.cosh(E.variable("x")), E.variable("x")),
        E.sinh(E.variable("x")),
    ),
    # (
    #     E.diff(
    #         E.rotor(e_12, E.scalar_variable("phi"))
    #         * e_1
    #         * ~E.rotor(e_12, E.scalar_variable("phi")),
    #         E.scalar_variable("phi"),
    #     ),
    #     # R' ~R
    #     E.diff(
    #         E.rotor(e_12, E.scalar_variable("phi")),
    #         E.scalar_variable("phi"),
    #     )
    #     * ~E.rotor(e_12, E.scalar_variable("phi")),
    # ),
    (
        E.rotor(e_12, E.scalar_variable("phi"))
        * ~E.rotor(e_12, E.scalar_variable("phi")),
        E.scalar_literal(1.0),
    ),
    (
        E.sin(E.variable("x")) * E.sin(E.variable("x")),
        E.scalar_literal(0.5)
        * (E.scalar_literal(1.0) - E.cos(E.scalar_literal(2.0) * E.variable("x"))),
    ),
    (
        E.rotor(e_12, E.scalar_variable("phi"))
        * e_1
        * ~E.rotor(e_12, E.scalar_variable("phi")),
        E.cos(E.scalar_variable("phi")) * e_1 + E.sin(E.scalar_variable("phi")) * e_2,
    ),
    (
        E.rotor(e_12, E.scalar_variable("phi"))
        * (E.scalar_variable("x") * e_1 + E.scalar_variable("y") * e_2)
        * ~E.rotor(e_12, E.scalar_variable("phi")),
        e_1
        * (
            E.scalar_variable("x") * E.cos(E.scalar_variable("phi"))
            - E.scalar_variable("y") * E.sin(E.scalar_variable("phi"))
        )
        + e_2
        * (
            E.scalar_variable("x") * E.sin(E.scalar_variable("phi"))
            + E.scalar_variable("y") * E.cos(E.scalar_variable("phi"))
        ),
    ),
    (e_1 + e_1, E.scalar_literal(2.0) * e_1),
    (E.scalar_literal(1.5) * e_1 + e_1, E.scalar_literal(2.5) * e_1),
    (e_1 + E.scalar_literal(1.5) * e_1, E.scalar_literal(2.5) * e_1),
    (
        E.scalar_literal(1.5) * e_1 + E.scalar_literal(5.5) * e_1,
        E.scalar_literal(7.0) * e_1,
    ),
    (
        E.scalar_literal(1.5) * e_1 - E.scalar_literal(0.1) * e_1,
        E.scalar_literal(1.4) * e_1,
    ),
    (
        E.scalar_literal(1.5) * e_1 + E.scalar_literal(-0.1) * e_1,
        E.scalar_literal(1.4) * e_1,
    ),
    (
        E.grade(e_0 * e_1 * e_2 * e_3 * e_4),
        E.scalar_literal(5.0),
    ),
]


@pytest.mark.parametrize("equation", test_equations)
def test_equalities(equation):
    ga.egraph.push()

    caught = None

    try:
        should_be_equal = True
        eq_len = len(equation)
        if eq_len == 2:
            lhs, rhs = equation
        elif eq_len == 3:
            lhs, rhs, should_be_equal = equation
        else:
            raise ValueError("Invalid number of values in equation tuple")

        stop_cond = None
        if should_be_equal:
            stop_cond = eq(lhs).to(rhs)

        ga.egraph.register(lhs, rhs)
        run_ruleset(ga, options=RunRulesetOptions(limit=100, until=stop_cond))

        # Check if LHS is (not) equal to RHS, don't run any additional steps
        check_passes = check_equality(
            ga, lhs, rhs, options=CheckEqualityOptions(limit=0, equal=should_be_equal)
        )

        assert check_passes

        # Check against some basic contradictions
        ga.egraph.check_fail(eq(E.scalar_literal(0.0)).to(E.scalar_literal(1.0)))
        ga.egraph.check_fail(eq(E.scalar_literal(1.0)).to(E.scalar_literal(2.0)))
    except Exception as e:
        caught = e

    ga.egraph.pop()

    if caught is not None:
        raise caught from caught


@pytest.mark.skip
@pytest.mark.parametrize("equations", list(combinations(test_equations, 2)))
def test_contradiction(equations):
    ga.egraph.push()

    caught = None

    try:
        contradictions = [
            eq(E.scalar_literal(0.0)).to(E.scalar_literal(1.0)),
            eq(E.scalar_literal(0.0)).to(E.scalar_literal(2.0)),
            eq(E.scalar_literal(1.0)).to(E.scalar_literal(2.0)),
        ]

        ga.egraph.register(
            equations[0][0], equations[1][0], equations[0][1], equations[1][1]
        )

        run_ruleset(
            ga,
            options=RunRulesetOptions(ruleset="full", limit=9, until=contradictions[0]),
        )

        for contradiction in contradictions:
            ga.egraph.check_fail(contradiction)
    except Exception as e:
        caught = e

    ga.egraph.pop()

    if caught is not None:
        raise caught from caught
