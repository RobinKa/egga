import pytest

from egga.geometric_algebra import GeometricAlgebra
from egga.utils import CheckEqualityOptions, check_equality, simplify


@pytest.fixture
def ga() -> GeometricAlgebra:
    return GeometricAlgebra([1.0, 1.0])


@pytest.fixture
def ga_eq_solve() -> GeometricAlgebra:
    return GeometricAlgebra([1.0, 1.0], eq_solve=True)


def test_simplify(ga: GeometricAlgebra):
    simplified = simplify(ga, ga.basis_vectors[0] * ga.basis_vectors[0])
    expected = ga.expr_cls.scalar_literal(1.0)
    assert str(simplified) == str(expected)


def test_check_equality(ga: GeometricAlgebra):
    lhs = ga.basis_vectors[0] * ga.basis_vectors[0]
    rhs = ga.expr_cls.scalar_literal(1.0)
    not_rhs = ga.expr_cls.scalar_literal(2.0)

    assert check_equality(ga, lhs, rhs)
    assert not check_equality(ga, lhs, not_rhs)

    # Check negated too
    assert not check_equality(ga, lhs, rhs, options=CheckEqualityOptions(equal=False))
    assert check_equality(ga, lhs, not_rhs, options=CheckEqualityOptions(equal=False))
