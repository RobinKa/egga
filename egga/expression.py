from __future__ import annotations

from typing import Protocol

from egglog import StringLike, f64Like, i64Like


class Expression(Protocol):
    def __ne__(self, other: Expression) -> Expression:
        ...

    def __eq__(self, other: Expression) -> Expression:
        ...

    def equal(self, other: Expression) -> Expression:
        ...

    def not_equal(self, other: Expression) -> Expression:
        ...

    def __add__(self, other: Expression) -> Expression:
        ...

    def __sub__(self, other: Expression) -> Expression:
        ...

    def __mul__(self, other: Expression) -> Expression:
        ...

    def __neg__(self) -> Expression:
        ...

    def __invert__(self) -> Expression:
        ...

    def __pow__(self, other: Expression) -> Expression:
        ...

    def __truediv__(self, other: Expression) -> Expression:
        ...

    def __xor__(self, other: Expression) -> Expression:
        ...

    def __or__(self, other: Expression) -> Expression:
        ...

    @staticmethod
    def inverse(other: Expression) -> Expression:
        ...

    @staticmethod
    def boolean(b: i64Like) -> Expression:
        ...

    @staticmethod
    def scalar(value: Expression) -> Expression:
        ...

    @staticmethod
    def scalar_literal(value: f64Like) -> Expression:
        ...

    @staticmethod
    def scalar_variable(value: StringLike) -> Expression:
        ...

    @staticmethod
    def e(s: StringLike) -> Expression:
        ...

    @staticmethod
    def e2(s_1: StringLike, s_2: StringLike) -> Expression:
        ...

    @staticmethod
    def e3(s_1: StringLike, s_2: StringLike, s_3: StringLike) -> Expression:
        ...

    @staticmethod
    def variable(name: StringLike) -> Expression:
        ...

    @staticmethod
    def cos(value: Expression) -> Expression:
        ...

    @staticmethod
    def sin(value: Expression) -> Expression:
        ...

    @staticmethod
    def cosh(value: Expression) -> Expression:
        ...

    @staticmethod
    def sinh(value: Expression) -> Expression:
        ...

    @staticmethod
    def exp(value: Expression) -> Expression:
        ...

    @staticmethod
    def grade(value: Expression) -> Expression:
        ...

    @staticmethod
    def mix_grades(a: Expression, b: Expression) -> Expression:
        ...

    @staticmethod
    def select_grade(value: Expression, grade: Expression) -> Expression:
        ...

    @staticmethod
    def abs_(value: Expression) -> Expression:
        ...

    @staticmethod
    def rotor(basis_blade: Expression, angle: Expression) -> Expression:
        ...

    @staticmethod
    def sandwich(r: Expression, x: Expression) -> Expression:
        ...

    @staticmethod
    def diff(value: Expression, wrt: Expression) -> Expression:
        ...
