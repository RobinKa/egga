from dataclasses import dataclass
from typing import Callable, Literal, Optional, Union

from egglog import Fact, egraph, eq, run

from egga.expression import Expression
from egga.geometric_algebra import GeometricAlgebra


@dataclass
class RunRulesetOptions:
    ruleset: Union[Literal["fast", "medium", "full"], egraph.Ruleset] = "full"
    limit: int = 10
    until: Optional[Fact] = None


def run_ruleset(ga: GeometricAlgebra, options: RunRulesetOptions = RunRulesetOptions()):
    if isinstance(options.ruleset, str):
        ruleset = ga.rulesets.__dict__[options.ruleset]
    else:
        ruleset = options.ruleset

    return ga.egraph.run(
        run(
            ruleset,
            options.limit,
            *([options.until] if options.until is not None else []),
        ).saturate()
    )


@dataclass
class RunScheduledOptions:
    limit: int = 10
    full_interval: int = 4
    fast_count: int = 1
    until: Optional[Fact] = None
    on_step: Optional[Callable] = None


def run_scheduled(
    ga: GeometricAlgebra, options: RunScheduledOptions = RunScheduledOptions()
):
    def do_step(ruleset: egraph.Ruleset, step_limit: int):
        run_ruleset(
            ga,
            options=RunRulesetOptions(
                ruleset=ruleset, limit=step_limit, until=options.until
            ),
        )

    step = 0
    while step < options.limit:
        # Fast step
        do_step(ruleset=ga.rulesets.fast, step_limit=options.fast_count)

        if options.on_step is not None:
            if options.on_step():
                break

        # Medium or full step
        do_step(
            ruleset=ga.rulesets.full
            if step % options.full_interval != 0
            else ga.rulesets.medium,
            step_limit=1,
        )

        if options.on_step is not None:
            if options.on_step():
                break

        step += 1


def simplify(
    ga: GeometricAlgebra,
    expression: Expression,
    options: RunScheduledOptions = RunScheduledOptions(),
) -> Expression:
    """
    Returns a simplified expression.
    """
    ga.egraph.push()

    ga.egraph.register(expression)
    run_scheduled(ga=ga, options=options)
    simplified = ga.egraph.extract(expression)

    ga.egraph.pop()

    return simplified


@dataclass
class CheckEqualityOptions:
    limit: int = 10
    full_interval: int = 4
    fast_count: int = 4
    equal: bool = True
    on_step: Optional[Callable] = None


def check_equality(
    ga: GeometricAlgebra,
    lhs: Expression,
    rhs: Expression,
    options: CheckEqualityOptions = CheckEqualityOptions(),
) -> bool:
    """
    Returns whether two expressions are equal.
    """
    ga.egraph.push()

    ga.egraph.register(lhs)
    ga.egraph.register(rhs)
    predicate = eq(lhs).to(rhs)
    run_scheduled(
        ga=ga,
        options=RunScheduledOptions(
            limit=options.limit,
            full_interval=options.full_interval,
            fast_count=options.fast_count,
            until=predicate if options.equal else None,
            on_step=options.on_step,
        ),
    )
    check_fn = ga.egraph.check if options.equal else ga.egraph.check_fail
    check_passed = True
    try:
        check_fn(predicate)
    except:
        check_passed = False

    ga.egraph.pop()
    return check_passed
