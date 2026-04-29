# Constraint Validation Layer — Working Prototype

## Purpose

This prototype demonstrates that the Constraint Validation Layer
proposed for `sktime-mcp` is not hypothetical. The core validation
logic — typed constraints, stateful pipeline propagation, and
sktime `_tags` integration — can be expressed in under 120 lines
of pure Python with zero external dependencies.

This is the direct extension of
[PR #229](https://github.com/sktime/sktime-mcp/pull/229)
(parameter-key validation in `instantiate_estimator`):

> PR #229 validates that parameter *names* are valid for an estimator.
> The CVL validates that the estimator is *mathematically appropriate*
> for the data it will receive. Same abstraction layer. Next tier of strictness.

---

## State Schema

Pipeline state is a typed dictionary accumulating guarantees
as each component is validated. Keys are drawn from a fixed schema
to prevent silent misuse.

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


ALLOWED_STATE_KEYS = {
    "stationary",    # bool  — series passes ADF + KPSS
    "no_missing",    # bool  — series contains no NaN values
    "is_series",     # bool  — input scitype is Series (not Panel)
    "n_obs",         # int   — number of observations in the series
    "has_exog",      # bool  — exogenous variable X is present
    "exog_2d",       # bool  — X has shape (n, k), k >= 1
}


def validate_state(state: dict) -> None:
    """Enforce schema discipline on pipeline state.

    Parameters
    ----------
    state : dict
        Pipeline state dictionary to validate.

    Raises
    ------
    ValueError
        If any key is not in ALLOWED_STATE_KEYS.
    """
    unknown = set(state.keys()) - ALLOWED_STATE_KEYS
    if unknown:
        raise ValueError(
            f"Unknown state keys: {unknown}. "
            f"Allowed keys: {ALLOWED_STATE_KEYS}"
        )
```

---

## Typed Constraint Objects

Constraints are not boolean flags — they are typed predicate
objects that carry a description and a callable test.
This makes them extensible to numeric, relational, and
probabilistic constraints in the production CVL.

```python
@dataclass
class Constraint:
    """A typed precondition on pipeline state.

    Parameters
    ----------
    key : str
        State key this constraint tests. Must be in ALLOWED_STATE_KEYS.
    predicate : Callable[[Any], bool]
        Function that returns True when the constraint is satisfied.
    description : str
        Human-readable correction hint returned to the LLM caller.
    """
    key: str
    predicate: Callable[[Any], bool]
    description: str


@dataclass
class ConstraintViolation:
    """Result of a failed pre-execution constraint check.

    Parameters
    ----------
    component : str
        Name of the estimator that raised the violation.
    missing : set of str
        State keys that were not satisfied.
    correction_hint : str
        Actionable message returned to the LLM or developer.
    """
    component: str
    missing: set
    correction_hint: str
```

---

## Component Contracts

Each estimator declares a typed contract: what it requires
from the pipeline state and what it guarantees after running.
In the production CVL, these contracts are auto-generated
from `cls.get_class_tags()` — see the integration stub below.

```python
@dataclass
class ComponentSpec:
    """Pipeline component contract.

    Parameters
    ----------
    name : str
        Estimator name matching the sktime registry key.
    requires : list of Constraint
        Typed preconditions that must hold before this component runs.
    produces : dict
        State keys this component guarantees after successful execution.
    """
    name: str
    requires: list
    produces: dict = field(default_factory=dict)
```

---

## Validation Logic

```python
def validate_preconditions(
    component: ComponentSpec,
    state: dict,
) -> ConstraintViolation | None:
    """Check all typed preconditions for a single component.

    Parameters
    ----------
    component : ComponentSpec
        Component being validated.
    state : dict
        Accumulated pipeline state.

    Returns
    -------
    ConstraintViolation or None
        None if all constraints are satisfied.
    """
    failures = []
    for constraint in component.requires:
        value = state.get(constraint.key)
        if value is None or not constraint.predicate(value):
            failures.append((constraint.key, constraint.description))

    if not failures:
        return None

    missing = {k for k, _ in failures}
    hint = " | ".join(desc for _, desc in failures)
    return ConstraintViolation(
        component=component.name,
        missing=missing,
        correction_hint=hint,
    )


def validate_pipeline(
    components: list,
    initial_state: dict,
) -> list:
    """Run pre-execution validation across all pipeline components.

    Parameters
    ----------
    components : list of ComponentSpec
        Ordered list of pipeline components.
    initial_state : dict
        Initial state derived from data inspection
        (e.g. stationarity test result, NaN check, observation count).

    Returns
    -------
    list of ConstraintViolation
        Empty list means the pipeline is valid and safe to execute.
    """
    validate_state(initial_state)
    state = initial_state.copy()
    violations = []
    for component in components:
        violation = validate_preconditions(component, state)
        if violation:
            violations.append(violation)
        else:
            state.update(component.produces)
    return violations
```

---

## sktime `_tags` Integration Stub

This is the production bridge. In `sktime-mcp`, the
`ConstraintGraph` will call this at initialization — reading
live estimator metadata instead of hand-coded contracts.
The same `_tags` system I worked with in
[PR #9313](https://github.com/sktime/sktime/pull/9313) (merged).

```python
def from_sktime_estimator(cls) -> ComponentSpec:
    """Build a ComponentSpec from a sktime estimator's _tags metadata.

    This is the production CVL bridge: constraints are derived
    automatically from the estimator's declared capabilities,
    not hand-coded per estimator.

    Parameters
    ----------
    cls : sktime estimator class
        Any class with a get_class_tags() method.

    Returns
    -------
    ComponentSpec
        Typed contract derived from the estimator's _tags.

    Examples
    --------
    >>> from sktime.forecasting.arima import ARIMA
    >>> spec = from_sktime_estimator(ARIMA)
    >>> spec.name
    'ARIMA'
    """
    tags = cls.get_class_tags()
    requires = []

    if tags.get("requires_stationarity", False):
        requires.append(Constraint(
            key="stationary",
            predicate=lambda x: x is True,
            description=(
                "Series must be stationary (ADF p < 0.05, KPSS p > 0.05). "
                "Insert Differencer(lags=d) before this estimator."
            ),
        ))

    # maps sktime tag "handles-missing-data" -> CVL state key "no_missing"
    if not tags.get("handles-missing-data", True):
        requires.append(Constraint(
            key="no_missing",
            predicate=lambda x: x is True,
            description="Series contains NaN values. Apply Imputer() first.",
        ))

    if tags.get("capability:exogenous", False):
        requires.append(Constraint(
            key="exog_2d",
            predicate=lambda x: x is True,
            description=(
                "Exogenous variable X must be 2-D with shape (n, k). "
                "Reshape via X.reshape(-1, 1) if 1-D."
            ),
        ))

    return ComponentSpec(
        name=cls.__name__,
        requires=requires,
        produces={},
    )
```

---

## Example 1: ARIMA on Non-Stationary Data

The most common LLM hallucination in `sktime-mcp`. An LLM
composes ARIMA directly on raw I(2) data without differencing.

```python
Differencer = ComponentSpec(
    name="Differencer",
    requires=[],
    produces={"stationary": True},
)

ARIMA = ComponentSpec(
    name="ARIMA",
    requires=[
        Constraint(
            key="stationary",
            predicate=lambda x: x is True,
            description=(
                "Input must be I(0) covariance-stationary. "
                "Insert Differencer(lags=1) before ARIMA."
            ),
        ),
        Constraint(
            key="n_obs",
            predicate=lambda x: x > 30,
            description=(
                "ARIMA requires at least 30 observations for "
                "reliable parameter estimation."
            ),
        ),
    ],
    produces={"forecast": True},
)

# ADF p-value = 0.43 — series is non-stationary, 120 observations
initial_state = {
    "stationary": False,
    "no_missing":  True,
    "is_series":   True,
    "n_obs":       120,
}

# LLM hallucination: ARIMA without Differencer
violations = validate_pipeline([ARIMA], initial_state)
# Returns:
# ConstraintViolation(
#   component='ARIMA',
#   missing={'stationary'},
#   correction_hint='Input must be I(0) covariance-stationary.
#                    Insert Differencer(lags=1) before ARIMA.'
# )

# Correct pipeline: Differencer then ARIMA
violations = validate_pipeline([Differencer, ARIMA], initial_state)
# Returns: []  — pipeline is valid, safe to dispatch to sktime
```

---

## Example 2: Observation Count Constraint

Numeric constraints catch failures that boolean flags cannot.
An LLM may compose ARIMA on a short series where parameter
estimation is unreliable.

```python
short_state = {
    "stationary": True,
    "no_missing":  True,
    "is_series":   True,
    "n_obs":       12,   # too short for ARIMA
}

violations = validate_pipeline([ARIMA], short_state)
# Returns:
# ConstraintViolation(
#   component='ARIMA',
#   missing={'n_obs'},
#   correction_hint='ARIMA requires at least 30 observations...'
# )
```

---

## Example 3: Exogenous Shape Contract

```python
SARIMAX = ComponentSpec(
    name="SARIMAX",
    requires=[
        Constraint(
            key="stationary",
            predicate=lambda x: x is True,
            description="Insert Differencer before SARIMAX.",
        ),
        Constraint(
            key="exog_2d",
            predicate=lambda x: x is True,
            description=(
                "X must be 2-D. Reshape via X.reshape(-1, 1) if 1-D."
            ),
        ),
    ],
    produces={"forecast": True},
)

# LLM passed X as 1-D array
state = {
    "stationary": True,
    "no_missing":  True,
    "is_series":   True,
    "n_obs":       200,
    "exog_2d":     False,   # X.shape = (200,) not (200, 1)
}

violations = validate_pipeline([SARIMAX], state)
# Returns correction_hint: 'X must be 2-D. Reshape via X.reshape(-1, 1).'
```

---

## Runtime Integration Sketch

The CVL sits at the MCP tool boundary, validating pipelines
before dispatch to sktime. This is the 15-line interceptor
described in the proposal's Section 2.5:

```python
def validated_fit_predict(pipeline, data, validate=True, dry_run=False):
    if not validate:
        return fit_predict_tool(pipeline, data)

    state = derive_initial_state(data)  # runs ADF/KPSS, NaN check, shape check
    violations = validate_pipeline(pipeline, state)

    if dry_run:
        return {
            "dry_run": True,
            "violations": [v.correction_hint for v in violations],
        }

    if violations:
        return {"error": [v.correction_hint for v in violations]}

    return fit_predict_tool(pipeline, data)
```

Invalid pipelines are never executed. Corrections are surfaced
immediately to the LLM agent as structured text — identical in
form to any other MCP tool response.

---

## Computational Overhead

Validation runs in O(n_components) time — each constraint is a
constant-time predicate on the state dictionary. Compared to
model fitting (O(n log n) for stationarity tests, O(n²) or
higher for ARIMA), the CVL adds negligible overhead while
preventing costly invalid executions from ever reaching sktime.

---

## Failure Modes and Design Boundaries

This prototype intentionally simplifies several aspects of the
production CVL. These are deliberate trade-offs to isolate the
core abstraction: stateful, pre-execution constraint validation.

**Constraints are currently predicate-only.**
Production CVL will support relational constraints between
components — e.g., SARIMAX's seasonal period must match the
series frequency detected at data inspection time.

**No conflict resolution.**
If two components produce incompatible guarantees, resolution
will be handled at the tool level, not inside the validator.

**Static contracts in this prototype.**
The production CVL derives contracts dynamically from
`cls.get_class_tags()` at initialization — shown in the
integration stub above but not wired to live sktime here.

**No probabilistic reasoning yet.**
ADF/KPSS stationarity tests introduce a confidence dimension.
The production CVL treats ambiguous test results (ADF and KPSS
disagree) as `WARNING` severity rather than `ERROR`, ensuring
the validator never blocks a valid pipeline on a false positive.

---

## Mapping to Production CVL in sktime-mcp

| Prototype | Production CVL |
|---|---|
| `Constraint` dataclass | Typed constraint objects derived from `cls.get_class_tags()` |
| `ALLOWED_STATE_KEYS` schema | `ValidationResult` dataclass with structured fields |
| `from_sktime_estimator()` stub | `build_constraint_graph()` reading live registry at import |
| `validate_pipeline()` | `ConstraintValidator.validate_dynamic()` with ADF/KPSS |
| `ConstraintViolation.correction_hint` | `CorrectionPromptEngine` string template returned via MCP |
| `validate_state()` guard | Schema enforcement at `fit_predict_tool` boundary |
| `validated_fit_predict()` sketch | Full interceptor wired into `fit_predict_tool` (15 new lines) |

---

## Key Takeaway

The CVL is not an added layer on top of `sktime-mcp` — it is
a natural extension of sktime's `_tags` system into a
constraint-aware execution model for agentic pipelines.

The `_tags` system already encodes what every estimator requires
and what it can handle. The CVL makes those requirements
executable at the MCP boundary, before a single line of sktime
code runs. That is the entire idea — and this prototype shows
it works.
