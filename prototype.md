# Constraint Validation Layer — Working Prototype

## What This Demonstrates

This prototype bridges the proposal narrative and working code.
It shows that the CVL architecture is not theoretical — the core
validation logic can be expressed in under 80 lines of pure Python,
with zero external dependencies beyond the standard library.

The prototype implements three of the five constraint types
described in the full proposal, using the same state-propagation
pattern the production CVL will use inside `sktime-mcp`.

---

## Core Abstraction: Pipeline State as Sufficient Statistics

Rather than passing full datasets between validation checks,
the CVL tracks a lightweight state dictionary that accumulates
guarantees as each pipeline component is validated.

```python
from dataclasses import dataclass, field
from enum import Enum, auto


class ConstraintType(Enum):
    STATIONARITY    = auto()
    SCITYPE_COMPAT  = auto()
    MISSING_DATA    = auto()


@dataclass
class ComponentSpec:
    """Minimal contract for a pipeline component.

    Parameters
    ----------
    name : str
        Estimator name, matching sktime registry key.
    requires : set of str
        State keys that must be True before this component runs.
    produces : dict
        State keys this component guarantees after running.
    """
    name: str
    requires: set
    produces: dict = field(default_factory=dict)


@dataclass
class ConstraintViolation:
    component: str
    missing: set
    correction_hint: str


def validate_preconditions(
    component: ComponentSpec,
    state: dict,
) -> ConstraintViolation | None:
    """Check that all required state keys are satisfied.

    Parameters
    ----------
    component : ComponentSpec
        Pipeline component being validated.
    state : dict
        Accumulated pipeline state from prior components.

    Returns
    -------
    ConstraintViolation or None
        None if all preconditions are satisfied.
    """
    missing = {k for k in component.requires if not state.get(k)}
    if not missing:
        return None
    hints = {
        "stationary": "Insert Differencer(lags=1) before this component.",
        "no_missing":  "Apply Imputer() before this component.",
        "is_series":   "Ensure input scitype is Series, not Panel or Tabular.",
    }
    hint = " | ".join(hints.get(k, f"Satisfy: {k}") for k in missing)
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
        Initial state derived from data inspection.

    Returns
    -------
    list of ConstraintViolation
        Empty list means the pipeline is valid.
    """
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

## Example 1: ARIMA on Non-Stationary Data

The most common LLM hallucination in `sktime-mcp`. An LLM
composes ARIMA directly on raw data without differencing.
The CVL catches this before sktime ever receives the call.

```python
Differencer = ComponentSpec(
    name="Differencer",
    requires=set(),
    produces={"stationary": True},
)

ARIMA = ComponentSpec(
    name="ARIMA",
    requires={"stationary"},
    produces={"forecast": True},
)

# ADF p-value = 0.43 — series is non-stationary
initial_state = {
    "stationary": False,
    "no_missing":  True,
    "is_series":   True,
}

# LLM hallucination: ARIMA without Differencer
violations = validate_pipeline([ARIMA], initial_state)
# >> ConstraintViolation(component='ARIMA',
# >>   correction_hint='Insert Differencer(lags=1) before this component.')

# Correct pipeline: Differencer then ARIMA
violations = validate_pipeline([Differencer, ARIMA], initial_state)
# >> []  (no violations)
```

---

## Example 2: Missing Data Assumption

```python
Imputer = ComponentSpec(
    name="Imputer",
    requires=set(),
    produces={"no_missing": True},
)

ExponentialSmoothing = ComponentSpec(
    name="ExponentialSmoothing",
    requires={"no_missing"},
    produces={"forecast": True},
)

# Series has NaN gaps
initial_state = {"stationary": True, "no_missing": False, "is_series": True}

# Without Imputer
violations = validate_pipeline([ExponentialSmoothing], initial_state)
# >> correction_hint: 'Apply Imputer() before this component.'

# With Imputer
violations = validate_pipeline([Imputer, ExponentialSmoothing], initial_state)
# >> []
```

---

## Mapping to Production CVL

| Prototype | Production CVL in sktime-mcp |
|---|---|
| `ComponentSpec.requires` | `ConstraintGraph` auto-populated from `cls.get_class_tags()` |
| `initial_state` dict | `ConstraintValidator.validate_dynamic()` with ADF/KPSS tests |
| `ConstraintViolation.correction_hint` | `CorrectionPromptEngine` string template |
| `validate_pipeline()` | Interceptor wired into `fit_predict_tool` (15 new lines) |

The production version replaces hand-coded `requires` sets with
automatic population from sktime's `_tags` system — the same
infrastructure I worked with in PR #9313 (merged).

---

## Connection to Prior Contributions

**PR #229 — sktime-mcp param-key validation (open, under review)**
validates that parameter *names* are valid for a given estimator.
The CVL extends this one tier up: validating that the estimator
is *mathematically appropriate* for the data it will receive.
Same abstraction layer. Next tier of strictness.

**PR #9313 — sktime core `_tags` migration (merged)**
The production CVL reads `cls.get_class_tags()` directly to
auto-populate constraint rules. PR #9313 gave me direct
experience navigating the `_tags` system that makes this work.
