# esoc-2026-sktime-mcp-proposal
Internship proposal for the ESoC 2026 sktime agentic track (Constraint Validation Layer).

ESoC 2026 — Batch 2  |  sktime / GC.OS
Project Proposal — sktime-mcp Repository
Semantic Pipeline Synthesis &
Constraint Resolution for sktime-mcp
Applicant	Mohit Kumar  |  BS Data Science & Applications, IIT Madras
Email	25f1001180@ds.study.iitm.ac.in
GitHub	github.com/Mohit25f101
Organization	sktime / GC.OS  —  sktime-mcp
Availability	35–40 hrs/week  |  IST (UTC+5:30)  |  Full summer commitment
Merged PRs	sktime #9313 (merged), #9286 ( almost merged), sktime-mcp param-key validation (merged)



Abstract
sktime-mcp allows Large Language Models to access sktime’s rich estimator registry through the Model Context
Protocol. The current architecture does discovery and instantiation correctly — but there's a
critical gap: no mechanism to prevent, detect or fix the statistically invalid pipeline
compositions that LLMs tend to hallucinate. Classifier, ARIMA on a non-stationary series
inserted in a forecasting pipeline, SARIMAX getting a malformed exogenous tensor — all propagate
silently to sktime, raising opaque internal exceptions that the LLM cannot interpret or self-correct.
 
This proposal introduces the Constraint Validation Layer (CVL): a single, surgically scoped module
within sktime-mcp that detects invalid pipelines before execution, maps violations to a typed
constraint taxonomy, and produces structured, deterministic correction context that any caller — LLM
or human — can take action immediately. The CVL is purposefully narrow, protocol agnostic, and built
by the community through sktime's existing _tags infrastructure rather than a
hand-coded per-estimator register» It’s not a new system it’s the missing validation tier
which means that the current system is production-ready.




00  Design Philosophy: Five Architectural Decisions That Resolve Reviewer Concerns

All the design choices of this proposal were made with the five most common objections to constraint-layer architectures in mind. This section explicitly states each concern and resolves it at the architectural level, not as a risk mitigation afterthought, but as a first-class design principle.

Decision 1 — One Abstraction, Not Five Systems  [Resolves: Scope Concern]
A naive reading of 'ConstraintGraph + ConstraintValidator + SemanticErrorRouter + CorrectionPromptEngine + exception catalog' suggests five independent systems requiring five separate development tracks. This reading is wrong. The Constraint Validation Layer is one abstraction — a pre-execution guard function — that decomposes naturally into four thin modules:
•	ConstraintGraph — a Python dict. ~80 lines. Populated from sktime's existing _tags system, not hand-written per estimator.
•	ConstraintValidator — the core logic. ~150 lines. Two public methods: validate_static() and validate_dynamic().
•	SemanticErrorRouter — a regex dispatch table. ~50 lines. Maps exception message patterns to ConstraintType enums.
•	CorrectionPromptEngine — a string template dict. ~60 lines. No ML, no network calls, no external dependencies.

Scope Reality Check
Total estimated implementation: ~340 lines of pure Python in 4 files.
These are weeks 3 through 9. Weeks 1-2: Community bonding + Exception catalog.
Weeks 10-12: Integration, Testing and Documentation
The timeline is not ambitious, it is prudent.

Decision 2 — Zero LLM Dependency  [Resolves: LLM Coupling Concern]
The CorrectionPromptEngine does not invoke any LLM. It does not depend on Claude, Gpt-4, or any particular model. It is a pure python function to format a string template and a constraint violation data. The output is a structured text payload returned by the MCP tool, and is in the same form as any other MCP tool response. If the call is from an LLM it reads a correction message. If the caller is a human developer, they get the same message. The system is designed to be caller-neutral. LLM behavior changes are orthogonal to its correctness.

Concern (Rebutted)
Concern: “Self-healing pipeline could break if Claude or GPT-4 change behavior.”
 
REBUTTAL: The CVL knows nothing about which LLM (if any) is calling it. It gets
a pipeline spec as a Python dict and returns a ValidationResult as a Python dict.
dataclass() The MCP server serializes this into JSON. No LLM API is ever used.
The 'self-healing' behavior is a feature of the LLM reading a structured correction
message – just like a human reading a compiler error.” The compiler is not dependent
on the programmer's mental picture. CVL doesn't either.

Decision 3 — Tag-Driven, Auto-Scaling Graph  [Resolves: Maintenance Concern]
Rules are not per estimator in the ConstraintGraph. It contains rules for each constraint type and constraint types correspond to existing _tags keys in sktime. The ConstraintGraph automatically inherits this when sktime adds a new estimator setting ‘requires_fh’: True or ‘handles-missing-data’: False to its _tags dict via a tag-reader at initialization time. The maintainer does not write new graph entries.

# The graph is NOT a hard-coded dict of estimator names.
# It is a tag-reader that builds constraints from sktime's own metadata.

def build_constraint_graph() -> ConstraintGraph:
    graph = ConstraintGraph()
    registry = dict(all_estimators())

    for name, cls in registry.items():
        tags = cls.get_class_tags()  # sktime's existing tag system

        # Stationarity: ARIMA, SARIMAX, VAR all set 'requires_stationary': True
        if tags.get('requires_stationary', False):
            graph.register(name, Constraint(
                constraint_type=ConstraintType.STATIONARITY,
                estimator_name=name,
                precondition='Input series must be I(0) (covariance-stationary).',
                correction_hint='Insert Differencer before this estimator.',
            ))

        # Missing data: flag estimators where 'handles-missing-data' is False
        if not tags.get('handles-missing-data', True):
            graph.register(name, Constraint(
                constraint_type=ConstraintType.MISSING_DATA,
                estimator_name=name,
                precondition='Input series must have no NaN values.',
                correction_hint='Apply Imputer before this estimator.',
            ))

        # Scitype: derived from 'scitype:y' and 'scitype:y_inner' tags
        ...

    return graph

# When sktime adds NewARIMAVariant with requires_stationary=True,
# it gets the stationarity constraint for free. Zero maintenance.

Maintenance Reality Check
Constraints are written ONCE per constraint type (~5 total) not per estimator (200+).
The ConstraintGraph populates itself from sktime's own _tags metadata at start-up.
The CVL does not have to be changed when adding a new estimator to sktime.
The only maintenance required is to add new ConstraintTypes for new categories of
* mathematical constraint-a rare occurrence that requires a deliberate design discussion.

Decision 4 — Explicit Escape Hatches & Dry-Run Mode  [Resolves: Debugging Concern]
A validator that silently blocks valid pipelines is worse than no validator at all. The CVL is therefore opt-out by design. Every MCP tool that uses it exposes two flags:

# Every CVL-enabled tool accepts these two parameters:

def fit_predict_tool(
    estimator_handle,
    dataset=None,
    horizon=12,
    validate=True,    # set False to bypass CVL entirely — no performance cost
    dry_run=False,    # set True to return what WOULD be blocked without blocking
):
    if not validate:
        return _execute_sktime_directly(estimator_handle, dataset, horizon)

    result = _validator.validate(estimator_handle, dataset)

    if dry_run:
        return {
            'dry_run': True,
            'would_block': not result.is_valid,
            'violations': [v.to_dict() for v in result.violations],
            'warnings':   [w.to_dict() for w in result.warnings],
        }

    if not result.is_valid:
        return _format_correction_response(result)

    return _execute_sktime_directly(estimator_handle, dataset, horizon)

Developer Experience Guarantee
validate=True (default): CVL is on. Invalid pipelines return correction context in a structured way.
validate=False: Skip CVL. Sktime behaviour raw, no change.
dry_run= True: CVL runs, but does not block. what returns audit report for
would have been recognised. Good for debugging false positives.
 
If the CVL ever returns a false positive, the developer sets dry_run=True, files a
GitHub issue with the audit report. The fix is a one line tag correction in sktime.

Decision 5 — Protocol-Agnostic Core, Thin MCP Adapter  [Resolves: MCP Maturity Concern]
The full Constraint Validation Layer does not have any imports from any MCP library. All in pure-python: dataclasses, numpy and statsmodels. The only MCP-specific code is in one adapter function in each tool file that calls the CVL and serializes the ValidationResult to JSON. If the spec of MCP changes tomorrow, CVL does not care. The adapter changes, but the logic is the same.

Portability Design
CVL imports: dataclasses, numpy, statsmodels, sktime.registry — zero MCP dependencies.

MCP adapter (per tool, ~10 lines): calls CVL, serializes ValidationResult to JSON,
returns via MCP response envelope.

If MCP v2 changes response format: update 1 adapter function per tool. CVL untouched.
If MCP is deprecated entirely: CVL is extracted as standalone sktime-constraints package
with zero refactoring. The constraint logic has value independent of the transport layer.



01  The Core Problem: The Validation Gap in sktime-mcp


1.1  What the Current Architecture Gets Right
sktime-mcp's registry-first design is correct. The MCP tools for discovery (list_estimators, describe_estimator) and instantiation (instantiate_estimator) form a clean, minimal API surface. The recent parameter key validation work (merged PR) demonstrates that the maintainers are already moving toward stricter pre-execution contracts. The CVL is the logical next tier of that validation stack.

1.2  The Validation Gap
Parameter key validation confirms that param names are valid. It does not confirm that the chosen estimator is mathematically appropriate for the data. The gap between 'valid parameters' and 'valid pipeline' is where LLM hallucinations live:

Hallucination Category	Example	Why Current MCP Cannot Catch It
Stationarity Violation	ARIMA on raw I(2) series	Params are valid. Registry lookup succeeds. Failure occurs inside statsmodels at fit-time with a cryptic exception.
Scitype Mismatch	Classifier in forecasting pipeline	Both estimators exist in registry. No cross-component compatibility check exists at the MCP layer.
Exogenous Shape Error	SARIMAX with X.shape=(n,)	Parameter type is correct (array). Shape contract (2-D) is not enforced at the MCP boundary.
Missing Data Assumption	ARIMA on series with NaN gaps	No pre-flight check against the estimator's handles-missing-data tag before dispatching to sktime.



02  The Constraint Validation Layer: Full Architecture


2.1  Module Structure
The CVL is a new top-level module: src/sktime_mcp/constraints/. It contains four files, each with a single responsibility. The total implementation is approximately 340 lines of pure Python — deliberately kept small to maximize maintainability and reviewer confidence.

src/sktime_mcp/
├── constraints/
│   ├── __init__.py         # Public API: ConstraintValidator, ValidationResult
│   ├── graph.py            # ConstraintGraph, Constraint dataclass, ConstraintType enum
│   ├── validator.py        # ConstraintValidator: validate_static(), validate_dynamic()
│   ├── error_router.py     # SemanticErrorRouter: exception -> ConstraintType mapping
│   └── prompt_engine.py    # CorrectionPromptEngine: ConstraintType -> correction string
├── tools/
│   ├── fit_predict.py      # MODIFIED: calls ConstraintValidator, 15 new lines
│   └── instantiate.py      # ALREADY MODIFIED (merged PR): param key validation
└── tests/
    └── constraints/
        ├── test_graph.py
        ├── test_validator.py
        ├── test_error_router.py
        └── test_prompt_engine.py


2.2  graph.py — The ConstraintGraph
As established in Section 0, the ConstraintGraph populates itself from sktime's _tags system at initialization. The graph.py file defines the data structures; build_constraint_graph() is the constructor that reads live estimator metadata.

# src/sktime_mcp/constraints/graph.py

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional
from sktime.registry import all_estimators

class ConstraintType(Enum):
    STATIONARITY       = auto()  # covariance-stationarity precondition
    SCITYPE_COMPAT     = auto()  # input/output scitype compatibility
    EXOGENOUS_SHAPE    = auto()  # X tensor dimensionality contract
    MISSING_DATA       = auto()  # NaN-tolerance of estimator
    SEASONALITY_PERIOD = auto()  # seasonal period detectability

@dataclass
class Constraint:
    constraint_type : ConstraintType
    estimator_name  : str
    precondition    : str   # human-readable mathematical predicate
    correction_hint : str   # actionable fix description
    severity        : str = 'ERROR'  # ERROR | WARNING
    correction_ctx  : dict = field(default_factory=dict)  # runtime data

    def to_dict(self) -> dict:
        return {
            'constraint_type': self.constraint_type.name,
            'estimator': self.estimator_name,
            'precondition': self.precondition,
            'correction_hint': self.correction_hint,
            'severity': self.severity,
        }

class ConstraintGraph:
    def __init__(self):
        self._nodes: Dict[str, List[Constraint]] = {}

    def register(self, estimator: str, c: Constraint) -> None:
        self._nodes.setdefault(estimator, []).append(c)

    def get(self, estimator: str) -> List[Constraint]:
        return self._nodes.get(estimator, [])

    def get_pipeline(self, components: List[str]) -> List[Constraint]:
        out = []
        for c in components: out.extend(self.get(c))
        out.extend(self._cross_scitype_constraints(components))
        return out

    def _cross_scitype_constraints(self, components):
        # Flags pipelines whose final component scitype != 'forecaster'
        # when called from a forecasting context.
        ...

def build_constraint_graph() -> ConstraintGraph:
    graph = ConstraintGraph()
    for name, cls in dict(all_estimators()).items():
        tags = cls.get_class_tags()
        if tags.get('requires_stationary', False):
            graph.register(name, Constraint(
                ConstraintType.STATIONARITY, name,
                'Input must be I(0) covariance-stationary.',
                'Insert Differencer(lags=d) before this estimator.',
            ))
        if not tags.get('handles-missing-data', True):
            graph.register(name, Constraint(
                ConstraintType.MISSING_DATA, name,
                'Input must contain no NaN values.',
                'Apply Imputer before this estimator.',
            ))
        # ... scitype, seasonal, exogenous tags handled similarly
    return graph


2.3  validator.py — The ConstraintValidator

# src/sktime_mcp/constraints/validator.py

import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from dataclasses import dataclass
from typing import List, Optional
from .graph import ConstraintGraph, Constraint, ConstraintType

@dataclass
class ValidationResult:
    is_valid   : bool
    violations : List[Constraint]
    warnings   : List[Constraint]

    def to_dict(self) -> dict:
        return {
            'is_valid'  : self.is_valid,
            'violations': [v.to_dict() for v in self.violations],
            'warnings'  : [w.to_dict() for w in self.warnings],
        }

class ConstraintValidator:
    def __init__(self, graph: ConstraintGraph):
        self._graph = graph

    def validate_static(self, components: List[str]) -> ValidationResult:
        """No data required. Checks tag-derived constraints only. < 50ms."""
        violations, warnings = [], []
        for c in self._graph.get_pipeline(components):
            if c.test_callable is None:  # static constraint — no data test
                (violations if c.severity == 'ERROR' else warnings).append(c)
        return ValidationResult(not violations, violations, warnings)

    def validate_dynamic(self, components: List[str],
                         y: np.ndarray,
                         X: Optional[np.ndarray] = None) -> ValidationResult:
        """Data-aware validation. Runs statistical tests on y and X."""
        violations, warnings = [], []
        for c in self._graph.get_pipeline(components):
            if c.constraint_type == ConstraintType.STATIONARITY:
                d = self._check_stationarity(y)
                if d > 0:
                    import dataclasses
                    c = dataclasses.replace(c, correction_ctx={
                        'recommended_d': d,
                        'adf_pvalue'   : round(adfuller(y,autolag='AIC')[1], 4),
                    })
                    violations.append(c)
            elif c.constraint_type == ConstraintType.EXOGENOUS_SHAPE and X is not None:
                if X.ndim != 2:
                    violations.append(dataclasses.replace(c, correction_ctx={
                        'got_shape': str(X.shape),
                        'fix': f'X.reshape(-1,1)' if X.ndim==1 else 'Reduce to 2-D',
                    }))
            elif c.constraint_type == ConstraintType.MISSING_DATA:
                if np.isnan(y).any():
                    violations.append(c)
        return ValidationResult(not violations, violations, warnings)

    def _check_stationarity(self, y: np.ndarray) -> int:
        """Return estimated integration order I(d), d in {0,1,2}."""
        for d in range(3):
            series = np.diff(y, n=d) if d > 0 else y
            if adfuller(series, autolag='AIC')[1] < 0.05: return d
        return 2


2.4  error_router.py & prompt_engine.py

# src/sktime_mcp/constraints/error_router.py  (~50 lines)

import re
from .graph import ConstraintType

PATTERNS = [
    (re.compile(r'(not stationar|unit root|I\(\d\))',  re.I), ConstraintType.STATIONARITY),
    (re.compile(r'(scitype|mtype|expected.*Series.*got)', re.I), ConstraintType.SCITYPE_COMPAT),
    (re.compile(r'(exog|X.*shape|n_features)',            re.I), ConstraintType.EXOGENOUS_SHAPE),
    (re.compile(r'(NaN|null|missing)',                    re.I), ConstraintType.MISSING_DATA),
]

class SemanticErrorRouter:
    def route(self, exc: Exception) -> tuple:
        msg = str(exc)
        for pattern, ctype in PATTERNS:
            if pattern.search(msg): return ctype, msg
        return None, msg  # unrecognized — re-raise unchanged

# src/sktime_mcp/constraints/prompt_engine.py  (~60 lines)

from .graph import ConstraintType

# Templates are caller-agnostic strings — no LLM API calls, no external deps.
TEMPLATES = {
    ConstraintType.STATIONARITY: (
        'CONSTRAINT VIOLATION [{estimator}]: Input series is not stationary (I({recommended_d})). '
        'ADF p-value: {adf_pvalue}. '
        'REQUIRED ACTION: Insert Differencer(lags={recommended_d}) as first pipeline component. '
        'CORRECTED CALL: instantiate_pipeline(["Differencer", "{estimator}"], '
        '[{{"lags": {recommended_d}}}, {{original_params}}])'
    ),
    ConstraintType.EXOGENOUS_SHAPE: (
        'CONSTRAINT VIOLATION [{estimator}]: X has shape {got_shape}. '
        'Requires 2-D array with shape (n_samples, n_features). '
        'REQUIRED ACTION: {fix}'
    ),
    ConstraintType.MISSING_DATA: (
        'CONSTRAINT VIOLATION [{estimator}]: Input series contains NaN values. '
        'REQUIRED ACTION: Insert Imputer() before this estimator.'
    ),
}

class CorrectionPromptEngine:
    def generate(self, ctype: ConstraintType, ctx: dict) -> str:
        template = TEMPLATES.get(ctype)
        if not template: return f'Unclassified constraint violation. Context: {ctx}'
        try: return template.format(**ctx)
        except KeyError as e: return f'Template key missing: {e}. Context: {ctx}'


2.5  Integration into fit_predict (15 New Lines)
The entire CVL integrates into the existing fit_predict tool with 15 new lines. The public tool signature gains two optional flags (validate, dry_run) and the execution path is extended with two guard stages:

# src/sktime_mcp/tools/fit_predict.py — DIFF ONLY

+ from sktime_mcp.constraints import ConstraintValidator, build_constraint_graph
+ from sktime_mcp.constraints.error_router import SemanticErrorRouter
+ from sktime_mcp.constraints.prompt_engine import CorrectionPromptEngine
+ _validator = ConstraintValidator(build_constraint_graph())  # built once at import
+ _router    = SemanticErrorRouter()
+ _engine    = CorrectionPromptEngine()

  def fit_predict_tool(handle, dataset=None, horizon=12,
+                      validate=True, dry_run=False):

+     if validate:
+         components = _resolve_components(handle)
+         y, X = _resolve_data(dataset)
+         result = _validator.validate_dynamic(components, y, X)
+         if dry_run: return {'dry_run': True, **result.to_dict()}
+         if not result.is_valid:
+             ctype = result.violations[0].constraint_type
+             return {'success': False, 'error_type': ctype.name,
+                     'correction': _engine.generate(ctype, result.violations[0].correction_ctx)}

      try:
          return _execute_sktime(handle, y, X, horizon)  # unchanged
+     except Exception as exc:
+         ctype, msg = _router.route(exc)
+         if ctype: return {'success': False, 'error_type': ctype.name,
+                           'correction': _engine.generate(ctype, {'raw': msg})}
          raise



03  Statistical Depth: Mathematical Constraints Encoded from First Principles


3.1  Stationarity — ADF + KPSS Confirmatory Strategy
Formal Definition: Covariance Stationarity
A process {y t } is covariance-stationary iff for all t:
  (1) E[y_t] = mu (mean is stationary)
  (2) Var[y_t] = sigma^2 (constant finite variance)
  (3) Cov[y_t , y_{t-h} ] = gamma(h) (autocovariance is a function of lag h only)
 
CVL Strategy: Dual testing confirmation to reduce false positive rate
  ADF: H_0 = unit root (non-stationary). If p < 0.05, reject --> evidence of stationarity.
  KPSS: H_0 = stationary. If p < 0.05 we reject --> evidence of non-stationarity.
  Conclusive stationarity ADF p < 0.05 AND KPSS p > 0.05
  Ambiguous case (both reject): mark WARNING, not ERROR. Do not block execution.
 
Integration order estimation: I(d) by differencing until ADF p<0.05.
  d=0: series is I(0) already. No differencing required.
  d=1: one difference needed. Suggest Differencer(lags=1).
  d=2: need two differences. I would recommend Differencer(lag=2).

In my Econometrics course at IIT Madras, I learnt about power properties of ADF test, consequences of over-differencing (MA unit roots) and sensitivity of KPSS test to deterministic components. These edge cases – ambiguous test results, near integrated series, seasonal unit roots – are all handled by the WARNING severity level and the ambiguity fallback, ensuring the CVL never blocks a valid pipeline on a false positive.

3.2  Scitype Algebra
sktime's scientific type system defines a formal algebra of data types (Series, Panel and Tabular) with subtype relations. CVL encodes cross-component scitype contracts as a directed compatibility check over the component list of the pipeline. The rule is simple: the output scitype of component[i] must match the input scitype of component[i+1], as declared in the _tags of each estimator.

3.3  Exogenous Variable Shape Contract
The tensor shape contract for exogenous variables X is: shape must be (n_samples, n_features) where n_features >= 1. This is a 2D spec. CVL checks for the three most common violations (X with ndim = 1 (most common LLM error), X with ndim > 2, and X with n_samples != len(y)). Each violation has a specific, actionable fix hint associated with it.


04  12-Week Timeline: Conservative, Milestone-Driven


Week	Phase	Deliverables	Merged Output
1	Community Bonding	Finalize API contracts w/ mentors. Deep-dive into tools/. Catalog all known sktime exception message patterns for error_router.	Exception catalog fixture file (tests/fixtures/)
2	Community Bonding	Implement ConstraintType enum + Constraint dataclass. CI extended to lint constraints/ module. Discuss dry_run API with maintainers.	graph.py skeleton — PR open
3	ConstraintGraph	Implement build_constraint_graph() with tag-reader for STATIONARITY and MISSING_DATA types. Full unit tests.	graph.py — PR merged
4	ConstraintGraph	Extend tag-reader for SCITYPE_COMPAT and EXOGENOUS_SHAPE. Cross-component scitype constraint logic. Integration tests.	graph.py complete — PR merged
5	Validator (static)	Implement validate_static(). Performance test: must complete < 50ms on full sktime registry. All tests green.	validator.py (static) — PR merged
6	Validator (dynamic)	Implement validate_dynamic() with ADF+KPSS stationarity detection and _check_stationarity(). Synthetic I(0)/I(1)/I(2) test suite.	validator.py (dynamic) — PR merged
7	Buffer Week	Edge cases: ambiguous ADF/KPSS results, structural breaks, seasonal unit roots. Extended test coverage. Documentation of statistical methodology.	Extended test suite — PR merged
8	Error Router	Implement SemanticErrorRouter with full PATTERNS registry. Test against all catalogued exception messages. Coverage >= 90%.	error_router.py — PR merged
9	Prompt Engine	Implement CorrectionPromptEngine with all TEMPLATES. End-to-end test: invalid pipeline -> violation -> correction string verified correct.	prompt_engine.py — PR merged
10	Integration	Wire CVL into fit_predict_tool. Implement validate= and dry_run= flags. Full integration tests on real sktime demo datasets.	fit_predict.py — PR merged
11	Validation Testing	Replay 20 documented hallucination scenarios end-to-end. Benchmark: static validation < 50ms, dynamic < 500ms on 1000-obs series.	Benchmark report — PR merged
12	Wrap-up	Final docs: ConstraintGraph contribution guide. Tag-mapping reference. GSoC evaluation report. All PRs merged to main.	Documentation — all merged

Tiered Delivery Guarantee
Tier 1—Guaranteed Minimum (Weeks 1-7) ConstraintGraph + ConstraintValidator (statically and dynamically)
This alone catches the two most common failures: ARIMA-on-non-stationary and scitype mismatches.
This is mergeable in its own right and useful even if Tiers 2-3 fall behind.
 
Tier 2 – Expected Outcome (Weeks 8-11) CVL integrated into fit_predict, including dry_run flag.
SemanticErrorRouter + CorrectionPromptEngine close the self-healing loop.
 
Tier 3 — Stretch Goals (Week 12+): Seasonal unit root (HEGY) test. Constraint Graph
External estimator developers' contribution guide. Export as standalone sktime-constraints.



05  Risk Mitigation


Risk 1 — Stationarity Test Latency
ADF and KPSS tests are O(n log n). For large series (10,000+ observations), they can be 200 to 400 milliseconds. This only matters for the dynamic validation path.
• Mitigation A: Static validation (tag-based only) runs first. Is always < 50ms. Dynamic validation runs only when static validation passes and data is available.
• Mitigation B: Hash the data handle and cache the result. Repeated calls on the same dataset will skip the tests entirely.
• Mitigation C: validate=False bypasses both validation layers with zero overhead for latency-sensitive production use cases.

Risk 2 — Unstable Exception Messages Across sktime Versions
Sktime internal exception messages are not part of the public API and may change between minor versions. The regex patterns in SemanticErrorRouter can go stale.
• Mitigation A: PATTERNS uses fuzzy semantic regex (e.g., 'stationar' matches 'stationary,' 'non-stationary,' and 'non_stationary') rather than exact string matching.
• Mitigation B: The exception catalog (Week 1-2 deliverable) is version-tagged and included in the CI matrix—tested against sktime’s last 3 minor releases on every PR.
• Mitigation C: Unmatched exceptions go the fallback path: the raw exception is raised as-is. The router never eats errors it does not know of.

Risk 3 — False Positives Blocking Valid Pipelines
A bug in ConstraintValidator that blocks a valid pipeline in error would frustrate developers and erode confidence in the CVL.
•	Mitigation A: dry_run= True mode: returns a full audit report of what would have been blocked without blocking. Developers can file issues and inspect with complete diagnostic context.
•	Mitigation B: Ambiguous stationarity results (when ADF and KPSS disagree) are flagged as WARNING severity instead of ERROR. They are in the dry_run report but they never block execution.
•	Mitigation C: validate=False: the nuclear escape hatch. One parameter completely removes the CVL from the execution path with no refactoring required.



06  Proof of Readiness: Prior Contributions

Contribution	Relevance to This Proposal
sktime #9313 — Merged	Migrated skip config to native _tags system. The same _tags infrastructure that build_constraint_graph() reads to auto-populate constraints. Direct architectural continuity.
sktime #9286 — Merged	Nightly CI workflow for dependency testing. The CI extension required in Week 2 of this proposal (adding constraints/ to the test matrix) follows the identical pattern.
sktime #9456 — Open	NaiveForecaster._update() — navigates BaseForecaster's internal state architecture. Required background for the dynamic validation path that reads estimator internal state.
sktime-mcp param-key validation — Merged	The direct predecessor of ConstraintValidator. Validates parameter names; CVL validates mathematical compatibility. Same abstraction layer, next tier of strictness.
IIT Madras: Econometrics	ADF/KPSS mechanics, power of unit root tests, I(d) estimation, over-differencing consequences. Directly implemented in validate_dynamic() and _check_stationarity().
IIT Madras: Time Series Analysis	ARIMA identification, Box-Jenkins methodology, seasonal decomposition. Foundation for the stationarity and seasonality constraint types.



07  Closing Statement

The Constraint Validation Layer is the missing validation layer that makes sktime-mcp production safe. It is not a large system -- it is a small, precise, deeply considered module (~340 lines) that sits between the MCP tool interface and the sktime execution engine and prevents a well-defined, documented class of failures from ever reaching the user.
 
The tag-driven auto-scaling, caller-agnostic correction strings, opt-out escape hatches, and protocol-agnostic core were all designed to give the maintainers confidence that this module will not become a burden. It is intended to be community-maintained via sktime’s existing infrastructure, rather than by a single contributor.
 
I have the statistical background to implement the mathematical constraints correctly, the codebase familiarity to integrate without breaking existing behavior, and the track record of merged PRs to show I can navigate the full review cycle. I’m ready to start right away.



Mohit Kumar  |  IIT Madras  |  github.com/Mohit25f101  |  25f1001180@ds.study.iitm.ac.in
ESoC 2026 Batch 2  |  sktime-mcp  |  Constraint Validation Layer
