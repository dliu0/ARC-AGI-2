# Provider fallback: task-LM routing for benchmark repos

How a benchmark program reaches its pinned model when the provider serving it is
unreliable. Implemented in `langProBe/lm_provider.py` (+ `tests/test_lm_provider.py`)
in LangProBe-CodeEvolver as of 2026-09-02; this spec defines the behaviour every
other benchmark repo must reproduce and how to pick it up.

The problem it solves: the resellers serving these open-weight models fail
individually and often for billing reasons rather than outages. GMI answers a
single request with `402 {'error': 'Insufficient balance', 'reason':
'model_access_denied'}` and serves the next one normally; DeepSeek's first-party
API rejects everything the moment the account hits zero. LiteLLM retries
408/409/429/5xx and treats every other 4xx as permanent, so an unhandled provider
error fails the call, the row scores 0.0, and the eval aggregate is silently
depressed — the optimizer then attributes an infrastructure failure to the
program it is evolving.

## 1. The contract

A compliant benchmark repo satisfies all of these. They are the requirements;
the module below is one implementation of them.

- **C1 — One pinned model, several providers.** The benchmark pins a model. Each
  provider that serves it is a route to the *same weights*, so any of them may
  answer any call. Never fall back to a different model: that changes what is
  being measured.
- **C2 — Model ids live in one table.** One row per model, one column per
  provider, holding that provider's model string and route config. No model id
  appears anywhere else in the repo. Switching models is a one-row edit.
- **C3 — One preference order.** A single ranked list of providers decides both
  who serves a call and who covers for whom (a provider's default cover is the
  next one after it). No second hand-maintained "who covers whom" mapping.
- **C4 — Per-call divert on provider errors only.** A failed request is
  re-issued once on the cover with the *identical* model, messages, kwargs and
  reasoning settings. Divert any `openai.APIError` (all 4xx/5xx, `Timeout`,
  `APIConnectionError`) except the request-is-the-problem ones —
  `ContextWindowExceededError`, `UnsupportedParamsError` — which fail identically
  elsewhere and are the program's own fault. Never divert a non-API exception: a
  `ValueError` from the program must stay visible to whoever is evolving it.
- **C5 — A sustained outage must cost O(1), not O(calls).** Provider health is
  remembered process-wide. After N consecutive diverted failures the provider is
  skipped entirely — no request, no retries, no backoff — until a cooldown
  elapses and one probe call re-tests it. A stray error followed by a success
  must never trip this.
- **C6 — Retry budget matched to the cover.** With a cover available the primary
  gets one retry (two attempts) rather than LiteLLM's three: the third attempt's
  backoff is dead time when an equivalent provider is idle. With no cover, keep
  the full budget — retrying is then the only thing that can save the row.
- **C7 — Configurable by environment, defaulted in code.** Routing is selectable
  per run without editing code, and the unconfigured default is whatever is
  currently correct.
- **C8 — Outside the evolvable package.** The provider wiring is infrastructure,
  not part of the program under optimization. CodeEvolver must not be able to
  mutate it, and the model is a benchmark constraint. Keep the module out of the
  program package the optimizer edits, and keep provider/model ids out of the
  program's own source.

## 2. The module

Two files, self-contained apart from `dspy`, `litellm`, `openai`:

| file | role |
|---|---|
| `langProBe/lm_provider.py` | routing table, preference order, fallback LM, circuit breaker |
| `tests/test_lm_provider.py` | 95 tests, no network (the base `dspy.LM.forward` is monkeypatched) |

Public surface:

```python
build_task_lm(**overrides) -> dspy.LM   # the entrypoint every pipeline calls
ProviderFallbackLM                       # dspy.LM subclass hooking forward/aforward
MODEL_ROUTES, Route, TASK_MODEL, route_for(model, provider)
PROVIDER_PREFERENCE, PROVIDERS, DEFAULT_PROVIDER, DEFAULT_FALLBACK
ProviderBreaker, breaker_for(provider), reset_breakers()
should_fallback(exc), resolve_model/resolve_provider/resolve_fallback
```

The routing table (C2). Adding a provider is a column; adding a model is a row:

```python
MODEL_ROUTES = {
    "deepseek-v4-flash": {
        # GMI has no LiteLLM provider of its own: OpenAI-compatible route, and
        # its key must be passed explicitly or LiteLLM falls back to OPENAI_API_KEY.
        "gmi":       Route("openai/deepseek-ai/DeepSeek-V4-Flash",
                           api_base=GMI_API_BASE, api_key_env="GMI_API_KEY"),
        # First-party API: its own short ids, NOT the HuggingFace-style ones the
        # resellers use. Confirm against GET https://api.deepseek.com/models --
        # a wrong id silently benchmarks a different model.
        "deepseek":  Route("deepseek/deepseek-v4-flash"),
        "deepinfra": Route("deepinfra/deepseek-ai/DeepSeek-V4-Flash"),
    },
}
TASK_MODEL = "deepseek-v4-flash"
```

The preference order (C3), and the temporary pin that expresses "the preferred
provider is currently unusable":

```python
PROVIDER_PREFERENCE = ("gmi", "deepseek", "deepinfra")  # cheapest/fastest first
DEFAULT_PROVIDER = "deepseek"   # GMI's account is unfunded; restore to
                                # PROVIDER_PREFERENCE[0] when it is healthy
DEFAULT_FALLBACK = {p: the next provider after p, wrapping}
                  # gmi->deepseek, deepseek->deepinfra, deepinfra->gmi
```

Restoring the head of the order is the *only* edit needed when a provider comes
back: the covers follow the same list.

### Environment variables (C7)

| variable | default | effect |
|---|---|---|
| `LM_MODEL` | `TASK_MODEL` | which row of `MODEL_ROUTES` to serve |
| `LM_PROVIDER` | `DEFAULT_PROVIDER` | which provider serves the request first |
| `LM_FALLBACK` | on | `0/false/no/off/none` disarms the divert; a provider *name* picks the cover explicitly |
| `LM_BREAKER` | on | `0/false/...` disables the health flag (every call re-tries the primary) |
| `LM_BREAKER_FAILURES` | 3 | consecutive failures that mark a provider unhealthy |
| `LM_BREAKER_COOLDOWN` | 60 | seconds an unhealthy provider is skipped before one probe |

Unknown values fail loudly at construction rather than silently routing
somewhere unintended.

### Circuit breaker (C5)

`ProviderBreaker` is **process-global, one per provider**, not per LM and not
thread-local:

- Per-LM would fragment the state — `dspy.LM.copy()` is `copy.deepcopy`, and a
  `threading.Lock` is not deepcopy-able, so an attached breaker also breaks
  copying. The LM therefore stores the provider *name* and looks the breaker up
  per call.
- Thread-local would make each of 25 threads learn about the outage
  independently: 25 waves of failures instead of one.

State machine: **closed** → N consecutive diverted failures (any success resets
the count) → **open**, calls go straight to the cover at zero cost → after the
cooldown exactly one call is let through, **probing** → success closes it,
failure restarts the cooldown. Only diverted (C4) errors count: a
`ContextWindowExceededError` says nothing about provider health. Transitions are
logged once, not per call, and stamped on the active OTel span
(`lm.breaker.*`, `lm.fallback.*`, `lm.primary_skipped`).

## 3. Adoption

### Path A — a clone of LangProBe-CodeEvolver (hover, ragqa, …)

The module ships in the repo, so a clone picks it up by pulling:

```bash
git -C ../LangProBe-CodeEvolver-hover pull --ff-only origin main
git -C ../LangProBe-CodeEvolver-ragqa pull --ff-only origin main
```

Prerequisite: the change has to be on `origin/main` first. The primary working
dir has **no `origin` remote** (only a `codeevolver.origin-url` config key), so
it cannot push — commit and push from one of the clones, or add the remote in
the primary dir.

Nothing else to port. Confirm the pipeline in that clone still calls
`build_task_lm()` (§4) and that the provider keys are exported (§5). Each clone
runs its own optimization, so pull between jobs, not during one — a live job
checks iteration branches out in place.

### Path B — a benchmark repo that does not share this codebase

1. Copy `langProBe/lm_provider.py` and `tests/test_lm_provider.py` verbatim.
   Put the module in an infrastructure package, **never** inside the package
   CodeEvolver evolves (C8). Adjust the import path in the test file only.
2. Edit exactly two things for the new benchmark: the `MODEL_ROUTES` row for the
   model that benchmark pins, and `TASK_MODEL`. If the benchmark pins the same
   model, edit nothing.
3. Re-point every LM construction in the repo at `build_task_lm()` (§4),
   including eval harnesses and optimizer entrypoints.
4. Run the verification in §6.

Do not fork the logic per repo. If a repo needs different behaviour, express it
through the table, the preference order or the env vars; a divergent copy stops
being comparable across benchmarks.

## 4. Wiring a pipeline

The pipeline owns the LM and scopes it around its own execution:

```python
from langProBe.lm_provider import build_task_lm

class MyPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        # Provider wiring lives in langProBe/lm_provider.py; $LM_PROVIDER and
        # $LM_FALLBACK repoint the provider / name the cover / disarm the fallback.
        self.lm = build_task_lm()

    def forward(self, **kwargs):
        with dspy.context(lm=self.lm):
            return self.program(**kwargs)
```

Rules:

- Every LM the *task* uses comes from `build_task_lm()`. Overrides such as
  `cache=False` or `num_retries=...` pass through to both routes.
- A judge/metric LM is a different LM with a different purpose; if it also needs
  provider resilience, give it its own `MODEL_ROUTES` row rather than reusing
  the task row.
- Beware harness-level `dspy.configure(lm=...)`: in `simple_eval/evaluate_*.py`
  the `--lm` flag configures a global LM, but the pipeline's
  `dspy.context(lm=self.lm)` wins for the task. The flag does not govern which
  provider serves the benchmark — `$LM_PROVIDER` does.

## 5. Keys

| provider | variable | read when |
|---|---|---|
| GMI | `GMI_API_KEY` | **construction** — a GMI arm fails immediately without it |
| DeepInfra | `DEEPINFRA_API_KEY` | call time (LiteLLM) |
| DeepSeek | `DEEPSEEK_API_KEY` | call time (LiteLLM) |

Only the primary's key is needed to build the LM; the cover's key must still be
present or the first divert fails. Export all three for any real run. Keys read
at call time never land in `dump_state` or the trace files, which is why only
the `openai/`-route provider passes one explicitly.

## 6. Verification

```bash
# 1. Unit tests: no network, must be green before anything else.
python -m pytest tests/test_lm_provider.py -q          # 95 passed

# 2. Default route reaches a real provider, with reasoning still enabled.
python -c "
from langProBe.lm_provider import build_task_lm
lm = build_task_lm(cache=False)
print(lm.provider, lm.model, '->', lm.fallback_provider, lm._fallback.model)
print(lm('Reply with exactly: ok')[0])
print(lm.history[-1]['usage']['completion_tokens_details'].reasoning_tokens)  # > 0
"

# 3. Concurrency smoke at the real thread count (25 threads, 50 short calls).
#    Expect: 0 errors, breaker 'closed'.
```

Provider health checks before blaming the code — most failures here are billing:

```bash
curl -s -H "Authorization: Bearer $DEEPSEEK_API_KEY" https://api.deepseek.com/user/balance
#   {"is_available":true,"balance_infos":[{"total_balance":"49.87",...}]}
curl -s -H "Authorization: Bearer $DEEPSEEK_API_KEY" https://api.deepseek.com/models
```

## 7. Operating it

- **Normal run:** set nothing. The defaults are the currently correct routing.
- **Measuring or diagnosing one provider:** `LM_PROVIDER=<p> LM_FALLBACK=0`.
  Mandatory for any latency/throughput comparison — with the divert armed a
  "DeepInfra" arm silently contains another provider's calls and the numbers
  describe neither.
- **A provider is down for the day:** nothing is required (the breaker routes
  around it after one wave), but `LM_PROVIDER=<healthy>` avoids paying that wave
  on every process start.
- **A provider comes back:** set `DEFAULT_PROVIDER = PROVIDER_PREFERENCE[0]`.
- **Reproducing pre-breaker behaviour:** `LM_BREAKER=0`.

## 8. Why it is built this way

Measured on hover, 200 rows (seed 0), 25 threads, each arm pinned with
`LM_FALLBACK=0`, on 2026-09-01:

| | GMI | DeepInfra | DeepSeek 1st-party |
|---|---|---|---|
| wall (200 rows) | 318s | 594s | 491s |
| median LM call | 7.1s | 12.1s | 5.6s |
| mean score | 0.395 | 0.430 | 0.440 |

Scores are statistically indistinguishable (paired McNemar, p = 0.37 / 0.23 /
0.89), so **provider choice is a throughput and cost decision, not a quality
one** — which is what makes diverting between them sound, and why GMI (cheapest,
fastest) leads the order.

Outage cost, simulated at the real shape (800 calls, 25 threads, primary hard
down, 0.4s per failed attempt): **25 wasted primary attempts in total** — the
threads already in flight — then zero for the remaining 775 calls. Overhead is
one wave, roughly `(1 + PRIMARY_NUM_RETRIES) × fail_latency`, plus ~2 probe
attempts per minute of eval. Without the breaker every one of the 800 calls pays
that.

The unbounded case, stated honestly: a provider that accepts connections and
**hangs**. There is no request timeout, and a tight one is not safe — legitimate
calls on this model reach 189s (reasoning-length variance, not throttling). Only
a cap well above that (~240s) would bound it, and it would bound the first wave
only.

## 9. Changing the model or adding a provider

- **New model:** add a row to `MODEL_ROUTES` with one entry per provider that
  serves it, then point `TASK_MODEL` (or `$LM_MODEL`) at it. Confirm each
  provider's exact id against that provider's own model listing; ids differ in
  form between resellers and first-party APIs, and a wrong one benchmarks
  different weights without erroring.
- **New provider:** add it to every row that it serves, and place it in
  `PROVIDER_PREFERENCE` by cost/throughput. Covers re-derive themselves. A
  provider missing from a row fails loudly at construction, naming the row.
- **Never** add a model id, endpoint or key lookup anywhere else in the repo.
