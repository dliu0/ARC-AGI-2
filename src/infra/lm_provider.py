"""Task LM for the ARC-AGI-2 benchmark program: one pinned model, served by a
ranked list of providers, with a per-call cross-provider fallback, a shared
health flag and a streaming hang guard.

Why this exists
---------------
The resellers serving this model are individually unreliable. GMI intermittently
answers a single request with a 4xx -- observed as ``Error code: 402 - {'error':
'Insufficient balance', 'reason': 'model_access_denied'}`` -- and then serves the
very next request normally. LiteLLM retries 408/409/429/5xx but treats every
other 4xx as permanent, so the call fails, the row scores 0.0 and the eval
aggregate is silently depressed -- and the optimizer then attributes an
infrastructure failure to the program it is evolving.

``ProviderFallbackLM`` keeps one provider as the primary and, when its request
fails with a provider/API error, re-issues the SAME request -- same model, same
messages, same kwargs, same ``reasoning_effort`` -- on the next provider down
the preference order, which serves the identical model. Every diverted error is
logged to stderr with a ``[WARNING]`` tag, and the fallback outcome is stamped
on the current OpenTelemetry span so it is visible in the trace files.

Relationship to docs/provider_fallback.md
-----------------------------------------
That document specifies the behaviour (C1-C8) and describes a DSPy
implementation, because the repo it was written in (LangProBe-CodeEvolver)
is a DSPy codebase. ARC-AGI-2 is not: the solver is a direct LiteLLM call, and
pulling DSPy in purely to reach its ``dspy.LM`` request hook would add a large
dependency to the benchmark image, change the client stack relative to the
arc-agi (v1) arm, and -- decisively -- break the streaming hang guard below,
which only works because this module owns the request.

So the CONTRACT is implemented here, not the module: same routing table, same
single preference order, same divert rule, same process-global breaker, same
retry budgets, same environment variables, same public names. What changed is
that requests go through ``litellm.completion`` directly rather than through
``dspy.LM.forward``. Keep it that way -- if the contract needs to change, change
it in both repos so the benchmarks stay comparable.

Where a model id lives
----------------------
In ``MODEL_ROUTES`` and nowhere else: one row per model, one column per provider,
holding that provider's LiteLLM model string plus whatever else its route needs
(``api_base``, an explicit key). The three providers name the same weights
differently -- ``openai/deepseek-ai/DeepSeek-V4-Flash`` on GMI,
``deepinfra/deepseek-ai/DeepSeek-V4-Flash`` on DeepInfra, ``deepseek/deepseek-
v4-flash`` on DeepSeek's own API -- so switching models means editing ONE row,
not hunting three constants. ``TASK_MODEL`` (or ``$LM_MODEL``) picks the row.

Which provider serves a call
----------------------------
``PROVIDER_PREFERENCE`` ranks the providers -- GMI first (cheapest, and the
fastest of the three on the hover benchmark), then DeepSeek's own first-party
API, then DeepInfra -- and that single ordering decides both who serves a call
and who covers for whom: a provider's default fallback is the next one after it.
``DEFAULT_PROVIDER`` pins the primary one step down the list while GMI's API is
erroring, so an unconfigured run is DeepSeek primary with DeepInfra covering.

Cost of an outage (the circuit breaker)
---------------------------------------
Diverting per call is right for a stray error and wrong for a dead provider: at
50 threads every one of ~800 calls would pay the primary's failed attempts
before reaching the cover. So provider health is remembered process-wide.
``ProviderBreaker`` counts CONSECUTIVE diverted failures (any success resets the
count): ``BREAKER_FAILURES`` of them mark the provider unhealthy, and calls then
go STRAIGHT to the cover -- no attempt, no retries, no wait -- for
``BREAKER_COOLDOWN_SECONDS``, after which exactly one call is let through as a
probe whose outcome either restores the provider or restarts the cooldown.

The hang guard
--------------
An error is not the worst failure mode; silence is. GMI occasionally accepts a
request and then sends nothing until its gateway kills the connection at ~20
minutes -- measured on arc-agi at ~4.7% of rows, enough that nearly EVERY
parallel eval walled at the straggler. A hang raises nothing, so neither the
breaker nor the divert would ever see it.

Streaming makes it detectable. These providers stream reasoning deltas
continuously (measured max inter-chunk gap ~3s), so ``READ_GAP_TIMEOUT_S`` of
total silence is an unambiguous hang. httpx applies ``timeout`` per READ on a
stream rather than to the whole response, so a long generation is unaffected --
which matters here, because legitimate ARC rows at ``reasoning_effort=high``
run for many minutes. That is also why a plain total-request timeout is not a
substitute: any cap tight enough to catch a hang quickly would kill real rows.

A hang is retried ``MAX_ATTEMPTS`` times on the same route and then propagates
as a ``litellm.Timeout``, which ``should_fallback`` diverts like any other
provider error -- so a hung primary ends up served by the cover and counted
against the primary's health. ``TOTAL_BUDGET_S`` is a deadline on the whole
call, primary and cover together; exceeding it raises ``BudgetExceeded``, which
is deliberately NOT an ``openai.APIError``: the budget is gone, so there is
nowhere useful to divert to.

Environment variables
---------------------
``LM_MODEL``          which row of ``MODEL_ROUTES`` to serve (default
                      ``TASK_MODEL``).
``LM_PROVIDER``       ``gmi``, ``deepseek`` or ``deepinfra`` -- which provider
                      serves the request first. Defaults to ``DEFAULT_PROVIDER``
                      (``deepseek`` while GMI is down). Set it to route a whole
                      run through one provider (e.g. an A/B of provider latency).
``LM_FALLBACK``       ``1``/``true``/``yes``/``on`` (default), ``0``/``false``/
                      ``no``/``off``/``none``, or the NAME of a provider. Whether
                      a provider error is re-issued elsewhere, and on whom: a
                      bare "on" uses the provider that follows the primary in
                      ``PROVIDER_PREFERENCE``. Turn it OFF when the point of the
                      run is to measure or diagnose one provider on its own --
                      with it on, a slow or failing primary is silently papered
                      over by the secondary and the numbers describe neither.
``LM_BREAKER``        ``0``/``false``/``no``/``off``/``none`` disables the health
                      flag, so every call re-tries the primary (the pre-breaker
                      behaviour). On by default, and inert without a cover.
``LM_BREAKER_FAILURES``   consecutive failures that mark a provider unhealthy
                      (default ``BREAKER_FAILURES``).
``LM_BREAKER_COOLDOWN``   seconds an unhealthy provider is skipped before one
                      probe call (default ``BREAKER_COOLDOWN_SECONDS``).

Unknown values fail loudly at construction rather than silently routing
somewhere unintended.

What is diverted (see ``should_fallback``)
------------------------------------------
Everything caught here comes from the LM request path -- never from program
code -- so the question is only whether a second provider could plausibly
answer the same request. It is diverted when the error is an API error
(``openai.APIError`` is the base of every LiteLLM exception): any 4xx
(401/402/403/404/422/429, generic 400), any 5xx, ``Timeout`` (including a
detected hang) and ``APIConnectionError`` -- all AFTER LiteLLM's own retries
for the transient ones have run out. Only these errors count toward the breaker.

It is NOT diverted when the request itself is the problem and the same model
elsewhere would fail identically -- ``ContextWindowExceededError`` (the prompt
does not fit) and ``UnsupportedParamsError`` (an invalid request parameter) --
nor when the row's total budget is gone (``BudgetExceeded``). Non-API
exceptions (a ``ValueError`` from the program) are never diverted either: they
must stay visible to whoever is evolving the program.

This module is deliberately outside ``src/program/``, the package CodeEvolver
evolves: the benchmark constraints pin the task model, and the provider wiring
is infrastructure, not part of the program under optimization.

Keys: GMI's is passed explicitly (otherwise LiteLLM's ``openai/`` route would
fall back to ``OPENAI_API_KEY``); DeepInfra's and DeepSeek's are read by LiteLLM
from ``DEEPINFRA_API_KEY`` / ``DEEPSEEK_API_KEY`` at call time so they never land
in the trace files. Only the primary's and the cover's keys are needed.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

import litellm
import openai

try:  # httpx is a LiteLLM dependency; guarded so the module imports without it
    import httpx as _httpx
except Exception:  # pragma: no cover - httpx is always present with litellm
    _httpx = None

# ---------------------------------------------------------------------------
# Model routing table
# ---------------------------------------------------------------------------

# GMI Cloud has no LiteLLM provider of its own; it is reached through the
# OpenAI-compatible route (model="openai/<id>" + api_base=<GMI endpoint>).
GMI_API_BASE = "https://api.gmi-serving.com/v1"


@dataclass(frozen=True)
class Route:
    """How ONE provider serves ONE model.

    ``model`` is the LiteLLM model string. ``api_base`` and ``api_key_env`` are
    only needed by providers reached through the generic ``openai/`` route --
    LiteLLM's native providers (``deepinfra/``, ``deepseek/``) know their own
    endpoint and read their own key from the environment at call time, which
    also keeps the key out of the trace files.
    """

    model: str
    api_base: str | None = None
    api_key_env: str | None = None

    def request_kwargs(self) -> dict[str, Any]:
        """``litellm.completion`` kwargs pinning a request to this route."""
        kwargs: dict[str, Any] = {"model": self.model}
        if self.api_base is not None:
            kwargs["api_base"] = self.api_base
        if self.api_key_env is not None:
            # Explicit: LiteLLM's openai/ route would otherwise fall back to
            # OPENAI_API_KEY. KeyError here is the right failure -- the run
            # cannot use this provider without its key.
            kwargs["api_key"] = os.environ[self.api_key_env]
        return kwargs


# One row per model, one column per provider. This is the ONLY place a model id
# lives: to serve a different model, add a row here and point TASK_MODEL (or
# $LM_MODEL) at it. A provider missing from a row cannot serve that model and
# says so at construction rather than at the first call.
MODEL_ROUTES: dict[str, dict[str, Route]] = {
    "deepseek-v4-flash": {
        "gmi": Route(
            "openai/deepseek-ai/DeepSeek-V4-Flash",
            api_base=GMI_API_BASE,
            api_key_env="GMI_API_KEY",
        ),
        # DeepSeek's first-party API uses its own short ids rather than the
        # HuggingFace-style ``deepseek-ai/<name>`` the resellers use; picking
        # the wrong one silently benchmarks a DIFFERENT model. This one is
        # listed by GET https://api.deepseek.com/models and is what the 3-way
        # provider benchmark ran on.
        "deepseek": Route("deepseek/deepseek-v4-flash"),
        "deepinfra": Route("deepinfra/deepseek-ai/DeepSeek-V4-Flash"),
    },
}

# The model the benchmark pins; $LM_MODEL selects another row of MODEL_ROUTES.
TASK_MODEL = "deepseek-v4-flash"
MODEL_ENV_VAR = "LM_MODEL"

# Reasoning is enabled via the standard OpenAI `reasoning_effort` param. None of
# the routes allow it by default, so it must be forwarded via
# allowed_openai_params=[...] (BerriAI/litellm#14039). Do NOT use
# thinking={"type": "enabled"}: DeepInfra's endpoint rejects the kwarg.
REASONING_EFFORT = "high"
_ALLOWED_OPENAI_PARAMS = ["reasoning_effort"]

# ---------------------------------------------------------------------------
# Provider preference
# ---------------------------------------------------------------------------

# Most preferred first. GMI leads: it is the cheapest per token and was the
# fastest of the three on the 200-row hover benchmark, where all three scored
# equivalently. DeepSeek's first-party API is next, DeepInfra last (uniformly
# ~1.7x slower than GMI).
PROVIDER_PREFERENCE = ("gmi", "deepseek", "deepinfra")
PROVIDERS = PROVIDER_PREFERENCE

# The primary when $LM_PROVIDER is unset. Normally ``PROVIDER_PREFERENCE[0]``,
# but GMI's API is erroring on every call as of 2026-09-02, so runs start one
# step down the order. Set this back to ``PROVIDER_PREFERENCE[0]`` once GMI is
# healthy -- that alone restores GMI primary with DeepSeek covering, because the
# fallbacks below follow the same ordering.
DEFAULT_PROVIDER = "deepseek"

PROVIDER_ENV_VAR = "LM_PROVIDER"
FALLBACK_ENV_VAR = "LM_FALLBACK"
# Who covers for whom when LM_FALLBACK is on but does not name a provider: the
# next provider down the preference order, wrapping at the end. So GMI diverts
# to DeepSeek, DeepSeek (today's primary) to DeepInfra.
DEFAULT_FALLBACK = {
    provider: PROVIDER_PREFERENCE[(i + 1) % len(PROVIDER_PREFERENCE)]
    for i, provider in enumerate(PROVIDER_PREFERENCE)
}

# LiteLLM's own retry budget, which covers the transient statuses (408/409/429/
# 5xx) INSIDE one attempt. With a cover available the primary gets one retry,
# i.e. two attempts: LiteLLM's third attempt is dead time when an equivalent
# provider is sitting idle. Without a cover the full budget stands -- there is
# nowhere else to go, so retrying is all that can save the row.
PRIMARY_NUM_RETRIES = 1
COVER_NUM_RETRIES = 3

# ---------------------------------------------------------------------------
# Hang guard
# ---------------------------------------------------------------------------

# Seconds of total stream silence that count as a hang. Applied by httpx PER
# READ, so it does not cap a long generation -- only a dead connection.
READ_GAP_TIMEOUT_S = 240.0
# Deadline for one completion() call across every attempt and both providers.
TOTAL_BUDGET_S = 2400.0
# Attempts per route before a hang is propagated (and therefore diverted).
MAX_ATTEMPTS = 2

# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

BREAKER_ENV_VAR = "LM_BREAKER"
BREAKER_FAILURES_ENV_VAR = "LM_BREAKER_FAILURES"
BREAKER_COOLDOWN_ENV_VAR = "LM_BREAKER_COOLDOWN"
# Three CONSECUTIVE diverted failures. A stray 402 is followed by a success,
# which resets the count, so strays never trip it; an outage trips it inside the
# first wave of concurrent calls.
BREAKER_FAILURES = 3
BREAKER_COOLDOWN_SECONDS = 60.0

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off", "none", ""}

_LOG_ERROR_CHARS = 300


class BudgetExceeded(TimeoutError):
    """The row's ``TOTAL_BUDGET_S`` ran out.

    Deliberately NOT an ``openai.APIError``: the budget is spent, so diverting
    to the cover would only spend more of a budget that is already gone. It
    reaches the program as a plain exception, which is the signal to give up on
    this row rather than to try somewhere else.
    """


# Request-is-the-problem errors: the identical request on the identical model
# fails the same way on the other provider, so diverting only hides a program fault.
NON_FALLBACK_ERRORS: tuple[type[BaseException], ...] = (
    litellm.ContextWindowExceededError,
    litellm.UnsupportedParamsError,
)

# Silence on an open connection. LiteLLM raises its own ``Timeout`` (a subclass
# of ``openai.APITimeoutError``) for most of these, but a read timeout surfacing
# while the stream is being ITERATED can come straight from httpx, outside
# LiteLLM's translation layer -- hence both.
_HANG_ERRORS: tuple[type[BaseException], ...] = (
    litellm.Timeout,
    openai.APITimeoutError,
) + ((_httpx.TimeoutException,) if _httpx is not None else ())


def _status_code(exc: BaseException) -> int | None:
    """HTTP status carried by a LiteLLM/OpenAI exception, or None."""
    code = getattr(exc, "status_code", None)
    if isinstance(code, bool) or not isinstance(code, int):
        return None
    return code


def should_fallback(exc: BaseException) -> bool:
    """True for a provider/API error a second provider could plausibly answer.

    ``openai.APIError`` is the common base of every LiteLLM exception
    (4xx, 5xx, timeouts, connection errors); anything else is not an API
    error and is left alone. ``NON_FALLBACK_ERRORS`` carves out the
    request-is-the-problem cases, and ``BudgetExceeded`` is excluded by not
    being an API error at all.
    """
    if not isinstance(exc, openai.APIError):
        return False
    return not isinstance(exc, NON_FALLBACK_ERRORS)


def is_hang(exc: BaseException) -> bool:
    """True when the connection went silent, as opposed to answering an error.

    A hang is worth one more attempt on the same route (the connection, not the
    provider, may be the problem). ``BudgetExceeded`` is not a hang: the row is
    out of time, and retrying anything cannot help.
    """
    if isinstance(exc, BudgetExceeded):
        return False
    return isinstance(exc, _HANG_ERRORS)


def _env_flag(name: str, flag: bool | None = None, default: bool = True) -> bool:
    """A boolean env var: explicit ``flag`` wins, then ``$name``, then ``default``."""
    if flag is not None:
        return bool(flag)
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    raise ValueError(
        f"{name}={value!r} is not a boolean; expected one of {sorted(_TRUTHY)} "
        f"or {sorted(v for v in _FALSY if v)}"
    )


def resolve_model(name: str | None = None) -> str:
    """The model row: explicit ``name``, else ``$LM_MODEL``, else ``TASK_MODEL``."""
    value = (name if name is not None else os.environ.get(MODEL_ENV_VAR, "")).strip()
    value = value or TASK_MODEL
    if value not in MODEL_ROUTES:
        raise ValueError(
            f"{value!r} is not in MODEL_ROUTES; known models: {list(MODEL_ROUTES)}. "
            f"Add a row (one entry per provider that serves it) to route it."
        )
    return value


def resolve_provider(name: str | None = None) -> str:
    """The primary: explicit ``name``, else ``$LM_PROVIDER``, else DEFAULT_PROVIDER."""
    value = (name if name is not None else os.environ.get(PROVIDER_ENV_VAR, "")).strip().lower()
    if not value:
        return DEFAULT_PROVIDER
    if value not in PROVIDERS:
        raise ValueError(
            f"{PROVIDER_ENV_VAR}={value!r} is not a known provider; expected one of {list(PROVIDERS)}"
        )
    return value


def resolve_fallback(primary: str, flag: bool | str | None = None) -> str | None:
    """The provider that covers for ``primary``, or None if diversion is off.

    ``$LM_FALLBACK`` (or ``flag``) is either a boolean -- on meaning
    ``DEFAULT_FALLBACK[primary]`` -- or the name of a provider to use instead.
    """
    if flag is None:
        flag = os.environ.get(FALLBACK_ENV_VAR, "").strip().lower()
    if flag is True:
        return DEFAULT_FALLBACK[primary]
    if flag is False:
        return None
    value = str(flag).strip().lower()
    if not value or value in _TRUTHY:
        return DEFAULT_FALLBACK[primary]
    if value in _FALSY:
        return None
    if value in PROVIDERS:
        if value == primary:
            raise ValueError(
                f"{FALLBACK_ENV_VAR}={value!r} names the primary provider; a provider "
                f"cannot be its own fallback"
            )
        return value
    raise ValueError(
        f"{FALLBACK_ENV_VAR}={value!r} is neither a boolean nor a provider; expected one of "
        f"{sorted(_TRUTHY)}, {sorted(v for v in _FALSY if v)} or {list(PROVIDERS)}"
    )


def route_for(model: str, provider: str) -> Route:
    """The ``Route`` serving ``model`` on ``provider``, or a loud failure."""
    routes = MODEL_ROUTES[model]
    try:
        return routes[provider]
    except KeyError:
        raise ValueError(
            f"provider {provider!r} has no route for model {model!r}; MODEL_ROUTES[{model!r}] "
            f"serves {sorted(routes)}. Add the provider's model string to that row, or route "
            f"the run elsewhere with {PROVIDER_ENV_VAR}/{FALLBACK_ENV_VAR}."
        ) from None


def _describe(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    if len(text) > _LOG_ERROR_CHARS:
        text = text[:_LOG_ERROR_CHARS] + "..."
    return f"{type(exc).__name__}: {text}"


def _warn(msg: str) -> None:
    print(f"[WARNING] {msg}", file=sys.stderr, flush=True)


def _error(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr, flush=True)


def _mark_span(**attrs: Any) -> None:
    """Best-effort: stamp the fallback on the active OTel span (never raises)."""
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        for key, value in attrs.items():
            span.set_attribute(key, value)
    except Exception:  # noqa: BLE001 -- tracing must never break a call
        pass


class ProviderBreaker:
    """Process-wide health flag for one provider, shared by every thread.

    CLOSED -- calls go to the provider. ``threshold`` consecutive diverted
    failures (any success resets the count) open it.

    OPEN -- ``allow()`` returns False and callers skip the provider entirely:
    no attempt, no retries, no backoff. Costs nothing per call, which is the
    whole point at 50 threads.

    PROBING -- after ``cooldown`` seconds exactly one call is let through; its
    outcome closes the breaker or restarts the cooldown. Only one probe is in
    flight at a time, so a still-dead provider costs one call per cooldown.

    State is three numbers mutated under a lock that is never held across an
    I/O call. Instances are process-global (see ``breaker_for``) rather than
    per-LM: provider health is a property of the provider, and a thread-local
    breaker would make each of 50 threads learn about the outage independently.
    """

    def __init__(
        self,
        provider: str,
        threshold: int = BREAKER_FAILURES,
        cooldown: float = BREAKER_COOLDOWN_SECONDS,
        clock=time.monotonic,
    ):
        self.provider = provider
        self.threshold = threshold
        self.cooldown = cooldown
        self._clock = clock
        self._lock = threading.Lock()
        self._failures = 0
        self._opened_at: float | None = None  # None => closed
        self._probing = False
        self._skipped = 0

    @property
    def state(self) -> str:
        with self._lock:
            if self._opened_at is None:
                return "closed"
            return "probing" if self._probing else "open"

    def allow(self) -> bool:
        """False when the provider is known-unhealthy and should be skipped."""
        with self._lock:
            if self._opened_at is None:
                return True
            if self._probing or self._clock() - self._opened_at < self.cooldown:
                self._skipped += 1
                return False
            self._probing = True
            waited = self._clock() - self._opened_at
        _warn(
            f"{self.provider} has been skipped for {waited:.0f}s "
            f"({self._skipped} calls served by the fallback) -- probing it with one call"
        )
        return True

    def record_success(self) -> None:
        with self._lock:
            recovered = self._opened_at is not None
            skipped = self._skipped
            self._failures = 0
            self._opened_at = None
            self._probing = False
            self._skipped = 0
        if recovered:
            _warn(
                f"{self.provider} answered the probe -- marking it healthy again "
                f"({skipped} calls went to the fallback while it was down)"
            )
            _mark_span(**{"lm.breaker.provider": self.provider, "lm.breaker.state": "closed"})

    def record_failure(self) -> None:
        """Record a diverted provider error; may open (or re-open) the breaker."""
        with self._lock:
            was_probe = self._probing
            self._probing = False
            self._failures += 1
            failures = self._failures
            opened = False
            if self._opened_at is not None:
                self._opened_at = self._clock()  # probe failed: restart the cooldown
            elif failures >= self.threshold:
                self._opened_at = self._clock()
                opened = True
        if opened:
            _warn(
                f"{self.provider} failed {failures} calls in a row -- marking it unhealthy "
                f"and sending every call straight to the fallback for {self.cooldown:.0f}s"
            )
            _mark_span(**{"lm.breaker.provider": self.provider, "lm.breaker.state": "open"})
        elif was_probe:
            _warn(
                f"{self.provider} failed the probe -- still unhealthy, "
                f"skipping it for another {self.cooldown:.0f}s"
            )


_BREAKERS: dict[str, ProviderBreaker] = {}
_BREAKERS_LOCK = threading.Lock()


def breaker_for(provider: str) -> ProviderBreaker:
    """The shared breaker for ``provider`` -- one per process, not per LM."""
    breaker = _BREAKERS.get(provider)
    if breaker is None:
        with _BREAKERS_LOCK:
            breaker = _BREAKERS.setdefault(
                provider,
                ProviderBreaker(
                    provider,
                    threshold=int(
                        os.environ.get(BREAKER_FAILURES_ENV_VAR, "").strip() or BREAKER_FAILURES
                    ),
                    cooldown=float(
                        os.environ.get(BREAKER_COOLDOWN_ENV_VAR, "").strip()
                        or BREAKER_COOLDOWN_SECONDS
                    ),
                ),
            )
    return breaker


def reset_breakers() -> None:
    """Forget every provider's recorded health (tests; a fresh run in-process)."""
    with _BREAKERS_LOCK:
        _BREAKERS.clear()


class ProviderFallbackLM:
    """``TASK_MODEL`` on ``provider``; a provider error re-issues the call elsewhere.

    ``completion(messages)`` is the whole surface: it streams the reply, watches
    for a hang, and returns the assistant text. ``model`` / ``provider`` /
    ``fallback`` / ``breaker`` default to ``$LM_MODEL``, ``$LM_PROVIDER``,
    ``$LM_FALLBACK`` and ``$LM_BREAKER`` -- today DeepSeek primary, DeepInfra
    covering, health flag armed. With ``fallback=False`` no cover route is
    built, the breaker is inert and errors propagate.

    Every other keyword (``temperature=...``, ``max_tokens=...``) is applied to
    both routes, so the two providers always receive the identical request.
    """

    def __init__(
        self,
        provider: str | None = None,
        fallback: bool | str | None = None,
        model: str | None = None,
        breaker: bool | None = None,
        read_gap_timeout: float = READ_GAP_TIMEOUT_S,
        total_budget: float = TOTAL_BUDGET_S,
        max_attempts: int = MAX_ATTEMPTS,
        **overrides: Any,
    ):
        self.task_model = resolve_model(model)
        self.provider = resolve_provider(provider)
        self.fallback_provider: str | None = resolve_fallback(self.provider, fallback)
        self.read_gap_timeout = float(read_gap_timeout)
        self.total_budget = float(total_budget)
        self.max_attempts = int(max_attempts)

        # An explicit num_retries applies to both routes; otherwise the primary
        # gets the short budget only because there is somewhere else to go.
        retries = overrides.pop("num_retries", None)
        cover = self.fallback_provider
        self._request: dict[str, dict[str, Any]] = {
            self.provider: self._route_kwargs(
                self.provider,
                retries
                if retries is not None
                else (PRIMARY_NUM_RETRIES if cover is not None else COVER_NUM_RETRIES),
                overrides,
            )
        }
        if cover is not None:
            self._request[cover] = self._route_kwargs(
                cover, retries if retries is not None else COVER_NUM_RETRIES, overrides
            )

        # Armed only when there is somewhere to divert to: skipping the primary
        # is meaningless without a cover.
        self.breaker_enabled = cover is not None and _env_flag(BREAKER_ENV_VAR, breaker)

    def _route_kwargs(
        self, provider: str, num_retries: int, overrides: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            **route_for(self.task_model, provider).request_kwargs(),
            "reasoning_effort": REASONING_EFFORT,
            "allowed_openai_params": list(_ALLOWED_OPENAI_PARAMS),
            "num_retries": num_retries,
            **overrides,
        }

    @property
    def model(self) -> str:
        """The primary route's LiteLLM model string."""
        return self._request[self.provider]["model"]

    @property
    def fallback_model(self) -> str | None:
        """The cover route's LiteLLM model string, or None when uncovered."""
        if self.fallback_provider is None:
            return None
        return self._request[self.fallback_provider]["model"]

    @property
    def breaker(self) -> ProviderBreaker | None:
        """The primary's shared health flag, or None when it is not armed."""
        return breaker_for(self.provider) if self.breaker_enabled else None

    # -- the request --------------------------------------------------------

    def completion(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """Assistant text for ``messages``, from whichever provider answers.

        Raises whatever the last provider raised -- an ``openai.APIError`` when
        both routes failed, ``BudgetExceeded`` when the row ran out of time, or
        the original exception untouched when it was never a provider problem.
        """
        deadline = time.monotonic() + self.total_budget
        breaker = self.breaker

        if breaker is not None and not breaker.allow():
            self._mark_skipped()
            return self._attempt(self.fallback_provider, messages, deadline, kwargs)

        try:
            result = self._attempt(self.provider, messages, deadline, kwargs)
        except Exception as exc:  # noqa: BLE001 -- classified below
            if self.fallback_provider is None or not should_fallback(exc):
                raise
            if breaker is not None:
                breaker.record_failure()
            code = _status_code(exc)
            started = self._log_diverted(code, exc)
            try:
                result = self._attempt(self.fallback_provider, messages, deadline, kwargs)
            except Exception as exc2:  # noqa: BLE001
                self._log_fallback_failed(code, exc2, started)
                raise exc2 from exc
            self._log_fallback_ok(code, started)
            return result
        if breaker is not None:
            breaker.record_success()
        return result

    def _attempt(
        self,
        provider: str,
        messages: list[dict[str, Any]],
        deadline: float,
        kwargs: dict[str, Any],
    ) -> str:
        """Stream one request on ``provider``, retrying only a HANG.

        An API error is NOT retried here: LiteLLM's own ``num_retries`` has
        already covered the transient statuses, so a second identical attempt
        on the same provider is wasted time when a cover is standing by. It
        propagates, and ``completion`` decides whether to divert.
        """
        request = {**self._request[provider], **kwargs, "messages": messages}
        last_hang: BaseException | None = None

        for attempt in range(self.max_attempts):
            remaining = deadline - time.monotonic()
            if remaining <= 0 or (last_hang is not None and remaining <= self.read_gap_timeout):
                # No time for another full read-gap window: stop paying for one.
                break
            try:
                return self._stream(request, deadline)
            except Exception as exc:  # noqa: BLE001 -- classified below
                if not is_hang(exc):
                    raise
                last_hang = exc
                _warn(
                    f"{provider} sent nothing for {self.read_gap_timeout:.0f}s on "
                    f"{request['model']} (attempt {attempt + 1}/{self.max_attempts}) "
                    f"-- treating it as a hang"
                )
                _mark_span(**{"lm.hang.provider": provider, "lm.hang.attempts": attempt + 1})

        if last_hang is not None:
            raise last_hang
        raise BudgetExceeded(
            f"row exceeded its total budget of {self.total_budget:.0f}s before "
            f"{provider} could be attempted"
        )

    def _stream(self, request: dict[str, Any], deadline: float) -> str:
        """One streaming request. Silence longer than the read gap raises.

        ``timeout`` is applied by httpx PER READ on a stream, not to the whole
        response, so a long generation is unaffected -- only a dead connection
        trips it. ``deadline`` additionally caps the row across all attempts.
        """
        stream = litellm.completion(
            **request, stream=True, timeout=self.read_gap_timeout
        )
        parts: list[str] = []
        for chunk in stream:
            if time.monotonic() > deadline:
                raise BudgetExceeded(
                    f"row exceeded its total budget of {self.total_budget:.0f}s mid-stream"
                )
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            content = getattr(delta, "content", None) if delta is not None else None
            if content:
                parts.append(content)
        return "".join(parts)

    # -- logging ------------------------------------------------------------

    def _mark_skipped(self) -> None:
        """Stamp the trace: this call never touched the unhealthy primary."""
        _mark_span(**{"lm.primary_skipped": True, "lm.primary": self.provider})

    def _log_diverted(self, code: int | None, exc: BaseException) -> float:
        desc = _describe(exc)
        _warn(
            f"{self.provider} error on {self.model} (status={code}, {desc}) "
            f"-- retrying this call on {self.fallback_provider} ({self.fallback_model})"
        )
        _mark_span(
            **{
                "lm.fallback.provider": self.fallback_provider,
                "lm.fallback.model": self.fallback_model,
                "lm.fallback.primary_status": code if code is not None else -1,
                "lm.fallback.primary_error": desc,
            }
        )
        return time.monotonic()

    def _log_fallback_ok(self, code: int | None, started: float) -> None:
        elapsed = time.monotonic() - started
        _warn(
            f"{self.fallback_provider} fallback succeeded in {elapsed:.1f}s "
            f"(primary_status={code})"
        )
        _mark_span(**{"lm.fallback.outcome": "ok", "lm.fallback.seconds": elapsed})

    def _log_fallback_failed(
        self, code: int | None, exc: BaseException, started: float
    ) -> None:
        elapsed = time.monotonic() - started
        desc = _describe(exc)
        _error(
            f"{self.fallback_provider} fallback failed after {elapsed:.1f}s "
            f"(primary_status={code}, fallback_status={_status_code(exc)}): {desc}"
        )
        _mark_span(
            **{
                "lm.fallback.outcome": "failed",
                "lm.fallback.seconds": elapsed,
                "lm.fallback.error": desc,
            }
        )


def build_task_lm(**overrides: Any) -> ProviderFallbackLM:
    """The benchmark's task LM.

    ``model`` / ``provider`` / ``fallback`` / ``breaker`` select the routing
    (defaulting to ``$LM_MODEL``, ``$LM_PROVIDER``, ``$LM_FALLBACK`` and
    ``$LM_BREAKER``); ``read_gap_timeout`` / ``total_budget`` / ``max_attempts``
    tune the hang guard; every other keyword applies to both routes.
    """
    return ProviderFallbackLM(**overrides)
