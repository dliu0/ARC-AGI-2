"""Unit tests for the cross-provider fallback task LM.

No network: ``litellm.completion`` is monkeypatched to simulate each provider,
including a provider that accepts the request and then goes silent.

These cover the contract in docs/provider_fallback.md (C1-C8). The reference
implementation there is a DSPy one; this repo's is LiteLLM-native (see the
module docstring), so the tests exercise the same behaviour through
``ProviderFallbackLM.completion`` rather than ``dspy.LM.forward``.
"""

import subprocess
import sys
import threading
import types
from pathlib import Path

import litellm
import openai
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.infra import lm_provider
from src.infra.lm_provider import (
    BREAKER_FAILURES,
    COVER_NUM_RETRIES,
    DEFAULT_FALLBACK,
    DEFAULT_PROVIDER,
    GMI_API_BASE,
    MAX_ATTEMPTS,
    MODEL_ROUTES,
    PRIMARY_NUM_RETRIES,
    PROVIDER_PREFERENCE,
    PROVIDERS,
    REASONING_EFFORT,
    TASK_MODEL,
    BudgetExceeded,
    ProviderBreaker,
    ProviderFallbackLM,
    Route,
    breaker_for,
    build_task_lm,
    is_hang,
    reset_breakers,
    resolve_fallback,
    resolve_model,
    resolve_provider,
    route_for,
    should_fallback,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Every model string comes from the one routing table -- the tests read it the
# same way the module does, so a model swap is still a one-row edit.
GMI_MODEL = MODEL_ROUTES[TASK_MODEL]["gmi"].model
DEEPSEEK_MODEL = MODEL_ROUTES[TASK_MODEL]["deepseek"].model
DEEPINFRA_MODEL = MODEL_ROUTES[TASK_MODEL]["deepinfra"].model

GMI_402_TEXT = (
    "OpenAIException - Error code: 402 - "
    "{'error': 'Insufficient balance', 'reason': 'model_access_denied'}"
)

HANG = object()  # sentinel: accept the request, then stream nothing


def api_error(status_code=402, text=GMI_402_TEXT, model=GMI_MODEL):
    return litellm.APIError(
        status_code=status_code, message=text, llm_provider="openai", model=model
    )


def hang_error(model=DEEPSEEK_MODEL):
    return litellm.Timeout(
        message="Read timed out.", model=model, llm_provider="deepseek"
    )


def _chunk(content):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content=content))]
    )


class FakeProviders:
    """Records every litellm.completion call and answers per model string.

    A behaviour is either a string (streamed back, split into chunks), an
    exception instance (raised by ``completion`` itself, as a real 4xx/5xx is),
    or ``HANG`` (``completion`` returns, then the stream raises a read timeout
    the way a silent connection does).
    """

    def __init__(self, **by_provider):
        self.behaviour = {
            GMI_MODEL: by_provider.get("gmi", "gmi-answer"),
            DEEPSEEK_MODEL: by_provider.get("deepseek", "deepseek-answer"),
            DEEPINFRA_MODEL: by_provider.get("deepinfra", "deepinfra-answer"),
        }
        self.calls = []  # one dict of kwargs per request

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.behaviour[kwargs["model"]]
        if isinstance(outcome, BaseException):
            raise outcome
        if outcome is HANG:
            def silent():
                raise hang_error(kwargs["model"])
                yield  # pragma: no cover -- generator marker
            return silent()

        def stream():
            # Deltas of a few characters, as a real provider sends them: the
            # text must reassemble byte-for-byte, whitespace included.
            for i in range(0, len(outcome), 3):
                yield _chunk(outcome[i:i + 3])

        return stream()

    @property
    def models(self):
        return [c["model"] for c in self.calls]

    def kwargs_for(self, model):
        return [c for c in self.calls if c["model"] == model]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Every test starts with no routing env and no remembered provider health."""
    for var in (
        "LM_MODEL",
        "LM_PROVIDER",
        "LM_FALLBACK",
        "LM_BREAKER",
        "LM_BREAKER_FAILURES",
        "LM_BREAKER_COOLDOWN",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GMI_API_KEY", "test-gmi-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test-deepinfra-key")
    reset_breakers()
    yield
    reset_breakers()


@pytest.fixture
def providers(monkeypatch):
    """Install a FakeProviders as litellm.completion; configure per test."""

    def install(**by_provider):
        fake = FakeProviders(**by_provider)
        monkeypatch.setattr(litellm, "completion", fake)
        return fake

    return install


MESSAGES = [{"role": "user", "content": "grid?"}]


# ---------------------------------------------------------------------------
# C1/C2 -- one pinned model, several providers, ids in one table
# ---------------------------------------------------------------------------


def test_task_model_has_a_row():
    assert TASK_MODEL in MODEL_ROUTES


def test_every_preferred_provider_serves_the_task_model():
    for provider in PROVIDER_PREFERENCE:
        assert isinstance(route_for(TASK_MODEL, provider), Route)


def test_all_routes_name_the_same_weights():
    """C1: a provider is a route to the SAME model, never a different one."""
    for provider in PROVIDER_PREFERENCE:
        model = route_for(TASK_MODEL, provider).model.lower()
        assert "deepseek" in model and "v4" in model and "flash" in model


def test_route_for_unknown_provider_names_the_row():
    with pytest.raises(ValueError, match="has no route for model"):
        route_for(TASK_MODEL, "nope")


def test_gmi_route_carries_its_own_endpoint_and_key():
    kwargs = route_for(TASK_MODEL, "gmi").request_kwargs()
    assert kwargs["api_base"] == GMI_API_BASE
    assert kwargs["api_key"] == "test-gmi-key"


def test_native_routes_pass_no_key(monkeypatch):
    """LiteLLM reads DEEPSEEK_API_KEY / DEEPINFRA_API_KEY itself, at call time,
    so the key never lands in a request record or a trace file."""
    for provider in ("deepseek", "deepinfra"):
        kwargs = route_for(TASK_MODEL, provider).request_kwargs()
        assert "api_key" not in kwargs
        assert "api_base" not in kwargs


def test_gmi_without_its_key_fails_at_construction(monkeypatch):
    monkeypatch.delenv("GMI_API_KEY", raising=False)
    with pytest.raises(KeyError):
        build_task_lm(provider="gmi")


def test_model_ids_appear_nowhere_outside_the_table():
    """C2/C8: no model id, endpoint or key lookup in the evolvable program."""
    hits = subprocess.run(
        [
            "grep", "-rnE", "DeepSeek-V4|deepseek-v4|gmi-serving|gpt-|_API_KEY",
            str(REPO_ROOT / "src" / "program"), str(REPO_ROOT / "src" / "metric"),
        ],
        capture_output=True, text=True,
    ).stdout.strip()
    assert hits == "", f"model/provider ids leaked into the program package:\n{hits}"


def test_provider_module_is_outside_the_evolvable_package():
    """C8: CodeEvolver evolves src/program; the wiring must not live there."""
    assert Path(lm_provider.__file__).parent.name == "infra"


# ---------------------------------------------------------------------------
# C3 -- one preference order drives both selection and cover
# ---------------------------------------------------------------------------


def test_default_fallback_is_derived_from_the_preference_order():
    for i, provider in enumerate(PROVIDER_PREFERENCE):
        nxt = PROVIDER_PREFERENCE[(i + 1) % len(PROVIDER_PREFERENCE)]
        assert DEFAULT_FALLBACK[provider] == nxt


def test_default_provider_is_the_head_of_the_preference_order():
    assert DEFAULT_PROVIDER == PROVIDER_PREFERENCE[0] == "gmi"


def test_unconfigured_run_is_gmi_covered_by_deepseek():
    lm = build_task_lm()
    assert (lm.provider, lm.fallback_provider) == ("gmi", "deepseek")
    assert lm.model == GMI_MODEL
    assert lm.fallback_model == DEEPSEEK_MODEL


# ---------------------------------------------------------------------------
# C7 -- configurable by environment, defaulted in code
# ---------------------------------------------------------------------------


def test_lm_provider_env_selects_the_primary(monkeypatch):
    monkeypatch.setenv("LM_PROVIDER", "gmi")
    lm = build_task_lm()
    assert (lm.provider, lm.fallback_provider) == ("gmi", "deepseek")


def test_lm_provider_env_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("LM_PROVIDER", "  GMI  ")
    assert resolve_provider() == "gmi"


def test_unknown_provider_fails_loudly(monkeypatch):
    monkeypatch.setenv("LM_PROVIDER", "openrouter")
    with pytest.raises(ValueError, match="not a known provider"):
        build_task_lm()


def test_unknown_model_fails_loudly(monkeypatch):
    monkeypatch.setenv("LM_MODEL", "llama-9")
    with pytest.raises(ValueError, match="not in MODEL_ROUTES"):
        build_task_lm()


def test_resolve_model_defaults_to_task_model():
    assert resolve_model() == TASK_MODEL
    assert resolve_model("  ") == TASK_MODEL


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "none"])
def test_lm_fallback_off_disarms_the_divert(monkeypatch, value):
    monkeypatch.setenv("LM_FALLBACK", value)
    lm = build_task_lm()
    assert lm.fallback_provider is None
    assert lm.fallback_model is None
    assert lm.breaker_enabled is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_lm_fallback_on_uses_the_next_provider(monkeypatch, value):
    monkeypatch.setenv("LM_FALLBACK", value)
    assert resolve_fallback("deepseek") == "deepinfra"


def test_lm_fallback_can_name_a_provider(monkeypatch):
    """Any provider but the primary: the default cover is deepseek, so naming
    deepinfra is what proves the env var is read rather than derived."""
    monkeypatch.setenv("LM_FALLBACK", "deepinfra")
    lm = build_task_lm()
    assert lm.fallback_provider == "deepinfra"


def test_a_provider_cannot_cover_for_itself(monkeypatch):
    monkeypatch.setenv("LM_FALLBACK", "deepseek")
    with pytest.raises(ValueError, match="cannot be its own fallback"):
        build_task_lm(provider="deepseek")


def test_nonsense_fallback_fails_loudly(monkeypatch):
    monkeypatch.setenv("LM_FALLBACK", "maybe")
    with pytest.raises(ValueError, match="neither a boolean nor a provider"):
        build_task_lm()


def test_nonsense_breaker_flag_fails_loudly(monkeypatch):
    monkeypatch.setenv("LM_BREAKER", "sometimes")
    with pytest.raises(ValueError, match="is not a boolean"):
        build_task_lm()


def test_breaker_thresholds_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("LM_BREAKER_FAILURES", "7")
    monkeypatch.setenv("LM_BREAKER_COOLDOWN", "12.5")
    breaker = breaker_for("deepseek")
    assert (breaker.threshold, breaker.cooldown) == (7, 12.5)


# ---------------------------------------------------------------------------
# C6 -- retry budget matched to the cover
# ---------------------------------------------------------------------------


def test_covered_primary_gets_the_short_retry_budget():
    lm = build_task_lm()
    assert lm._request[lm.provider]["num_retries"] == PRIMARY_NUM_RETRIES
    assert lm._request[lm.fallback_provider]["num_retries"] == COVER_NUM_RETRIES


def test_uncovered_primary_keeps_the_full_budget(monkeypatch):
    monkeypatch.setenv("LM_FALLBACK", "0")
    lm = build_task_lm()
    assert lm._request[lm.provider]["num_retries"] == COVER_NUM_RETRIES


def test_explicit_num_retries_applies_to_both_routes():
    lm = build_task_lm(num_retries=9)
    assert lm._request[lm.provider]["num_retries"] == 9
    assert lm._request[lm.fallback_provider]["num_retries"] == 9


def test_overrides_apply_to_both_routes():
    lm = build_task_lm(temperature=0.3)
    assert lm._request[lm.provider]["temperature"] == 0.3
    assert lm._request[lm.fallback_provider]["temperature"] == 0.3


def test_reasoning_is_enabled_on_both_routes():
    lm = build_task_lm()
    for provider in (lm.provider, lm.fallback_provider):
        request = lm._request[provider]
        assert request["reasoning_effort"] == REASONING_EFFORT
        assert "reasoning_effort" in request["allowed_openai_params"]


# ---------------------------------------------------------------------------
# C4 -- per-call divert, on provider errors only
# ---------------------------------------------------------------------------


def test_happy_path_never_touches_the_cover(providers):
    fake = providers(gmi="all good")
    lm = build_task_lm()
    assert lm.completion(MESSAGES) == "all good"
    assert fake.models == [GMI_MODEL]


def test_streamed_chunks_are_joined(providers):
    """The reply is assembled from deltas, not from a single response body."""
    fake = providers(gmi="a b c  d\ne")
    assert build_task_lm().completion(MESSAGES) == "a b c  d\ne"
    assert len(fake.calls) == 1


def test_empty_and_contentless_chunks_are_skipped(monkeypatch):
    """Reasoning-only deltas carry no `content`; they must not crash the join."""
    chunks = [
        types.SimpleNamespace(choices=[]),                                    # keep-alive
        types.SimpleNamespace(choices=[types.SimpleNamespace(delta=None)]),   # no delta
        _chunk(None),                                                         # reasoning only
        _chunk("real"),
    ]
    monkeypatch.setattr(litellm, "completion", lambda **kw: iter(chunks))
    assert build_task_lm().completion(MESSAGES) == "real"


def test_provider_error_diverts_to_the_cover(providers):
    fake = providers(gmi=api_error(model=GMI_MODEL), deepseek="covered")
    lm = build_task_lm()
    assert lm.completion(MESSAGES) == "covered"
    assert fake.models == [GMI_MODEL, DEEPSEEK_MODEL]


def test_the_cover_receives_the_identical_request(providers):
    fake = providers(gmi=api_error(model=GMI_MODEL), deepseek="covered")
    lm = build_task_lm(temperature=0.25)
    lm.completion(MESSAGES, max_tokens=77)

    primary, cover = fake.calls
    for key in ("messages", "temperature", "max_tokens", "reasoning_effort", "stream"):
        assert primary[key] == cover[key], key
    assert primary["model"] != cover["model"]  # ...only the route differs


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 422, 429, 500, 503])
def test_every_api_status_diverts(providers, status):
    fake = providers(
        gmi=api_error(status_code=status, model=GMI_MODEL), deepseek="covered"
    )
    assert build_task_lm().completion(MESSAGES) == "covered"
    assert fake.models == [GMI_MODEL, DEEPSEEK_MODEL]


def test_connection_error_diverts(providers):
    fake = providers(
        gmi=litellm.APIConnectionError(
            message="conn reset", llm_provider="openai", model=GMI_MODEL
        ),
        deepseek="covered",
    )
    assert build_task_lm().completion(MESSAGES) == "covered"


def test_context_window_error_is_the_programs_own_fault(providers):
    """C4: the same prompt would not fit on the cover either."""
    exc = litellm.ContextWindowExceededError(
        message="too long", model=GMI_MODEL, llm_provider="openai"
    )
    fake = providers(gmi=exc, deepseek="covered")
    with pytest.raises(litellm.ContextWindowExceededError):
        build_task_lm().completion(MESSAGES)
    assert fake.models == [GMI_MODEL]  # never diverted


def test_program_exceptions_are_never_diverted(providers):
    fake = providers(gmi=ValueError("program bug"), deepseek="covered")
    with pytest.raises(ValueError, match="program bug"):
        build_task_lm().completion(MESSAGES)
    assert fake.models == [GMI_MODEL]


def test_should_fallback_classification():
    assert should_fallback(api_error()) is True
    assert should_fallback(hang_error()) is True
    assert should_fallback(ValueError("nope")) is False
    assert should_fallback(BudgetExceeded("out of time")) is False
    assert (
        should_fallback(
            litellm.ContextWindowExceededError(
                message="x", model=GMI_MODEL, llm_provider="openai"
            )
        )
        is False
    )


def test_both_providers_failing_raises_the_covers_error(providers):
    providers(
        gmi=api_error(status_code=402, model=GMI_MODEL),
        deepseek=api_error(status_code=500, model=DEEPSEEK_MODEL),
    )
    with pytest.raises(litellm.APIError) as caught:
        build_task_lm().completion(MESSAGES)
    assert caught.value.status_code == 500
    assert isinstance(caught.value.__cause__, litellm.APIError)  # primary chained


def test_uncovered_errors_propagate(monkeypatch, providers):
    monkeypatch.setenv("LM_FALLBACK", "0")
    fake = providers(gmi=api_error(model=GMI_MODEL))
    with pytest.raises(litellm.APIError):
        build_task_lm().completion(MESSAGES)
    assert fake.models == [GMI_MODEL]


def test_divert_is_logged_and_visible(providers, capsys):
    providers(gmi=api_error(model=GMI_MODEL), deepseek="covered")
    build_task_lm().completion(MESSAGES)
    err = capsys.readouterr().err
    assert f"[WARNING] gmi error on {GMI_MODEL} (status=402" in err
    assert "retrying this call on deepseek" in err
    assert "deepseek fallback succeeded" in err


# ---------------------------------------------------------------------------
# The hang guard
# ---------------------------------------------------------------------------


def test_a_hang_is_retried_on_the_same_route_then_diverted(providers):
    fake = providers(gmi=HANG, deepseek="covered")
    assert build_task_lm().completion(MESSAGES) == "covered"
    # MAX_ATTEMPTS on the primary, then the cover.
    assert fake.models == [GMI_MODEL] * MAX_ATTEMPTS + [DEEPSEEK_MODEL]


def test_a_hang_counts_against_provider_health(providers):
    providers(gmi=HANG, deepseek="covered")
    lm = build_task_lm()
    for _ in range(BREAKER_FAILURES):
        lm.completion(MESSAGES)
    assert lm.breaker.state == "open"


def test_an_api_error_is_not_retried_on_the_same_route(providers):
    """LiteLLM's own num_retries already covered the transient statuses."""
    fake = providers(gmi=api_error(model=GMI_MODEL), deepseek="covered")
    build_task_lm().completion(MESSAGES)
    assert fake.models.count(GMI_MODEL) == 1


def test_read_gap_timeout_is_passed_per_request(providers):
    fake = providers(gmi="ok")
    build_task_lm(read_gap_timeout=17).completion(MESSAGES)
    assert fake.calls[0]["timeout"] == 17
    assert fake.calls[0]["stream"] is True


def test_budget_exhaustion_is_not_diverted(providers):
    """C4-adjacent: with the row's budget gone there is nowhere useful to go."""
    fake = providers(gmi=HANG, deepseek="covered")
    lm = build_task_lm(total_budget=0)
    with pytest.raises(BudgetExceeded):
        lm.completion(MESSAGES)
    assert fake.calls == []  # not even one attempt


def test_budget_exceeded_is_not_an_api_error():
    assert not isinstance(BudgetExceeded("x"), openai.APIError)
    assert is_hang(BudgetExceeded("x")) is False
    assert is_hang(hang_error()) is True


def test_max_attempts_is_configurable(providers):
    fake = providers(gmi=HANG, deepseek="covered")
    build_task_lm(max_attempts=1).completion(MESSAGES)
    assert fake.models == [GMI_MODEL, DEEPSEEK_MODEL]


# ---------------------------------------------------------------------------
# C5 -- a sustained outage costs O(1), not O(calls)
# ---------------------------------------------------------------------------


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def test_breaker_opens_after_consecutive_failures():
    clock = Clock()
    breaker = ProviderBreaker("gmi", threshold=3, cooldown=60, clock=clock)
    assert breaker.state == "closed"
    for _ in range(2):
        breaker.record_failure()
    assert breaker.state == "closed"
    breaker.record_failure()
    assert breaker.state == "open"
    assert breaker.allow() is False


def test_a_stray_error_never_trips_the_breaker():
    """A lone 402 followed by a success must not mark a provider unhealthy."""
    breaker = ProviderBreaker("gmi", threshold=3, cooldown=60, clock=Clock())
    for _ in range(20):
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_success()
    assert breaker.state == "closed"
    assert breaker.allow() is True


def test_open_breaker_probes_once_after_the_cooldown():
    clock = Clock()
    breaker = ProviderBreaker("gmi", threshold=1, cooldown=60, clock=clock)
    breaker.record_failure()

    clock.advance(59)
    assert breaker.allow() is False

    clock.advance(2)
    assert breaker.allow() is True       # the probe
    assert breaker.state == "probing"
    assert breaker.allow() is False      # only one probe in flight

    breaker.record_success()
    assert breaker.state == "closed"


def test_a_failed_probe_restarts_the_cooldown():
    clock = Clock()
    breaker = ProviderBreaker("gmi", threshold=1, cooldown=60, clock=clock)
    breaker.record_failure()
    clock.advance(61)
    assert breaker.allow() is True
    breaker.record_failure()             # probe failed
    assert breaker.state == "open"
    clock.advance(59)
    assert breaker.allow() is False
    clock.advance(2)
    assert breaker.allow() is True


def test_breakers_are_process_global_not_per_lm():
    a, b = build_task_lm(), build_task_lm()
    assert a.breaker is b.breaker is breaker_for("gmi")


def test_reset_breakers_forgets_health():
    first = breaker_for("gmi")
    reset_breakers()
    assert breaker_for("gmi") is not first


def test_an_open_breaker_stops_paying_the_primary(providers):
    fake = providers(gmi=api_error(model=GMI_MODEL), deepseek="covered")
    lm = build_task_lm()

    for _ in range(BREAKER_FAILURES):
        assert lm.completion(MESSAGES) == "covered"
    assert lm.breaker.state == "open"

    fake.calls.clear()
    for _ in range(25):
        assert lm.completion(MESSAGES) == "covered"
    assert fake.models == [DEEPSEEK_MODEL] * 25  # zero primary attempts


def test_a_skipped_primary_still_answers_from_the_cover(providers):
    providers(gmi=api_error(model=GMI_MODEL), deepseek="covered")
    lm = build_task_lm()
    for _ in range(BREAKER_FAILURES):
        lm.completion(MESSAGES)
    assert lm.completion(MESSAGES) == "covered"


def test_success_resets_the_failure_count_end_to_end(providers):
    fake = providers(gmi=api_error(model=GMI_MODEL), deepseek="covered")
    lm = build_task_lm()
    lm.completion(MESSAGES)
    lm.completion(MESSAGES)
    fake.behaviour[GMI_MODEL] = "recovered"
    assert lm.completion(MESSAGES) == "recovered"

    fake.behaviour[GMI_MODEL] = api_error(model=GMI_MODEL)
    lm.completion(MESSAGES)
    lm.completion(MESSAGES)
    assert lm.breaker.state == "closed"  # count restarted after the success


def test_breaker_can_be_disabled(monkeypatch, providers):
    monkeypatch.setenv("LM_BREAKER", "0")
    fake = providers(gmi=api_error(model=GMI_MODEL), deepseek="covered")
    lm = build_task_lm()
    assert lm.breaker is None
    for _ in range(6):
        lm.completion(MESSAGES)
    assert fake.models.count(GMI_MODEL) == 6  # every call re-tried the primary


def test_breaker_is_inert_without_a_cover(monkeypatch):
    monkeypatch.setenv("LM_FALLBACK", "0")
    assert build_task_lm().breaker is None


def test_breaker_state_is_shared_across_threads(providers):
    fake = providers(gmi=api_error(model=GMI_MODEL), deepseek="covered")
    lm = build_task_lm()
    errors = []

    def run():
        try:
            for _ in range(8):
                lm.completion(MESSAGES)
        except Exception as exc:  # pragma: no cover -- surfaced via `errors`
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert lm.breaker.state == "open"
    # 80 calls, but the primary was attempted only while the breaker was closed.
    assert fake.models.count(GMI_MODEL) < 80
    assert fake.models.count(DEEPSEEK_MODEL) >= 1


def test_transitions_are_logged_once_not_per_call(providers, capsys):
    providers(gmi=api_error(model=GMI_MODEL), deepseek="covered")
    lm = build_task_lm()
    for _ in range(BREAKER_FAILURES + 10):
        lm.completion(MESSAGES)
    err = capsys.readouterr().err
    assert err.count("marking it unhealthy") == 1
