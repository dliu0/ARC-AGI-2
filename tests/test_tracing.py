"""The ARC pipeline must emit readable spans for its LLM calls.

Before this, `tracer` was a hard no-op, so a row produced exactly ONE span —
the harness's own root, carrying the returned grids. Two things were therefore
invisible:

  1. every LLM call, including the hang-guard retries (measured at ~4.7% of
     rows on arc-agi);
  2. the JSON-parse fallback, which appends `test_input` — the program answers
     with the QUESTION. That scores 0 but is indistinguishable in the output
     from a genuine wrong answer, so a run dominated by unparseable responses
     would read as a reasoning failure.

Deliberately API-only: no OTLP exporter import, so `requirements.txt` stays
litellm + tenacity (importing the exporter is what made it a hard image
requirement in arc-agi v1).

Run with: python -m pytest tests/test_tracing.py
Needs `opentelemetry-sdk` (the harness installs it into every client venv).
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import types

PIPELINE_PATH = pathlib.Path(__file__).resolve().parents[1] / "src" / "program" / "arc_pipeline.py"


def _load_pipeline_module():
    """Load `arc_pipeline` by path.

    This repo's `src/` has no `__init__.py`, so `import src.program...` resolves
    against whatever `src` package happens to be importable — which is not this
    one when the test borrows another project's venv. In production the mounted
    evaluator merges the client tree into `src.__path__`
    (`evaluate.py::_add_workspace_to_import_path`); for a standalone test,
    loading the file directly is unambiguous.
    """
    sys.modules.pop("arc_pipeline_under_test", None)
    spec = importlib.util.spec_from_file_location("arc_pipeline_under_test", PIPELINE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["arc_pipeline_under_test"] = module
    spec.loader.exec_module(module)
    return module

TRAIN = [{"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]}]
TEST = [{"input": [[1, 1], [0, 0]]}]
GRID_RESPONSE = "[[0, 0], [1, 1]]"


def _install_fake_litellm(monkeypatch, *, fail_times: int = 0, response: str = GRID_RESPONSE):
    calls = {"n": 0}

    class _Delta:
        def __init__(self, c):
            self.content = c

    class _Choice:
        def __init__(self, c):
            self.delta = _Delta(c)

    class _Chunk:
        def __init__(self, c):
            self.choices = [_Choice(c)]

    def completion(**kw):
        calls["n"] += 1
        # The seed's provider wiring and hang guard are load-bearing.
        assert kw["reasoning_effort"] == "high"
        assert kw["stream"] is True
        assert kw["timeout"] == 240
        assert kw["api_base"] == "https://api.gmi-serving.com/v1"
        if calls["n"] <= fail_times:
            raise RuntimeError(f"simulated hang #{calls['n']}")
        return [_Chunk(response[i : i + 8]) for i in range(0, len(response), 8)]

    fake = types.ModuleType("litellm")
    fake.completion = completion
    monkeypatch.setitem(sys.modules, "litellm", fake)
    return calls


def _run(monkeypatch, *, train=TRAIN, test=TEST, **stub_kwargs):
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    _install_fake_litellm(monkeypatch, **stub_kwargs)

    exporter = InMemorySpanExporter()
    existing = trace.get_tracer_provider()
    if hasattr(existing, "add_span_processor"):
        existing.add_span_processor(SimpleSpanProcessor(exporter))
    else:
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

    out = _load_pipeline_module().ARCPipeline()(train=train, test=test, task_id="t-123")
    return out, list(exporter.get_finished_spans())


def _by_name(spans, name):
    return [s for s in spans if s.name == name]


def test_a_solved_row_emits_chain_llm_and_solve_spans(monkeypatch):
    out, spans = _run(monkeypatch)

    assert out == [[[0, 0], [1, 1]]], "the prediction contract must not change"

    chain = _by_name(spans, "arc_predict")[0]
    assert dict(chain.attributes)["ce.span_kind"] == "chain"
    assert dict(chain.attributes)["task_id"] == "t-123"
    assert dict(chain.attributes)["ce.inputs.num_demonstrations"] == 1
    assert dict(chain.attributes)["num_predictions"] == 1

    llm = _by_name(spans, "arc_llm")[0]
    attrs = dict(llm.attributes)
    assert attrs["ce.span_kind"] == "llm"
    assert attrs["ce.output"] == GRID_RESPONSE
    assert attrs["ce.inputs.reasoning_effort"] == "high"
    assert attrs["attempts"] == 1
    assert attrs["stream_chunks"] > 1

    parse = _by_name(spans, "solve_test_case")[0]
    pattrs = dict(parse.attributes)
    assert pattrs["fell_back_to_input"] is False
    assert pattrs["predicted_shape"] == "2x2"
    assert pattrs["ce.inputs.test_input_shape"] == "2x2"


def test_one_llm_span_per_test_case(monkeypatch):
    two_tests = [{"input": [[1]]}, {"input": [[2]]}]
    _, spans = _run(monkeypatch, test=two_tests, response="[[9]]")

    assert len(_by_name(spans, "arc_llm")) == 2
    assert len(_by_name(spans, "solve_test_case")) == 2
    assert len(_by_name(spans, "arc_predict")) == 1


def test_unparseable_response_is_flagged_as_echoing_the_input(monkeypatch):
    """The failure that used to be invisible: the program returns the question."""
    out, spans = _run(monkeypatch, response="I cannot solve this puzzle.")

    assert out == [[[1, 1], [0, 0]]], "the fallback still echoes the test input"
    pattrs = dict(_by_name(spans, "solve_test_case")[0].attributes)
    assert pattrs["fell_back_to_input"] is True
    assert "echoed the test input" in pattrs["ce.error"]
    assert "cannot solve" in pattrs["ce.output"], "the unusable response must be inspectable"
    # The LLM call itself succeeded — that distinction is the point.
    assert "ce.error" not in dict(_by_name(spans, "arc_llm")[0].attributes)


def test_valid_json_that_is_not_a_grid_is_flagged_separately(monkeypatch):
    out, spans = _run(monkeypatch, response='{"answer": "nope"}')

    assert out == [[[1, 1], [0, 0]]]
    pattrs = dict(_by_name(spans, "solve_test_case")[0].attributes)
    assert pattrs["fell_back_to_input"] is True
    assert "not a grid" in pattrs["ce.error"]


def test_leaked_think_block_is_recorded(monkeypatch):
    """The strip is defensive; whether it fired is worth knowing."""
    _, spans = _run(
        monkeypatch, response="<think>hmm, rotate it</think>\n[[0, 0], [1, 1]]"
    )

    pattrs = dict(_by_name(spans, "solve_test_case")[0].attributes)
    assert pattrs["leaked_think_block"] is True
    assert pattrs["fell_back_to_input"] is False


def test_retry_after_a_hang_is_visible(monkeypatch):
    out, spans = _run(monkeypatch, fail_times=1)

    assert out == [[[0, 0], [1, 1]]], "a recovered retry must still predict"
    attrs = dict(_by_name(spans, "arc_llm")[0].attributes)
    assert attrs["attempts"] == 2
    assert "simulated hang #1" in attrs["recovered_after_errors"]


def test_exhausted_attempts_record_every_error_and_fall_back(monkeypatch):
    out, spans = _run(monkeypatch, fail_times=99)

    assert out == [[[1, 1], [0, 0]]], "an exhausted call echoes the test input"
    attrs = dict(_by_name(spans, "arc_llm")[0].attributes)
    assert attrs["attempts"] == 2
    assert "simulated hang" in attrs["ce.error"]
    assert dict(_by_name(spans, "solve_test_case")[0].attributes)["fell_back_to_input"] is True


def test_span_content_stays_bounded_for_many_large_demonstrations(monkeypatch):
    """Demonstration grids make the prompt long; the row's full input is already
    in row.json, so it must not be copied into the span."""
    big = [
        {"input": [[i % 10] * 30 for _ in range(30)], "output": [[0] * 30 for _ in range(30)]}
        for i in range(8)
    ]
    _, spans = _run(monkeypatch, train=big)

    llm = _by_name(spans, "arc_llm")[0]
    payload = json.dumps({k: str(v) for k, v in llm.attributes.items()})
    assert len(payload) < 20_000, f"span payload grew to {len(payload)} bytes"
    messages = json.loads(dict(llm.attributes)["ce.inputs.messages"])
    assert messages[0]["chars"] > 20_000, "the true prompt length must still be reported"
    assert "elided" in messages[0]["content"]


def test_pipeline_works_without_opentelemetry(monkeypatch):
    """Running this file outside the harness venv must not fail."""
    _install_fake_litellm(monkeypatch)
    ap = _load_pipeline_module()

    monkeypatch.setattr(ap, "_otel_trace", None)
    assert ap.ARCPipeline()(train=TRAIN, test=TEST) == [[[0, 0], [1, 1]]]
