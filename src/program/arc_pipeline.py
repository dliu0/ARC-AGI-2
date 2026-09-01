import os
import json
import sys
import time
import litellm


class _NoOpSpan:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): return False
    def set_attribute(self, *args, **kwargs): return None


class _NoOpTracer:
    def start_as_current_span(self, name, *args, **kwargs): return _NoOpSpan()


# --- Tracing -----------------------------------------------------------------
# Still NO OTLP exporter: the arc-agi (v1) seed imports
# opentelemetry.exporter.otlp, which makes the OTLP/HTTP exporter a HARD
# requirement of its benchmark image -- omit it there and the mounted evaluator
# dies on ModuleNotFoundError before the first iteration. This uses only
# `opentelemetry.trace` (the API), which the engine installs into every client
# venv, so requirements.txt stays litellm + tenacity. v1 also ships its spans
# to http://127.0.0.1:4318 where nothing listens; the harness collects spans by
# attaching its own processor to whatever provider is installed, so no exporter
# is needed here at all.
#
# Until now `tracer` was a hard no-op, so this program emitted NOTHING: the
# only span a row produced was the harness's own root span carrying the return
# value. Every LLM call, and every JSON-parse failure that silently falls back
# to echoing the test input, was invisible to the architect.
#
# The tracer is resolved lazily because the engine installs its provider AFTER
# importing this module.
try:
    from opentelemetry import trace as _otel_trace
except ImportError:  # pragma: no cover - only outside the harness venv
    _otel_trace = None


class _LazyTracer:
    def start_as_current_span(self, name, *args, **kwargs):
        if _otel_trace is None:
            return _NoOpSpan()
        return _otel_trace.get_tracer("arc_agi_2.arc_pipeline").start_as_current_span(
            name, *args, **kwargs
        )


tracer = _LazyTracer()

MSG_EXCERPT_CHARS = 2000
RESPONSE_MAX_CHARS = 20000


def _set(span, key, value):
    """Set a `ce.*`/metadata attribute, coercing to a primitive. Never raises:
    a tracing failure must not fail a row."""
    try:
        span.set_attribute(
            key, value if isinstance(value, (str, int, float, bool)) else str(value)
        )
    except Exception:  # noqa: BLE001
        pass


def _summarize_messages(messages) -> str:
    """`[{role, chars, content}]` as JSON, each content bounded.

    The demonstration grids make these prompts long, and the row's full input
    is already stored in `row.json` — copying it into a span would multiply
    trace size for no new information, and these traces are read by an agent
    with a context budget.
    """
    out = []
    for m in messages or []:
        content = str(m.get("content") or "")
        excerpt = content[:MSG_EXCERPT_CHARS]
        if len(content) > MSG_EXCERPT_CHARS:
            excerpt += (
                f"\n... [{len(content) - MSG_EXCERPT_CHARS} more chars elided; "
                "the row's full input is in row.json]"
            )
        out.append({"role": m.get("role"), "chars": len(content), "content": excerpt})
    return json.dumps(out, indent=2)


def _grid_shape(grid) -> str:
    try:
        return f"{len(grid)}x{len(grid[0])}" if grid and isinstance(grid[0], list) else "empty"
    except Exception:  # noqa: BLE001
        return "unknown"


class ARCPipeline:
    """Seed baseline for ARC-AGI-2.

    A deliberately simple, fully general single-prompt LLM solver: for each
    test input it shows the model the demonstration pairs and asks it to deduce
    the abstract transformation and emit the output grid as JSON. There are no
    task-specific or hardcoded transformation rules here on purpose — the point
    is to evolve a *general* rule-discovery procedure, not to memorize rules
    that pass the validation tasks but fail held-out ones.
    """

    def __init__(self):
        # Solver LM: DeepSeek-V4-Flash on GMI Cloud (reasoning=high set on the
        # call below), matching the arc-agi seed so the two ARC arms differ in
        # the benchmark and nothing else. The optimizer runs on DeepInfra
        # GLM-5.2, so the GMI endpoint + key are passed explicitly here rather
        # than relying on OPENAI_* env, which belongs to neither provider.
        self.model = "openai/deepseek-ai/DeepSeek-V4-Flash"
        self.api_base = "https://api.gmi-serving.com/v1"
        self.api_key = os.environ.get("GMI_CLOUD_API_KEY") or os.environ.get("GMI_API_KEY")

    # --- Hang guard -------------------------------------------------------
    # GMI occasionally "hangs" a request: zero bytes until its gateway kills
    # the connection at ~20 min. Measured on arc-agi at ~4.7% of rows, enough
    # that nearly EVERY parallel eval walls at the straggler's timeout.
    # Streaming makes hangs detectable: GMI streams reasoning deltas
    # continuously (measured max inter-chunk gap ~3s), so READ_GAP_TIMEOUT_S of
    # total silence is an unambiguous hang -> abort fast and retry instead of
    # waiting for the gateway. httpx applies `timeout` per READ on a stream, so
    # long generations are unaffected; TOTAL_BUDGET_S caps the row across all
    # attempts.
    READ_GAP_TIMEOUT_S = 240
    TOTAL_BUDGET_S = 2400
    MAX_ATTEMPTS = 2

    def _complete(self, messages):
        """Streaming completion with hang detection; returns content text.

        Emits one `llm` span per call recording the request, the response and
        how the hang guard behaved (attempts, stream chunks, elapsed, the error
        of every failed attempt). Hangs were measured at ~4.7% of rows on
        arc-agi and previously left no trace at all.
        """
        start = time.monotonic()
        last_err = None
        attempts = 0
        chunks = 0
        errors = []
        with tracer.start_as_current_span("arc_llm") as span:
            _set(span, "ce.span_kind", "llm")
            _set(span, "ce.inputs.messages", _summarize_messages(messages))
            _set(span, "ce.inputs.model", self.model)
            _set(span, "ce.inputs.api_base", self.api_base)
            _set(span, "ce.inputs.reasoning_effort", "high")
            _set(span, "gen_ai.request.model", self.model)
            for _attempt in range(self.MAX_ATTEMPTS):
                if time.monotonic() - start > self.TOTAL_BUDGET_S - self.READ_GAP_TIMEOUT_S:
                    errors.append("skipped attempt: too little of the total budget left")
                    break
                attempts += 1
                try:
                    stream = litellm.completion(
                        model=self.model,
                        api_base=self.api_base,
                        api_key=self.api_key,
                        messages=messages,
                        reasoning_effort="high",
                        allowed_openai_params=["reasoning_effort"],
                        stream=True,
                        # Per-read gap cap on a stream (NOT total duration): only
                        # trips when the connection goes fully silent (real hang).
                        timeout=self.READ_GAP_TIMEOUT_S,
                    )
                    parts = []
                    for chunk in stream:
                        if time.monotonic() - start > self.TOTAL_BUDGET_S:
                            raise TimeoutError(
                                f"row exceeded total budget {self.TOTAL_BUDGET_S}s"
                            )
                        if chunk.choices:
                            chunks += 1
                            delta = chunk.choices[0].delta
                            if delta is not None and getattr(delta, "content", None):
                                parts.append(delta.content)
                    content = "".join(parts)
                    _set(span, "ce.output", content[:RESPONSE_MAX_CHARS])
                    _set(span, "response_chars", len(content))
                    _set(span, "attempts", attempts)
                    _set(span, "stream_chunks", chunks)
                    _set(span, "elapsed_s", round(time.monotonic() - start, 1))
                    if errors:
                        _set(span, "recovered_after_errors", json.dumps(errors))
                    return content
                except Exception as exc:  # noqa: BLE001 — hang/gap/transient
                    last_err = exc
                    errors.append(f"{type(exc).__name__}: {exc}")
            _set(span, "attempts", attempts)
            _set(span, "stream_chunks", chunks)
            _set(span, "elapsed_s", round(time.monotonic() - start, 1))
            _set(span, "ce.error", json.dumps(errors) if errors else "completion failed")
        raise last_err if last_err else RuntimeError("completion failed")

    def __call__(self, train: list = None, test: list = None, task_id: str = "unknown", **kwargs) -> list:
        with tracer.start_as_current_span("arc_predict") as span:
            _set(span, "ce.span_kind", "chain")
            _set(span, "task_id", task_id)

            train_cases = train or []
            test_cases = test or []
            _set(span, "ce.inputs.num_demonstrations", len(train_cases))
            _set(span, "ce.inputs.num_test_cases", len(test_cases))

            prompt = (
                "You are an expert at abstraction and reasoning. You will be given a few "
                "demonstration pairs of input and output grids. Deduce the abstract "
                "transformation rule that maps each input to its output, then apply that same "
                "rule to the final test input grid.\n\n"
            )

            prompt += "Demonstrations:\n"
            for i, case in enumerate(train_cases):
                prompt += f"Pair {i + 1}:\n"
                prompt += f"Input: {json.dumps(case.get('input'))}\n"
                prompt += f"Output: {json.dumps(case.get('output'))}\n\n"

            outputs = []
            for test_case in test_cases:
                test_input = test_case.get("input", [])

                test_prompt = prompt + f"Test Case:\nInput: {json.dumps(test_input)}\n\n"
                test_prompt += (
                    "Output ONLY a valid JSON array of arrays (the output grid) and nothing "
                    "else. No markdown, no explanation."
                )

                # One span per test case, covering the call, the parse and the
                # fallback. The fallback below appends `test_input` — the program answers
                # with the QUESTION. That scores 0 but is indistinguishable in
                # the output from a genuine wrong answer, so a run could be
                # dominated by unparseable responses and look like a reasoning
                # failure. This span separates the two.
                with tracer.start_as_current_span("solve_test_case") as pspan:
                    _set(pspan, "ce.span_kind", "chain")
                    _set(pspan, "ce.inputs.test_input_shape", _grid_shape(test_input))
                    try:
                        raw = self._complete(
                            [{"role": "user", "content": test_prompt}]
                        ).strip()
                        content = raw

                        # Strip any <think>...</think> block defensively: a
                        # reasoning model that leaks its scratchpad into content
                        # would otherwise fail the JSON parse and score a false 0.
                        import re
                        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                        _set(pspan, "leaked_think_block", content != raw)

                        if content.startswith("```json"):
                            content = content.split("```json")[1]
                        if content.startswith("```"):
                            content = content.split("```")[1]
                        if content.endswith("```"):
                            content = content.rsplit("```", 1)[0]

                        content = content.strip()
                        prediction = json.loads(content)

                        if isinstance(prediction, list) and all(isinstance(row, list) for row in prediction):
                            outputs.append(prediction)
                            _set(pspan, "ce.output", json.dumps(prediction)[:RESPONSE_MAX_CHARS])
                            _set(pspan, "predicted_shape", _grid_shape(prediction))
                            _set(pspan, "fell_back_to_input", False)
                        else:
                            outputs.append(test_input)  # Fallback
                            _set(pspan, "ce.error",
                                 "parsed JSON is not a grid (list of lists) — echoed the test input")
                            _set(pspan, "ce.output", json.dumps(prediction)[:2000])
                            _set(pspan, "fell_back_to_input", True)
                    except Exception as e:
                        print(f"Error calling LLM or parsing response: {e}", file=sys.stderr)
                        outputs.append(test_input)  # Fallback on error
                        _set(pspan, "ce.error", f"{type(e).__name__}: {e} — echoed the test input")
                        _set(pspan, "ce.output", str(locals().get("content", ""))[:2000])
                        _set(pspan, "fell_back_to_input", True)

            _set(span, "num_predictions", len(outputs))
            _set(span, "ce.output", json.dumps([_grid_shape(o) for o in outputs]))
            return outputs
