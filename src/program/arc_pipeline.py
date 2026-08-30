import os
import json
import litellm


class _NoOpSpan:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): return False
    def set_attribute(self, *args, **kwargs): return None


class _NoOpTracer:
    def start_as_current_span(self, name, *args, **kwargs): return _NoOpSpan()


# Deliberately a no-op rather than a real OTLP exporter. The arc-agi (v1) seed
# imports opentelemetry.exporter.otlp, which makes the OTLP/HTTP exporter a HARD
# requirement of its benchmark image -- omit it there and the mounted evaluator
# dies on ModuleNotFoundError before the first iteration. Keeping v2 free of
# that import is why requirements.txt here is just litellm + tenacity.
tracer = _NoOpTracer()


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
        """Streaming completion with hang detection; returns content text."""
        import time as _time

        start = _time.monotonic()
        last_err = None
        for _attempt in range(self.MAX_ATTEMPTS):
            if _time.monotonic() - start > self.TOTAL_BUDGET_S - self.READ_GAP_TIMEOUT_S:
                break
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
                    if _time.monotonic() - start > self.TOTAL_BUDGET_S:
                        raise TimeoutError(
                            f"row exceeded total budget {self.TOTAL_BUDGET_S}s"
                        )
                    if chunk.choices:
                        delta = chunk.choices[0].delta
                        if delta is not None and getattr(delta, "content", None):
                            parts.append(delta.content)
                return "".join(parts)
            except Exception as exc:  # noqa: BLE001 — hang/gap/transient
                last_err = exc
        raise last_err if last_err else RuntimeError("completion failed")

    def __call__(self, train: list = None, test: list = None, task_id: str = "unknown", **kwargs) -> list:
        with tracer.start_as_current_span("arc_predict") as span:
            span.set_attribute("task_id", task_id)

            train_cases = train or []
            test_cases = test or []

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

                try:
                    content = self._complete(
                        [{"role": "user", "content": test_prompt}]
                    ).strip()

                    # Strip any <think>...</think> block defensively: a
                    # reasoning model that leaks its scratchpad into content
                    # would otherwise fail the JSON parse and score a false 0.
                    import re
                    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

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
                    else:
                        outputs.append(test_input)  # Fallback
                except Exception as e:
                    print(f"Error calling LLM or parsing response: {e}")
                    outputs.append(test_input)  # Fallback on error

            span.set_attribute("num_predictions", len(outputs))
            return outputs
