import json
import re

# Provider wiring -- the pinned model, the provider preference order, the
# per-call cross-provider divert, the health flag and the streaming hang guard
from src.infra.lm_provider import build_task_lm

try:  
    from ._tracing import traceable
except ImportError:  # loaded as a top-level module rather than a package member
    from _tracing import traceable

try:
    from opentelemetry import trace as _otel_trace
except Exception:  
    _otel_trace = None

try:
    from openinference.instrumentation.litellm import LiteLLMInstrumentor
    LiteLLMInstrumentor().instrument()
except Exception:  
    pass


def _set_span_attr(key, value):
    """Attach an extra attribute to the span @traceable already opened."""
    if _otel_trace is None:
        return
    span = _otel_trace.get_current_span()
    if span is not None and span.is_recording():
        span.set_attribute(key, value)


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
        # The benchmark's pinned model, served by a ranked list of providers:
        # a provider error (or a hung connection) re-issues the identical
        # request on the cover provider, and a sustained outage is skipped
        # outright. Which provider serves a call is chosen by $LM_PROVIDER /
        # $LM_FALLBACK, not here -- the model id, endpoint and key lookup live
        # in the routing table, so none of them belongs in this file.
        self.lm = build_task_lm()

    @traceable("chain")
    def __call__(self, train: list = None, test: list = None, task_id: str = "unknown", **kwargs) -> list:
        train_cases = train or []
        test_cases = test or []

        outputs = []
        for i, test_case in enumerate(test_cases):
            outputs.append(self._solve_case(train_cases, test_case.get("input", []), i))
        return outputs

    @traceable("function")
    def _solve_case(self, train: list, test_input: list, case_index: int) -> list:
        """Solve one test input. Falls back to echoing the input on any failure."""
        prompt = self._build_prompt(train, test_input)
        try:
            content = self._call_llm(prompt)
            return self._parse_grid(content)
        except Exception as exc:
            # Recorded as metadata rather than ce.error: the fallback is a
            # deliberate, successful return, but the architect still needs to
            # know this row never produced a real prediction.
            _set_span_attr("fallback_reason", f"{type(exc).__name__}: {exc}")
            return test_input

    def _build_prompt(self, train_cases: list, test_input: list) -> str:
        """Not traced: its output is the prompt, which is already the input of
        the `_call_llm` span. A span here would duplicate that payload."""
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

        prompt += f"Test Case:\nInput: {json.dumps(test_input)}\n\n"
        prompt += (
            "Output ONLY a valid JSON array of arrays (the output grid) and nothing "
            "else. No markdown, no explanation."
        )
        return prompt

    @traceable("llm")
    def _call_llm(self, prompt: str) -> str:
        """One task-LM call, on whichever provider is serving this run.

        The request is streamed and guarded against a hung provider, and a
        provider error is re-issued on the cover -- all inside `self.lm`, so any
        LLM call added to this program inherits that simply by going through it.
        """
        return self.lm.completion([{"role": "user", "content": prompt}]).strip()

    @traceable("tool")
    def _parse_grid(self, content: str) -> list:
        """Strip reasoning leakage and markdown fencing, then decode the grid.

        Raises on malformed output so the failure shows up as `ce.error` on this
        span -- that is how the architect tells a bad parse apart from a bad
        answer.
        """
        # A reasoning model that leaks its scratchpad into `content` would
        # otherwise fail the JSON parse and score a false 0.
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        if content.startswith("```json"):
            content = content.split("```json")[1]
        if content.startswith("```"):
            content = content.split("```")[1]
        if content.endswith("```"):
            content = content.rsplit("```", 1)[0]

        content = content.strip()
        prediction = json.loads(content)

        if not (isinstance(prediction, list) and all(isinstance(row, list) for row in prediction)):
            raise ValueError(f"model output is not a list of lists: {type(prediction).__name__}")
        return prediction
