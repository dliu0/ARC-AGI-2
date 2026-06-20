import os
import json
import litellm


class _NoOpSpan:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): return False
    def set_attribute(self, *args, **kwargs): return None


class _NoOpTracer:
    def start_as_current_span(self, name, *args, **kwargs): return _NoOpSpan()


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
        self.model = "openai/gpt-5.4-mini"

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
                    response = litellm.completion(
                        model=self.model,
                        messages=[{"role": "user", "content": test_prompt}],
                        temperature=0.0,
                    )
                    content = response.choices[0].message.content.strip()

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
