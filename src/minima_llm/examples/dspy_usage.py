#!/usr/bin/env python3
"""DSPy integration example for minima-llm."""
import asyncio
import dspy
from minima_llm import MinimaLlmConfig, OpenAIMinimaLlm
from minima_llm.dspy_adapter import MinimaLlmDSPyLM


class QuestionAnswering(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.OutputField()


async def main() -> None:
    cfg = MinimaLlmConfig.from_env()
    backend = OpenAIMinimaLlm(cfg)

    lm = MinimaLlmDSPyLM(backend)
    dspy.configure(lm=lm)

    predictor = dspy.ChainOfThought(QuestionAnswering)

    questions = [
        "What is the capital of France?",
        "What is 2+2?",
        "Who wrote Hamlet?",
    ]

    try:
        inputs = [{"question": q} for q in questions]
        results = await backend.run_batched_callable(
            inputs,
            lambda inp: predictor.acall(**inp)
        )

        for q, r in zip(questions, results):
            print(f"Q: {q}")
            print(f"A: {r.answer}\n")

    finally:
        await backend.aclose()


if __name__ == "__main__":
    asyncio.run(main())
