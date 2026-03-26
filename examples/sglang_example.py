"""Example: TurboQuant KV cache quantization with SGLang.

Two approaches:
  1. Using TurboQuantCache with HuggingFace models (portable)
  2. Patching SGLang engine directly
"""

import torch


def sglang_example():
    """Apply TurboQuant to SGLang engine."""
    import sglang as sgl
    from turboquant.sglang_integration import apply_turboquant_to_sglang

    # Launch SGLang engine
    engine = sgl.Engine(
        model_path="meta-llama/Llama-3.1-8B-Instruct",
    )

    # Apply TurboQuant KV cache quantization
    apply_turboquant_to_sglang(engine, bit_width=3)

    # Use normally — quantization is transparent
    state = engine.generate(
        "The key insight of TurboQuant is",
        max_new_tokens=100,
        temperature=0.7,
    )
    print(state["text"])


def sglang_multi_turn_example():
    """Multi-turn conversation with quantized KV cache."""
    import sglang as sgl
    from turboquant.sglang_integration import apply_turboquant_to_sglang

    engine = sgl.Engine(model_path="meta-llama/Llama-3.1-8B-Instruct")
    apply_turboquant_to_sglang(engine, bit_width=3)

    @sgl.function
    def chat(s, question):
        s += sgl.system("You are a helpful assistant.")
        s += sgl.user(question)
        s += sgl.assistant(sgl.gen("answer", max_tokens=200))

    state = chat.run(question="Explain KV cache quantization in 2 sentences.")
    print(state["answer"])


if __name__ == "__main__":
    print("SGLang + TurboQuant example")
    print("Run with: python sglang_example.py")
    print()
    print("Note: Requires 'pip install sglang' and a GPU.")
    print("For a standalone demo without SGLang, see vllm_example.py")
