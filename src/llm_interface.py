"""
LLM interface for answer generation.

This module abstracts the language-model call behind a uniform
``generate()`` function so that the rest of the system is agnostic to
whether we are using a local HuggingFace SLM, an Ollama server, or an
OpenAI-compatible API.

SLM choice justification
--------------------------
**TinyLlama-1.1B-Chat-v1.0** is selected as the default local SLM:
  - 1.1 B parameters — fits in ≤ 4 GB VRAM (or CPU with ~3 GB RAM).
  - Chat-finetuned — follows the ChatML template, making prompt
    engineering straightforward.
  - Apache-2.0 licence — no usage restrictions.
  - Competitive quality for short-form QA when grounded with context.

For systems with more GPU memory, users can switch to larger models
(e.g. ``Qwen/Qwen2-1.5B-Instruct``, ``microsoft/phi-2``) by setting
the ``HF_MODEL_NAME`` environment variable.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    LLM_BACKEND,
    HF_MODEL_NAME,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENAI_API_BASE,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
)

# Global singleton so we only load the model once
_hf_pipeline = None


# ── HuggingFace local model ──────────────────────────────────────────

def _load_hf_pipeline():
    """Load a HuggingFace text-generation pipeline (singleton)."""
    global _hf_pipeline
    if _hf_pipeline is not None:
        return _hf_pipeline

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    print(f"Loading HuggingFace model: {HF_MODEL_NAME} ...")

    # Determine device
    if torch.cuda.is_available():
        device_map = "auto"
        dtype = torch.float16
        print("  Using CUDA (GPU)")
    else:
        device_map = "cpu"
        dtype = torch.float32
        print("  Using CPU (this may be slow)")

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_NAME,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    _hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    print("HuggingFace model loaded.\n")
    return _hf_pipeline


def _generate_hf(prompt: str, system_prompt: str = "") -> str:
    """Generate text using a local HuggingFace model."""
    pipe = _load_hf_pipeline()

    # Build chat messages for models that support chat templates
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        # Try chat-template generation first (works for chat-finetuned models)
        outputs = pipe(
            messages,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            return_full_text=False,
        )
        return outputs[0]["generated_text"].strip()
    except Exception:
        # Fallback: plain text generation
        full_prompt = ""
        if system_prompt:
            full_prompt += f"System: {system_prompt}\n\n"
        full_prompt += f"User: {prompt}\n\nAssistant:"

        outputs = pipe(
            full_prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            return_full_text=False,
        )
        text = outputs[0]["generated_text"]
        return text.strip()


# ── Ollama backend ────────────────────────────────────────────────────

def _generate_ollama(prompt: str, system_prompt: str = "") -> str:
    """Generate text using an Ollama server."""
    import requests

    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "num_predict": MAX_NEW_TOKENS,
        },
    }

    try:
        resp = requests.post(url, json=payload, timeout=600)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return (
            "[ERROR] Cannot connect to Ollama server at "
            f"{OLLAMA_BASE_URL}. Ensure Ollama is running."
        )
    except Exception as exc:
        return f"[ERROR] Ollama generation failed: {exc}"


# ── OpenAI-compatible API ─────────────────────────────────────────────

def _generate_openai(prompt: str, system_prompt: str = "") -> str:
    """Generate text using an OpenAI-compatible API endpoint."""
    import requests

    url = f"{OPENAI_API_BASE}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_NEW_TOKENS,
        "top_p": TOP_P,
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        return (
            f"[ERROR] Cannot connect to API at {OPENAI_API_BASE}. "
            "Check that the server is running."
        )
    except Exception as exc:
        return f"[ERROR] API generation failed: {exc}"


# ── Public interface ──────────────────────────────────────────────────

def generate(prompt: str, system_prompt: str = "", backend: str | None = None) -> str:
    """
    Generate text from the configured LLM backend.

    Parameters
    ----------
    prompt : str
        The user-facing prompt (may include retrieved context).
    system_prompt : str
        An optional system-level instruction.
    backend : str or None
        Override the configured backend ('huggingface', 'ollama',
        'openai_compatible').  Uses config.LLM_BACKEND if None.

    Returns
    -------
    str
        The generated text.
    """
    backend = backend or LLM_BACKEND

    if backend == "huggingface":
        return _generate_hf(prompt, system_prompt)
    elif backend == "ollama":
        return _generate_ollama(prompt, system_prompt)
    elif backend == "openai_compatible":
        return _generate_openai(prompt, system_prompt)
    else:
        return (
            f"[ERROR] Unknown LLM backend: {backend}. "
            "Set LLM_BACKEND to 'huggingface', 'ollama', or 'openai_compatible'."
        )


# ── CLI quick test ────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Testing LLM backend: {LLM_BACKEND}")
    print(f"Model: {HF_MODEL_NAME}")
    print()

    test_prompt = "Who is Hamlet? Answer in 2-3 sentences."
    test_system = "You are a Shakespeare expert. Be concise."

    print(f"Prompt: {test_prompt}")
    print(f"System: {test_system}")
    print()

    answer = generate(test_prompt, test_system)
    print(f"Answer:\n{answer}")
