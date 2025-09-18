# candle_rwkv7

A Rust implementation and tooling for RWKV v7 models using the Candle library by Huggingface--See: https://github.com/huggingface/candle.
This repository provides:
- A Rust-native model implementation (src/models/rwkv7.rs) compatible with the RWKV v7 reference.
- Tools and examples to convert PyTorch .pth checkpoints to safetensors, run interactive chat, and verify numerical parity against the official Python reference.

This README is written for developers and reviewers: it explains repository layout, how to convert and prepare models, how to run examples and verification, important implementation notes, and contribution guidance.

---

Table of Contents
- Project overview
- Quickstart (convert a model, run chat)
- Examples
- Verification & testing
- Architecture and important implementation details
- Development environment and build instructions
- Troubleshooting
- Contributing & PR guidance
- License and credits

---

Project overview
This project ports RWKV v7 inference to Rust on top of Candle (and candle_nn). Goals:
- Correctness: provide a mathematically equivalent implementation to the RWKV v7 numpy/PyTorch reference.
- Usability: provide conversion tooling and CLI examples (chat, verification, conversion).
- Performance: leverage Candle and CUDA when available; fall back to CPU.

Key components:
- src/models/rwkv7.rs — Model implementation, Tokenizer, Config, and State.
- convert_pth_direct.py — Python script that converts PyTorch .pth RWKV7 checkpoints to safetensors and can emit a config.json (uses torch).
- examples/ — CLI examples:
  - chat.rs — interactive chat CLI
  - verify_math.rs — numerical verification against the Python reference (generates logits/probs for comparison)
  - convert.rs — small wrapper utility to call the Python converter

---

Quickstart (convert a model and run examples)

Prerequisites
- Rust toolchain (stable, e.g. rustup + cargo)
- Python 3.8+ and PyTorch (for conversion)
- Optional: CUDA-enabled GPU with matching CUDA/cuDNN + compatible torch build
- Recommended: make a working virtualenv for the Python conversion step

1) Convert a PyTorch .pth model to safetensors
The project uses a "prepared" model directory: modeldir/prepared/model.safetensors and config.json

Example:
```bash
# from repo root
python convert_pth_direct.py \
  --src /path/to/rwkv7-g1-0.1b-20250307-ctx4096.pth \
  --dest modeldir/prepared/model.safetensors \
  --config modeldir/prepared/config.json
```

Notes:
- The converter squeezes and converts tensors to float32 and applies the specific transpositions expected by the Rust runtime (LoRA transposition behaviour is handled inside convert_pth_direct.py).
- If args.config is passed, the converter will emit a config.json with the model hyperparameters.

2) Build the Rust tooling
```bash
cargo build --release
# or for development
cargo build
```

3) Run the interactive chat example
```bash
cargo run --example chat -- \
  --model-pth modeldir/rwkv7-g1-0.1b-20250307-ctx4096.pth \
  --vocab modeldir/rwkv_vocab_v20230424.json \
  --cpu
```
Replace `--cpu` with GPU by omitting `--cpu` and ensuring CUDA device 0 is available; the code picks `Device::new_cuda(0)` by default.

CLI flags (examples/chat.rs):
- --model-pth (path to original .pth; the example will call the converter if prepared files are missing)
- --vocab (vocabulary .json)
- --cpu (force CPU)
- --prompt, --temperature, etc.

4) Convert via the provided Rust wrapper (convert example)
The examples/convert.rs program is a thin wrapper that invokes the Python converter and reports output size and config creation.

---

Examples and verification

verify_math.rs
- Purpose: run a suite of prompts, compute final logits and probabilities with the Rust model, and save them for bitwise / numeric comparison with the Python reference.
- Output:
  - rust_final_logits.bin (binary f32)
  - rust_final_probs.bin (binary f32)
  - rust_final_logits.json (readable JSON summarizing logits/probs/context/tokens/metrics)
- The example can optionally run the Python reference for side-by-side comparison (it spawns the python reference script). It computes stable softmax in double precision to minimize numeric differences.

compare_with_reference.py
- Included helper that computes L1/L2/Linf, KL divergences, top-k matches between Rust and reference logits/probs.
- Use this to establish and audit numerical parity. The project uses the verification approach and thresholds similar to the official RWKV reference.

---

Important implementation notes (read carefully)
- Tokenizer: Implemented in src/models/rwkv7.rs; it implements encode/decode and byte-level behavior consistent with the reference.
- Device handling: Examples create a small CPU-only model for exact numeric checks if running on GPU to avoid non-deterministic GPU kernels when comparing to reference.
- Data types: Model runtime uses f32 for inference. The stable softmax helper used in verification casts to f64 for numeric stability then back to f32.
- convert_pth_direct.py:
  - Uses torch.load(..., weights_only=True) when available. Older torch versions do not support the weights_only kwarg — see Troubleshooting.
  - The script performs name conversions and tensor transposition to match the naming scheme in the Rust model loader.
- LoRA or other parameter modalities: the converter contains logic to transpose LoRA weights; treat LoRA-specific tensors carefully when verifying.

---

Development environment & building
- Rust: use a modern stable toolchain. If you use nightly features, update README accordingly (no nightly required currently).
- Run unit tests if present: (this repo may not include a test harness; run examples for practical tests)
- Formatting / linting: cargo fmt, clippy:
```bash
cargo fmt
cargo clippy -- -D warnings
```

---

Troubleshooting & common issues
- Python/Torch errors during conversion:
  - If torch.load fails due to unknown argument weights_only, update PyTorch to a recent version, or edit convert_pth_direct.py to remove weights_only usage and load the full checkpoint (be mindful of RAM).
- Missing model.safetensors/config.json:
  - Examples will attempt conversion automatically if the prepared files are missing. If conversion fails, check Python error messages and ensure the .pth file is valid.
- Numeric mismatch between Rust and reference:
  - Ensure the converter produced the expected transpositions/squeezes.
  - If comparing GPU results to Python CPU reference, prefer generating CPU rust model (verify_math example does this by building a CPU model when comparing).
- Tokenization mismatch:
  - Tokenizer implementation is in src/models/rwkv7.rs. If you see decode/encode differences, inspect tokenization edge-cases (byte-level encoding, special tokens).

---

Contributing & PR guidance
If you want me to open a pull request that adds this README:
- Suggested branch name: add/readme
- Commit message: "docs: add comprehensive README for candle_rwkv7"
- PR title: "Add comprehensive README"
- PR body (suggested):
  - Summarize purpose of the README.
  - Explain how README was generated (manual review of code + examples).
  - Call out areas that may need updates (model hyperparameters in convert script, future docs for LoRA).

I have prepared the README content above and can create the PR for you using the repository's coding agent. Tell me to proceed and I will open the PR with the suggested branch, commit message, and PR body.

---

License & credits
- Check repository root for a LICENSE file. This README does not change licensing — follow the repo's existing license.
- Major attribution: RWKV authors and the Python reference code leveraged for parity tests; Candle and candle_nn libraries.

---

If you'd like, I can:
- Open the PR with the README, or
- Make small edits to README style/structure (shorter or more tutorial-like), or
- Also add a brief CONTRIBUTING.md and CODE_OF_CONDUCT.md in the same PR.

Tell me which option you prefer and I will proceed to open the PR.
