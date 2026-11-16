# NN4motion

Rust Transformer for **NN4motion**: predict continuous **target position**, **target velocity**, and **target current**, conditioned on any non-empty subset of those same channels.

## ML stack

**[Candle](https://github.com/huggingface/candle)** (`candle-core` + `candle-nn`) — pure-Rust tensors, modules, and **AdamW**. Chosen for a small dependency footprint and first-class CPU smoke runs without a Python bridge.

## Spec (v0)

| Knob | Value |
|------|-------|
| Layers | **1** |
| Context length | **1024** |
| Attention heads | **8** |
| FFN activation | **ReLU** |
| Optimizer | **AdamW** |
| Outputs | position, velocity, current (3 continuous) |

Default width: `d_model = 64`, `d_ff = 256` (overridable in code).

## Prompt / masking scheme

Each timestep is encoded as **6 floats**:

```text
[v_pos * m_pos, v_vel * m_vel, v_cur * m_cur, m_pos, m_vel, m_cur]
```

- `m_* ∈ {0,1}` — channel present in the prompt  
- Masked values are zeroed **and** the mask bit is `0`, so “missing” ≠ “present zero”  
- Any non-empty subset of `{position, velocity, current}` is valid  
- The model **always** regresses all three targets

## Layout

```text
crates/nn4motion/src/
  config.rs          # CONTEXT_LEN, heads, AdamW defaults
  data/encoding.rs   # prompt masks + synthetic batches
  model/
    attention.rs     # multi-head self-attention
    ffn.rs           # ReLU FFN
    layer.rs         # encoder block
    transformer.rs   # full model + output head
  train.rs           # AdamW loop
  main.rs            # CLI: info / smoke / train
```

## Quick start

```bash
# from repo root
cargo build -p nn4motion
cargo test  -p nn4motion

# architecture summary
cargo run -p nn4motion -- info

# tiny CPU smoke train (synthetic data)
cargo run -p nn4motion -- smoke --steps 5 --seq-len 16 --batch-size 2
```

Full context (1024) is supported by the model; smoke uses a short `seq_len` for speed. Cap remains `seq_len ≤ 1024`:

```bash
cargo run -p nn4motion -- train --steps 20 --seq-len 64 --batch-size 4
```

## License

MIT
