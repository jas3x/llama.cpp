# Loop-Aware Physical-Layer MoE Expert Cache (`exp-shared-expert-cache`)

## Overview
Looped Mixture-of-Experts architectures such as **Inkling-2-Mini (`inkling2_loop`)** contain 24 physical layers where the middle 12 layers (`layers 6..17`) execute **3 times per token** (48 total layer visits per token, routing 8 of 128 experts per pass).

In upstream `llama.cpp` (`src/llama-moe-cache.cpp`), `llama_moe_cache` indexed cache entries by **virtual layer index (`0..47`)** rather than **canonical physical weight tensor (`0..23`)**, causing three bottlenecks on VRAM-constrained GPUs (such as an 8 GB RTX 2070):
1. **3x Cache Fragmentation & Overwritten Tensor Bindings:** Virtual passes `6..17`, `24..35`, and `36..47` pointed to the same underlying `ggml_tensor *` weights (`ffn_gate_up_exps`, `ffn_down_exps`), yet `bindings[w]` was overwritten three times while `slot_map` and `moe_cache_lru` tracked 48 separate virtual layers—forcing identical physical experts to be re-uploaded over PCIe up to 3 times per token.
2. **Inflated Host-Byte Accounting:** `host_bytes` summed all 48 virtual layers (~22.1 GiB) instead of the 24 physical layers (~11.0 GiB).
3. **Uniform Eviction Priority:** Single-pass outer layers (`0..5`, `18..23`) evicted 3x-visited loop-core experts (`6..17`) at equal weight.

## Key Modifications (`src/llama-moe-cache.cpp`)
* **Physical Tensor Deduplication (`phys_owner` map):** Groups virtual loop layers by their underlying physical expert `ggml_tensor *` pointer, counting `loop_visits` (3 for layers `6..17`, 1 for outer layers) and aliasing virtual loop slots (`24..47`) directly to the canonical physical `slot_map`, `slot_map_cpu`, and `cpu_map`.
* **Loop-Core Priority Weighting (`loop_weight` in `moe_cache_lru`):** Scales CRF frequency increments (`use()`) and victim survival scores (`victim()`) by `loop_visits` (`3.0x` for looped-core layers), keeping 3x-visited experts pinned in high-bandwidth GDDR6 VRAM.

## Benchmark Results (RTX 2070 8 GB + i7-13700K + 32 GB DDR5-5600)
Model: `Inkling-2-Mini-Uncensored-AD-IQ3_S.gguf` (26.07B params, 12.41 GiB, `n_ctx=4096`, `Q8_0` KV cache, `--no-op-offload`, `--moe-cache-mib 4150`):

| Build | CPU Miss Share | Prompt Processing (`t/s`) | Token Generation (`t/s`) | Gain vs. Stock Cache |
| :--- | :---: | :---: | :---: | :--- |
| Upstream `llama.cpp` (`244640b7f`, no MoE cache) | 1.00 (CPU only) | `80.03 t/s` (`pp256`) | `15.59 t/s` (`tg128`) | Baseline (no VRAM cache) |
| Upstream `llama.cpp` (`--moe-cache-mib 4150`) | 0.50 | `24.6 t/s` | `17.1 t/s` | Fragmented 48-layer cache |
| **Patched `exp-shared-expert-cache`** | **0.25** | **`35.6 t/s`** | **`23.1 t/s`** | **+44.7% PP / +35.1% TG** |
| **Patched `exp-shared-expert-cache`** | **0.50** | **`40.8 t/s`** | **`22.7 t/s`** | **+65.9% PP / +32.7% TG** |

## Build & Run Commands

    cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_CURL=ON
    cmake --build build --config Release -j $(nproc)

    ./build/bin/llama-server \
      -m ~/models/Inkling-2-Mini-Uncensored-AD-IQ3_S.gguf \
      --alias "inkling-2-mini-uncensored" \
      --host 127.0.0.1 --port 8080 --cors-origins localhost \
      -np 1 -t 8 -cmoe -ngl 99 --no-op-offload --load-mode none \
      --moe-cache-mib 3800 --moe-cache-cpu-share 0.35 \
      -c 16384 -b 512 -ub 256 -fa on -ctk q8_0 -ctv q8_0
