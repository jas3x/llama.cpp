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

## Invention #1: Cache-Aware Top-K Tie-Breaking (`LLAMA_MOE_CACHE_BIAS`, `v1.1`)
* **Mechanism (`src/llama-graph.cpp` & `src/llama-moe-cache.cpp`):** Maintains a lightweight device-resident `F32 [n_expert]` bias tensor (`resident_bias`, 512 bytes/layer) on `CUDA0` updated asynchronously alongside `slot_map`. During low-batch decode (`n_tokens <= 8`), adds an epsilon tie-breaking bias (`0.025f` default) exclusively to `selection_probs` before `ggml_argsort_top_k`, while keeping the expert mixing weights `probs` completely unperturbed.
* **Measured Impact (`--temp 0` Dijkstra benchmark, `3800 MiB` VRAM cache):**
  * `LLAMA_MOE_CACHE_BIAS=0.000`: `23.2 t/s`
  * `LLAMA_MOE_CACHE_BIAS=0.015`: `24.5 t/s`
  * `LLAMA_MOE_CACHE_BIAS=0.030`: **`25.3 t/s`** (+62.3% over `15.59 t/s` upstream baseline)

## Invention #2: Cache-Aware Tail-Expert Pruning (`LLAMA_MOE_PRUNE_TAIL`, `v1.3`)
* **Mechanism (`src/llama-moe-cache.cpp`):** During single-token decode (`n_tokens == 1`) once the VRAM cache is warm, checks the lowest-confidence tail experts (ranks 6..7 of 0..7). If a tail expert is already resident in VRAM, it executes normally; if it is a cold cache miss (`slot >= n_slots`), it aliases to rank 0 to eliminate PCIe upload stalls and DDR5 CPU fallbacks for `<5%` tail weights while keeping all 6 primary experts and recurrent states intact.
* **Measured Impact (`--temp 0` Dijkstra benchmark, `3800 MiB` pre-warmed VRAM cache):**
  * `LLAMA_MOE_PRUNE_TAIL=0`: `52.4 t/s` prompt / `26.0 t/s` generation
  * `LLAMA_MOE_PRUNE_TAIL=2`: **`53.3 t/s` prompt / `27.9 t/s` generation** (+79.0% over `15.59 t/s` upstream baseline)

## Optional Model Recipe: Asymmetric Quantization Tiering (`29.4 t/s` TG / `59.0 t/s` PP)
To reproduce the **29.4 t/s** asymmetric hybrid GGUF locally from the standard `Inkling-2-Mini-Uncensored-AD-IQ3_S.gguf` (preserving 100% of the 3x-looped core `blk.6..17` and all attention projections at original precision while quantizing only the single-pass outer MoE experts to `Q2_K`):

    ./build/bin/llama-quantize \
      --allow-requantize \
      --output-tensor-type q6_K \
      --token-embedding-type q6_K \
      --tensor-type "attn_.*=q8_0" \
      --tensor-type "blk\.(6|7|8|9|10|11|12|13|14|15|16|17)\.ffn_gate_up_exps=iq3_s" \
      --tensor-type "blk\.(6|7|8|9|10|11|12|13|14|15|16|17)\.ffn_down_exps=iq4_nl" \
      Inkling-2-Mini-Uncensored-AD-IQ3_S.gguf \
      Inkling-2-Mini-Uncensored-Asym-IQ3S-Q2K.gguf \
      Q2_K 16
