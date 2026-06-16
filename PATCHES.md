# ownHUBAI patch stack

Each ownHUBAI feature carried on top of the `UPSTREAM_BASE` tag. Default build =
stock; every patch is opt-in. Keep each patch isolated and upstreamable
(see `ownHUBAI/docs/adr/0004-...`).

| # | Patch | Flag (compile / runtime) | Status | Files touched | Upstream PR |
|---|---|---|---|---|---|
| 1 | MTP/NextN overlap (Atomic-style: draft+verify overlap, in-graph argmax) | `-DOWNHUB_MTP_OVERLAP=ON` / `--mtp-head` | **planned** | `common/speculative.*`, `src/llama-context.*`, `src/models/qwen35moe-nextn.cpp`, `tools/server/server-context.cpp`, Metal kernels | tbd (target: generic Metal-sync fix → #23752) |
| 2 | TurboQuant-KV (WHT-rotated low-bit KV) | `-DOWNHUB_TURBOQUANT=ON` / `-ctk turbo3` | later | new cache types + backend kernels | none (upstream #21089 closed) |
| 3 | MoE expert-offload cache (run 35B-A3B in 16GB) | flags `--moe-expert-cache-size/-policy/-stats/--moe-prefetch`; env `OWNHUB_MOE_MADV_RANDOM` / `OWNHUB_MOE_PREFETCH` | **Strat A code-complete** | flags (`common/arg.cpp`,`common.h`); telemetry (`common/moe-stats.h`, tap `ffn_moe_topk`); Strat A complete: inc1 MADV_RANDOM + inc2 engine pager (`src/llama-model.cpp`,`include/llama.h`) + routing WILLNEED driver (`common/moe-stats.h`,`common.cpp`) + inc3 recency + cold-tail DONTNEED (`llama_ownhub_moe_step`, env `OWNHUB_MOE_EVICT_EVERY/AGE`); **inc3c frequency hot-set protection** (env `OWNHUB_MOE_HOTSET_K`, 0=off): per-expert use_count, protect top-K hottest/layer from DONTNEED + re-WILLNEED them (exploits routing skew). Default==stock. **Deferred:** inc3b MTP-draft look-ahead (accuracy-limited: per-layer data-dependent routing); Strat B slot cache (GPU OOMs on 16GB) | aligns with upstream #20757; validate on 16GB HW |

## Invariants

- **Classic native MTP (`--spec-type mtp`) is never modified** — other
  environments keep working as upstream intends. The overlap path is an *added*
  flag, not a replacement.
- A default `cmake` build enables none of these.

## Migration note (patch 1)

The Atomic/TheTom forks are 470–850 commits behind upstream, so their changes do
NOT cherry-pick cleanly onto `b9656` (upstream rebuilt the speculative API in
#22838). Patch 1 is a **re-implementation** against the current API, using
Atomic's `MTP.md` / `NEXTN.md` + code as the reference design. Port order:
additive first (new `*-nextn.cpp` model files, NextN GGUF export in
`convert_hf_to_gguf.py` + `gguf-py`, the `test-speculative-mtp` test), then the
deep integration (overlap pipeline in `llama-context.cpp` / `speculative.cpp`).
