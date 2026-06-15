# ownHUBAI patch stack

Each ownHUBAI feature carried on top of the `UPSTREAM_BASE` tag. Default build =
stock; every patch is opt-in. Keep each patch isolated and upstreamable
(see `ownHUBAI/docs/adr/0004-...`).

| # | Patch | Flag (compile / runtime) | Status | Files touched | Upstream PR |
|---|---|---|---|---|---|
| 1 | MTP/NextN overlap (Atomic-style: draft+verify overlap, in-graph argmax) | `-DOWNHUB_MTP_OVERLAP=ON` / `--mtp-head` | **planned** | `common/speculative.*`, `src/llama-context.*`, `src/models/qwen35moe-nextn.cpp`, `tools/server/server-context.cpp`, Metal kernels | tbd (target: generic Metal-sync fix → #23752) |
| 2 | TurboQuant-KV (WHT-rotated low-bit KV) | `-DOWNHUB_TURBOQUANT=ON` / `-ctk turbo3` | later | new cache types + backend kernels | none (upstream #21089 closed) |
| 3 | SD-aware expert prefetch (offload moat) | `-DOWNHUB_SD_PREFETCH=ON` / `--sd-prefetch` | later | expert offload path | tbd |

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
