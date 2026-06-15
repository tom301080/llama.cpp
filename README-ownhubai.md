# ownHUBAI fork of llama.cpp

This is ownHUBAI's downstream fork of [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp).
It exists to carry a **small, flag-gated, upstream-first patch stack** for serving
ownHUB's local nets on Apple Silicon (and, via CUDA/Vulkan, Windows/Linux).

Strategy and rules: see `ownHUBAI/docs/adr/0004-fork-maintenance-and-upstream-strategy.md`
and `ownHUBAI/docs/adr/0002-serving-runtime-strategy.md`.

## Invariant

**A default build is byte-for-byte stock llama.cpp behaviour.** Every ownHUBAI
feature is behind a CMake option (off by default) and/or a runtime flag with an
upstream fallback. In particular, **classic native MTP (`--spec-type mtp`) is left
untouched** so all other environments keep working exactly as upstream intends.

## Branches

- `master` — mirror of upstream (tracks tags; do not commit here).
- `ownhubai/release` — `UPSTREAM_BASE` tag + the rebased patch stack. **This is
  what ownHUB pins.**
- `feat/*` — one feature per branch, each kept upstreamable.

## Upstream base

Pinned in `UPSTREAM_BASE` (currently tag `b9656`, commit `581e8eca8b41`).

## Patch stack

Tracked in `PATCHES.md` — each patch with its flag, the files it touches, and its
upstream-PR status.

## Remotes (for maintainers)

- `origin`   → `tom301080/llama.cpp` (this fork)
- `upstream` → `ggml-org/llama.cpp`
- `atomic`   → `AtomicBot-ai/atomic-llama-cpp-turboquant` (MTP-overlap *reference*)
- `thetom`   → `TheTom/llama-cpp-turboquant` (TurboQuant-KV *reference*)

The `atomic`/`thetom` remotes are **reference designs to mine**, not branches to
merge — both are 470–850 commits behind upstream (see ADR 0004).
