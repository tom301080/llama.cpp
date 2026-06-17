#pragma once
// ownHUBAI: MoE expert-activation telemetry (routing-skew validation for the
// expert-offload cache; ADR 0005 / docs/moe-expert-offload-impl.md).
// Header-only. Activated via --moe-expert-stats (wired in common_init_from_params).
// Taps the per-layer "ffn_moe_topk-<il>" tensors (the routed expert IDs) through
// the ggml eval callback and accumulates a per-(layer,expert) activation count.

#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h" // ownHUBAI: llama_ownhub_moe_active / _advise (expert-offload pager)

#include <unordered_map>
#include <vector>
#include <algorithm>
#include <functional>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>

struct common_moe_stats {
    // key = ((uint64_t)layer << 32) | (uint32_t)expert  ->  activation count
    std::unordered_map<uint64_t, uint64_t> counts;
    uint64_t total = 0;  // total expert activations counted
    uint64_t calls = 0;  // number of ffn_moe_topk tensors observed
};

// ggml_backend_sched_eval_callback: bool(ggml_tensor * t, bool ask, void * ud)
//   ask==true : return true to observe this node (data ready in the ask==false call)
//   ask==false: node computed; return true to CONTINUE (false would ABORT the graph)
inline bool common_moe_stats_eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    const bool is_topk = std::strncmp(t->name, "ffn_moe_topk", 12) == 0;
    if (ask) {
        return is_topk; // only observe the routed-expert-id tensors
    }
    if (is_topk && t->type == GGML_TYPE_I32) {
        const int64_t n = ggml_nelements(t);
        if (n > 0) {
            std::vector<int32_t> ids((size_t) n);
            ggml_backend_tensor_get(t, ids.data(), 0, (size_t) n * sizeof(int32_t));
            int il = 0;
            const char * dash = std::strrchr(t->name, '-');
            if (dash) { il = atoi(dash + 1); }
            // ownHUBAI: prefetch — WILLNEED the just-used experts (temporal
            // locality: likely reused next token). No-op unless the engine pager
            // is active (OWNHUB_MOE_PREFETCH + mmap-offloaded experts). inc3: tick
            // the pager once per forward (layer 0) so it can sweep the cold tail.
            if (llama_ownhub_moe_active()) {
                if (il == 0) {
                    llama_ownhub_moe_step();
                }
                llama_ownhub_moe_advise(il, ids.data(), (int) n, /*willneed=*/1);
            }
            // telemetry accumulation (only when --moe-expert-stats gave a sink)
            if (auto * st = (common_moe_stats *) user_data) {
                for (int64_t i = 0; i < n; ++i) {
                    const int32_t e = ids[(size_t) i];
                    if (e < 0) { continue; }
                    st->counts[((uint64_t) (uint32_t) il << 32) | (uint32_t) e]++;
                    st->total++;
                }
                st->calls++;
            }
        }
    }
    return true; // always continue execution
}

inline void common_moe_stats_print(const common_moe_stats & st) {
    if (st.total == 0) {
        fprintf(stderr, "\n[ownHUBAI moe-stats] no expert activations captured "
                        "(model not MoE, or no decode happened)\n");
        return;
    }
    // regroup per layer
    std::unordered_map<uint32_t, std::unordered_map<uint32_t, uint64_t>> per_layer;
    for (const auto & kv : st.counts) {
        per_layer[(uint32_t) (kv.first >> 32)][(uint32_t) (kv.first & 0xffffffffu)] += kv.second;
    }
    // headline: average over layers of "fraction of experts covering 80% of that
    // layer's activations" — the smaller, the more a hot-expert cache helps.
    double sum_frac80 = 0.0;
    int    nlayers    = 0;
    size_t max_experts = 0;
    for (const auto & lp : per_layer) {
        std::vector<uint64_t> v;
        uint64_t lt = 0;
        for (const auto & ep : lp.second) { v.push_back(ep.second); lt += ep.second; }
        std::sort(v.begin(), v.end(), std::greater<uint64_t>());
        max_experts = std::max(max_experts, v.size());
        uint64_t acc = 0;
        size_t   need = 0;
        for (; need < v.size(); ++need) {
            acc += v[need];
            if (acc * 100 >= lt * 80) { ++need; break; }
        }
        sum_frac80 += v.empty() ? 0.0 : (double) need / (double) v.size();
        nlayers++;
    }
    fprintf(stderr, "\n========== ownHUBAI MoE expert-activation telemetry ==========\n");
    fprintf(stderr, "layers observed : %d\n", nlayers);
    fprintf(stderr, "distinct experts: up to %zu per layer\n", max_experts);
    fprintf(stderr, "total activations: %llu  (topk tensors: %llu)\n",
            (unsigned long long) st.total, (unsigned long long) st.calls);
    if (nlayers > 0) {
        fprintf(stderr, "avg %% of experts needed to cover 80%% of a layer's tokens: %.1f%%\n",
                100.0 * sum_frac80 / nlayers);
        fprintf(stderr, "  -> smaller = more skew = a hot-expert cache helps more.\n");
    }
    fprintf(stderr, "==============================================================\n");
    // ownHUBAI: raw per-(layer,expert) dump for offline B0-vs-B1 hit-rate analysis
    if (FILE * f = fopen("ownhub_moe_usage.csv", "w")) {
        fprintf(f, "layer,expert,count\n");
        for (const auto & kv : st.counts) {
            fprintf(f, "%u,%u,%llu\n",
                    (uint32_t) (kv.first >> 32), (uint32_t) (kv.first & 0xffffffffu),
                    (unsigned long long) kv.second);
        }
        fclose(f);
        fprintf(stderr, "[ownHUBAI moe-stats] raw per-(layer,expert) counts -> ownhub_moe_usage.csv\n");
    }
}
