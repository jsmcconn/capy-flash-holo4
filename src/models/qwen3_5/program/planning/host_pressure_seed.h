#pragma once
#include <algorithm>
#include <cstddef>
#include <optional>

namespace ninfer::models::qwen3_5::detail {
// The caller orders eligible whole-owner evictions by policy value, excludes
// protected sources and any all-owner target, and checks joint physical feasibility.
// Every returned upper bound has passed the joint proof. If an allocator's
// placement is non-monotone this heuristic may miss a smaller feasible prefix.
// Budget expiry during refinement keeps the already proved upper bound.
template <class Feasible, class Continue>
[[nodiscard]] std::optional<std::size_t>
host_eviction_prefix(std::size_t maximum, Feasible&& feasible, Continue&& can_probe) {
    if (maximum == 0) return std::nullopt;
    std::size_t lower = 0, upper = 1;
    for (;;) {
        if (!can_probe()) return std::nullopt;
        if (feasible(upper)) break;
        lower = upper;
        if (upper == maximum) return std::nullopt;
        upper += std::min(upper, maximum - upper);
    }
    while (upper - lower > 1) {
        if (!can_probe()) return upper;
        const auto middle = lower + (upper - lower) / 2;
        if (feasible(middle))
            upper = middle;
        else
            lower = middle;
    }
    return upper;
}
} // namespace ninfer::models::qwen3_5::detail
