#pragma once

#include "runtime/contract/resources.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::runtime {

struct ContextPortfolioOwnerPolicy {
    PlanningOwnerId owner;
    std::uint32_t private_retention_weight = 0;
    bool explicit_shared_credit            = false;
};

struct ContextPortfolioCheckpointValue {
    PlanningOwnerId owner;
    std::uint32_t demand_mask          = 0;
    std::uint64_t rebuild_ns           = 0;
    std::uint64_t baseline_recovery_ns = 0;
    std::uint64_t target_recovery_ns   = 0;
};

// Common retention value for bounded preserving-cover construction. Callers keep
// their established tie-breaks (materialization has hit history; capture uses owner ID).
template <class CheckpointPolicy>
[[nodiscard]] std::uint64_t
pressure_seed_owner_value(PlanningOwnerId owner, std::uint32_t retention_weight,
                          bool explicit_credit,
                          std::span<const CheckpointPolicy> checkpoints) noexcept {
    std::uint64_t saving = 0;
    std::uint32_t mask   = 0;
    for (const auto& checkpoint : checkpoints) {
        if (checkpoint.owner != owner) { continue; }
        saving = std::max(saving, checkpoint.rebuild_ns > checkpoint.baseline_recovery_ns
                                      ? checkpoint.rebuild_ns - checkpoint.baseline_recovery_ns
                                      : 0);
        mask |= checkpoint.demand_mask;
    }
    const auto weight =
        std::uint64_t(retention_weight) + std::popcount(mask) + unsigned(explicit_credit);
    return weight != 0 && saving > std::numeric_limits<std::uint64_t>::max() / weight
               ? std::numeric_limits<std::uint64_t>::max()
               : saving * weight;
}

struct ContextPortfolioValueResult {
    std::uint64_t baseline_public_value   = 0;
    std::uint64_t target_public_value     = 0;
    std::uint64_t private_transition_loss = 0;
    bool saturated                        = false;
};

// Folds one complete inactive checkpoint portfolio. Empirical demands and explicit shared credit
// form public portfolio values. A retention-weighted owner contributes the largest loss of one
// concrete checkpoint capability, so a surviving later checkpoint cannot mask earlier damage.
// Runtime also uses this legacy private-loss field for a dormant shared owner's one-use prior.
class ContextPortfolioValue {
public:
    ContextPortfolioValue() { owner_scratch_.reserve(32U); }

    [[nodiscard]] ContextPortfolioValueResult
    fold(std::span<const ContextPortfolioOwnerPolicy> owners,
         std::span<const ContextPortfolioCheckpointValue> checkpoints) {
        std::array<std::uint64_t, 32> baseline_demand{};
        std::array<std::uint64_t, 32> target_demand{};
        owner_scratch_.clear();
        bool dense_prefix = true;
        for (const ContextPortfolioOwnerPolicy& policy : owners) {
            // Runtime owner IDs are sequential. Keep sparse/opaque IDs supported without
            // paying a quadratic duplicate check for every complete portfolio fold.
            dense_prefix = dense_prefix && policy.owner.value == owner_scratch_.size();
            if (!dense_prefix &&
                std::find_if(owner_scratch_.begin(), owner_scratch_.end(), [&](const auto& item) {
                    return item.owner == policy.owner;
                }) != owner_scratch_.end()) {
                throw std::logic_error("portfolio owner policy ID is duplicated");
            }
            owner_scratch_.push_back(
                OwnerValue{.owner                    = policy.owner,
                           .private_retention_weight = policy.private_retention_weight,
                           .explicit_shared_credit   = policy.explicit_shared_credit});
        }

        for (const ContextPortfolioCheckpointValue& checkpoint : checkpoints) {
            auto owner = owner_scratch_.end();
            if (checkpoint.owner.value < owner_scratch_.size() &&
                owner_scratch_[checkpoint.owner.value].owner == checkpoint.owner) {
                owner = owner_scratch_.begin() + checkpoint.owner.value;
            } else {
                owner = std::find_if(
                    owner_scratch_.begin(), owner_scratch_.end(),
                    [&](const OwnerValue& item) { return item.owner == checkpoint.owner; });
            }
            if (owner == owner_scratch_.end()) {
                throw std::logic_error("portfolio checkpoint has no owner policy");
            }
            const std::uint64_t baseline_saving =
                checkpoint.rebuild_ns > checkpoint.baseline_recovery_ns
                    ? checkpoint.rebuild_ns - checkpoint.baseline_recovery_ns
                    : 0;
            const std::uint64_t target_saving =
                checkpoint.rebuild_ns > checkpoint.target_recovery_ns
                    ? checkpoint.rebuild_ns - checkpoint.target_recovery_ns
                    : 0;
            owner->baseline_best = std::max(owner->baseline_best, baseline_saving);
            owner->target_best   = std::max(owner->target_best, target_saving);
            owner->private_transition_loss =
                std::max(owner->private_transition_loss,
                         baseline_saving > target_saving ? baseline_saving - target_saving : 0);
            for (auto mask = checkpoint.demand_mask; mask != 0; mask &= mask - 1U) {
                const auto bit       = std::countr_zero(mask);
                baseline_demand[bit] = std::max(baseline_demand[bit], baseline_saving);
                target_demand[bit]   = std::max(target_demand[bit], target_saving);
            }
        }

        ContextPortfolioValueResult result;
        for (std::size_t bit = 0; bit < baseline_demand.size(); ++bit) {
            add(result.baseline_public_value, baseline_demand[bit], result.saturated);
            add(result.target_public_value, target_demand[bit], result.saturated);
        }
        for (const OwnerValue& owner : owner_scratch_) {
            if (owner.private_retention_weight != 0) {
                add_weighted(result.private_transition_loss, owner.private_transition_loss,
                             owner.private_retention_weight, result.saturated);
            }
            if (owner.explicit_shared_credit) {
                add(result.baseline_public_value, owner.baseline_best, result.saturated);
                add(result.target_public_value, owner.target_best, result.saturated);
            }
        }
        return result;
    }

private:
    struct OwnerValue {
        PlanningOwnerId owner;
        std::uint32_t private_retention_weight = 0;
        bool explicit_shared_credit            = false;
        std::uint64_t baseline_best            = 0;
        std::uint64_t target_best              = 0;
        std::uint64_t private_transition_loss  = 0;
    };

    static void add(std::uint64_t& value, std::uint64_t increment, bool& saturated) noexcept {
        if (increment > std::numeric_limits<std::uint64_t>::max() - value) {
            value     = std::numeric_limits<std::uint64_t>::max();
            saturated = true;
        } else {
            value += increment;
        }
    }

    static void add_weighted(std::uint64_t& value, std::uint64_t increment, std::uint32_t weight,
                             bool& saturated) noexcept {
        if (weight != 0 && increment > std::numeric_limits<std::uint64_t>::max() / weight) {
            add(value, std::numeric_limits<std::uint64_t>::max(), saturated);
            saturated = true;
            return;
        }
        add(value, increment * weight, saturated);
    }

    std::vector<OwnerValue> owner_scratch_;
};

} // namespace ninfer::runtime
