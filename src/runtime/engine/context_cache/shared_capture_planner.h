#pragma once

#include "ninfer/types.h"
#include "runtime/contract/resources.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "runtime/engine/context_cache/context_portfolio_value.h"
#include "runtime/engine/context_cache/materialization_budget.h"
#include "runtime/engine/context_cache/resource_search.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace ninfer::runtime {


// One optional publication search, shared across all replacement scenarios and both passes.
// Regret is an optimistic bound on the value still available, not a prediction of feasibility.
enum class CaptureSearchStop { None, Economic, Time, Targets, Canonical, Cancelled };

struct CaptureSearchControl {
    PlanningAllowance allowance;
    MaterializationSearchBudget wall;
    std::uint32_t targets_remaining      = 4096;
    std::uint32_t targets_evaluated      = 0;
    std::uint32_t improvements           = 0;
    std::uint64_t incumbent_gain         = 0;
    std::uint64_t assessment_estimate_ns = 1;
    bool stopped                         = false;
    CaptureSearchStop stop               = CaptureSearchStop::None;
    std::uint64_t search_started_ns      = 0;
    bool budget_started                  = false;
    std::uint64_t initial_upper_bound    = 0;

    CaptureSearchControl(PlanningAllowance boundary, std::uint64_t upper_bound,
                         std::uint64_t started = planning_now_ns())
        : allowance(boundary), wall(boundary, started, upper_bound), search_started_ns(started),
          initial_upper_bound(upper_bound) {}

    [[nodiscard]] bool allow(std::uint64_t upper_bound, std::uint64_t now = planning_now_ns()) {
        if (allowance.cancellation && allowance.cancellation->load(std::memory_order_relaxed)) {
            stopped = true;
            stop    = CaptureSearchStop::Cancelled;
            return false;
        }
        if (targets_remaining == 0) {
            stopped = true;
            stop    = CaptureSearchStop::Targets;
            return false;
        }
        // One identity is the discovery seed; its cost still counts against the whole boundary.
        if (targets_evaluated == 0) { return true; }
        if (!budget_started) {
            const auto initial_regret =
                initial_upper_bound > incumbent_gain ? initial_upper_bound - incumbent_gain : 0;
            wall = MaterializationSearchBudget(allowance, search_started_ns, initial_regret);
            budget_started = true;
        }
        const auto regret = upper_bound > incumbent_gain ? upper_bound - incumbent_gain : 0;
        if (!wall.allow(now, assessment_estimate_ns, assessment_estimate_ns, regret, false,
                        improvements)) {
            const bool time = wall.stop_reason() == MaterializationStopReason::TimeBudget;
            stop            = time ? CaptureSearchStop::Time : CaptureSearchStop::Economic;
            stopped         = time;
            return false;
        }
        stop = CaptureSearchStop::None;
        return true;
    }

    void assessed(std::uint64_t elapsed) {
        --targets_remaining;
        ++targets_evaluated;
        assessment_estimate_ns = std::max<std::uint64_t>(
            1, targets_evaluated == 1 ? elapsed : (assessment_estimate_ns + elapsed) / 2U);
    }

    void improve(std::uint64_t gain) {
        if (gain > incumbent_gain) {
            incumbent_gain = gain;
            ++improvements;
        }
    }
};

// Shared publication is optional. Unlike materialization, its incumbent is always Skip and a
// target is selectable only when the complete post-state has strictly positive net value.
template <class ModelContract, class SearchClock = std::chrono::steady_clock>
class SharedCapturePlanner {
public:
    using Program                = typename ModelContract::Program;
    using CaptureAssessment      = typename ModelContract::CaptureAssessment;
    using CapturePressurePlan    = typename ModelContract::CapturePressurePlan;
    using ContinuationHandle     = typename ModelContract::ContinuationHandle;
    using SharedPrefixHandle     = typename ModelContract::SharedPrefixHandle;
    using PressureTargetHandle   = typename ModelContract::PressureTargetHandle;
    using AssessedPressureTarget = typename ModelContract::AssessedPressureTarget;

    struct OwnerPolicy {
        PlanningOwnerId owner;
        std::uint32_t private_retention_weight = 0;
        bool explicit_shared_credit            = false;
    };

    struct CheckpointPolicy {
        PlanningOwnerId owner;
        CheckpointRef checkpoint;
        std::uint32_t demand_mask            = 0;
        std::uint64_t rebuild_ns             = 0;
        std::uint64_t baseline_recovery_ns   = 0;
        std::uint64_t optimistic_recovery_ns = 0;
    };

    struct Result;

    struct Input {
        const CaptureAssessment* capture = nullptr;
        std::span<const ContinuationHandle* const> private_owners;
        std::span<const PlanningOwnerId> private_owner_ids;
        std::span<const SharedPrefixHandle* const> shared_owners;
        std::span<const PlanningOwnerId> shared_owner_ids;
        std::span<const OwnerPolicy> owner_policies;
        std::span<const CheckpointPolicy> checkpoint_policies;
        std::optional<PlanningOwnerId> direct_shared_victim;
        std::uint32_t candidate_demand_mask         = 0;
        std::uint64_t candidate_rebuild_ns          = 0;
        std::uint64_t private_baseline_immediate_ns = 0;
        std::uint32_t blocked_runnable_requests     = 0;
        std::uint32_t stable_scenario_ordinal       = 0;
        std::uint32_t target_budget                 = kTargetBudget;
        // One feasible construction opportunity for the first competitive scenario.
        bool allow_deficit_seed = false;
        PlanningAllowance allowance;
        CaptureSearchControl* search = nullptr;
        bool prune                   = true;
        const Result* global_best    = nullptr;
        bool* complete               = nullptr;
    };

    struct Result {
        std::optional<CapturePressurePlan> pressure;
        std::vector<PressureOwnerOutcome> owner_outcomes;
        std::vector<PressureCheckpointOutcome> checkpoint_outcomes;
        std::uint64_t baseline_value          = 0;
        std::uint64_t target_value            = 0;
        std::uint64_t immediate_ns            = 0;
        std::uint64_t net_gain                = 0;
        std::uint32_t stable_scenario_ordinal = 0;
        std::uint32_t stable_target_ordinal   = 0;
        std::uint32_t targets_evaluated       = 0;
        std::uint32_t degradation_units       = 0;
        std::uint32_t dropped_checkpoints     = 0;
        std::uint64_t upper_bound             = 0;
    };

    // Survivor recovery and all transition costs are unsigned. Replacing each survivor's
    // recovery by zero, while keeping the direct victim and candidate fixed, can only increase
    // the portfolio's max/sum savings. Dropping private loss and immediate cost is optimistic.
    // This deliberately loose bound remains valid when releasing an alias makes recovery cheaper.
    [[nodiscard]] std::uint64_t upper_bound(const Input& input,
                                            const ContextMachineCostModel& cost) {
        const auto value = fold_target(input, PressureTargetAssessment{}, cost, true);
        return value.saturated ? std::numeric_limits<std::uint64_t>::max() : value.gain;
    }

    [[nodiscard]] static bool better(const Result& lhs, const Result& rhs) noexcept {
        return std::tuple{lhs.net_gain,
                          std::numeric_limits<std::uint32_t>::max() - lhs.degradation_units,
                          std::numeric_limits<std::uint32_t>::max() - lhs.dropped_checkpoints,
                          std::numeric_limits<std::uint32_t>::max() - lhs.stable_scenario_ordinal,
                          std::numeric_limits<std::uint32_t>::max() - lhs.stable_target_ordinal} >
               std::tuple{rhs.net_gain,
                          std::numeric_limits<std::uint32_t>::max() - rhs.degradation_units,
                          std::numeric_limits<std::uint32_t>::max() - rhs.dropped_checkpoints,
                          std::numeric_limits<std::uint32_t>::max() - rhs.stable_scenario_ordinal,
                          std::numeric_limits<std::uint32_t>::max() - rhs.stable_target_ordinal};
    }

    [[nodiscard]] static bool may_beat(std::uint64_t bound, std::uint32_t scenario,
                                       const Result& best) noexcept {
        if (bound != best.net_gain) { return bound > best.net_gain; }
        if (best.degradation_units || best.dropped_checkpoints) { return true; }
        // Identity is target zero. Every other target has a later stable ordinal.
        return scenario < best.stable_scenario_ordinal ||
               (scenario == best.stable_scenario_ordinal && best.stable_target_ordinal != 0);
    }

    SharedCapturePlanner() : target_ledger_(kTargetBudget + 1U) {
        queue_.reserve(kTargetBudget);
        impact_scratch_.reserve(64);
        owner_scratch_.reserve(32);
        checkpoint_scratch_.reserve(64);
    }

    [[nodiscard]] std::optional<Result>
    plan(Program& program, const ContextMachineCostModel& machine_cost, const Input& input) {
        validate(input);
        const std::uint64_t bound = upper_bound(input, machine_cost);
        if (input.complete) { *input.complete = false; }
        if (input.prune &&
            (bound == 0 || (input.global_best &&
                            !may_beat(bound, input.stable_scenario_ordinal, *input.global_best)))) {
            if (input.complete) { *input.complete = true; }
            return std::nullopt;
        }
        if (input.search && !input.search->allow(bound, planning_now_ns<SearchClock>())) {
            return std::nullopt;
        }
        queue_.clear();
        target_ledger_.reset(static_cast<std::size_t>(input.target_budget) + 1U);

        auto session = program.begin_capture_pressure_planning(
            *input.capture, input.private_owners, input.private_owner_ids, input.shared_owners,
            input.shared_owner_ids);
        const PlanningCandidateId candidate_id = session.candidate_id();

        const PressureTargetHandle identity         = session.identity_target();
        const PressureTargetGuidance identity_guide = session.guidance(identity);
        if (identity_guide.candidate != candidate_id) {
            throw std::logic_error("shared capture identity guidance changed candidate identity");
        }
        if (identity_guide.stable_target_ordinal != 0) {
            throw std::logic_error("shared capture identity must have the first target ordinal");
        }
        queue_.push_back(QueuedTarget{
            .target  = identity,
            .ordinal = identity_guide.stable_target_ordinal,
        });
        mark_target(identity_guide.stable_target_ordinal, kTargetDiscovered);
        std::optional<Incumbent> incumbent;
        // The identity target is already canonical in the Program session.  The search budget
        // bounds every canonical target admitted to this scenario, including children committed
        // now but still waiting in the breadth-first queue; assessed targets are only telemetry.
        std::uint32_t canonical_targets = 1;
        std::uint32_t targets_evaluated = 0;
        std::size_t cursor              = 0;
        bool incomplete                 = false;
        bool proven                     = false;

        while (cursor < queue_.size() && targets_evaluated < input.target_budget) {
            // Always inspect the identity seed; further work is optional and shares the
            // caller's capture boundary budget across every replacement scenario.
            if (targets_evaluated != 0 && !queue_[cursor].construction_seed &&
                (input.search ? !input.search->allow(bound, planning_now_ns<SearchClock>())
                              : input.allowance.remaining(planning_now_ns<SearchClock>()) == 0)) {
                incomplete = true;
                break;
            }
            const QueuedTarget queued = queue_[cursor++];
            if (target_marked(queued.ordinal, kTargetAssessed)) { continue; }
            const auto assessment_started   = planning_now_ns<SearchClock>();
            AssessedPressureTarget assessed = session.assess(queued.target);
            const auto assessment_elapsed   = planning_now_ns<SearchClock>() - assessment_started;
            if (input.search) { input.search->assessed(assessment_elapsed); }
            const PressureTargetAssessment& assessment = assessed.assessment();
            if (assessment.candidate != candidate_id ||
                assessment.stable_target_ordinal != queued.ordinal) {
                throw std::logic_error("shared capture target changed candidate identity");
            }
            mark_target(queued.ordinal, kTargetAssessed);
            ++targets_evaluated;

            if (assessment.physical_status == MaterializationPhysicalStatus::Feasible) {
                const TransitionValue value = fold_target(input, assessment, machine_cost);
                if (value.positive &&
                    (!incumbent || better(value, assessment, input, *incumbent))) {
                    if (input.search) { input.search->improve(value.gain); }
                    incumbent = Incumbent{
                        .target              = queued.target,
                        .assessed            = std::move(assessed),
                        .value               = value,
                        .stable_target       = assessment.stable_target_ordinal,
                        .degradation_units   = assessment.degradation_units,
                        .dropped_checkpoints = assessment.dropped_checkpoints,
                        .owner_outcomes      = std::vector<PressureOwnerOutcome>(
                            assessment.owner_outcomes.begin(), assessment.owner_outcomes.end()),
                        .checkpoint_outcomes =
                            [&] {
                                std::vector<PressureCheckpointOutcome> outcomes;
                                outcomes.reserve(assessment.checkpoint_impacts.size());
                                for (const PressureCheckpointRecoveryImpact& impact :
                                     assessment.checkpoint_impacts) {
                                    outcomes.push_back(PressureCheckpointOutcome{
                                        .owner      = impact.owner,
                                        .checkpoint = impact.checkpoint,
                                        .survives   = impact.survives,
                                    });
                                }
                                return outcomes;
                            }(),
                    };
                }
            }

            if (input.allow_deficit_seed && queued.ordinal == 0 && !incumbent &&
                assessment.physical_status != MaterializationPhysicalStatus::Feasible &&
                input.target_budget >= 2 &&
                (!input.search || input.search->targets_remaining != 0)) {
                // Like request materialization, first construct a feasible incumbent.
                // Optional economic search must not decide whether a joint cover is discovered.
                std::vector<std::pair<std::uint64_t, PlanningOwnerId>> priorities;
                const auto append = [&](std::span<const PlanningOwnerId> eligible) {
                    for (const auto owner : eligible) {
                        const auto policy =
                            std::find_if(input.owner_policies.begin(), input.owner_policies.end(),
                                         [&](const auto& value) { return value.owner == owner; });
                        if (policy == input.owner_policies.end()) {
                            throw std::logic_error("capture seed owner has no portfolio policy");
                        }
                        priorities.emplace_back(
                            pressure_seed_owner_value(owner, policy->private_retention_weight,
                                                      policy->explicit_shared_credit,
                                                      input.checkpoint_policies),
                            owner);
                    }
                };
                append(input.private_owner_ids);
                append(input.shared_owner_ids);
                std::stable_sort(priorities.begin(), priorities.end(),
                                 [](const auto& a, const auto& b) {
                                     return std::tuple{a.first, a.second.value} <
                                            std::tuple{b.first, b.second.value};
                                 });
                std::vector<PlanningOwnerId> owner_order;
                owner_order.reserve(priorities.size());
                for (const auto& priority : priorities) { owner_order.push_back(priority.second); }
                const auto started   = planning_now_ns<SearchClock>();
                const auto remaining = input.allowance.remaining(started);
                const auto reserve   = std::max<std::uint64_t>(1, assessment_elapsed);
                MaterializationSeedDiagnostics diagnostics;
                std::optional<PressureTargetHandle> seed;
                if (remaining > reserve) {
                    seed =
                        session.deficit_cover_target(owner_order, remaining - reserve, diagnostics);
                }
                if (seed) {
                    const auto guidance = session.guidance(*seed);
                    if (guidance.candidate != candidate_id || guidance.stable_target_ordinal == 0 ||
                        target_marked(guidance.stable_target_ordinal, kTargetDiscovered)) {
                        throw std::logic_error(
                            "capture cover seed changed identity or repeated a target");
                    }
                    ++canonical_targets;
                    mark_target(guidance.stable_target_ordinal, kTargetDiscovered);
                    queue_.push_back({.target            = *seed,
                                      .ordinal           = guidance.stable_target_ordinal,
                                      .construction_seed = true});
                    continue;
                }
                // This pass's second slot is for construction, not a partial BFS expansion.
                incomplete = assessment.expandable;
                break;
            }

            if (input.prune && incumbent && incumbent->value.gain == bound &&
                incumbent->degradation_units == 0 && incumbent->dropped_checkpoints == 0 &&
                incumbent->stable_target == 0) {
                proven = true;
                break;
            }
            if (targets_evaluated >= input.target_budget ||
                input.allowance.remaining(planning_now_ns<SearchClock>()) == 0 ||
                (input.search && input.search->targets_remaining == 0)) {
                incomplete                = assessment.expandable || cursor < queue_.size();
                const bool discovery_only = input.target_budget == 1 ||
                                            (input.allow_deficit_seed && input.target_budget == 2);
                if (input.search && incomplete && !discovery_only) {
                    input.search->stop    = input.search->targets_remaining == 0
                                                ? CaptureSearchStop::Targets
                                                : CaptureSearchStop::Time;
                    input.search->stopped = true;
                }
                break;
            }
            if (!assessment.expandable || target_marked(queued.ordinal, kTargetExpanded)) {
                continue;
            }
            const std::uint32_t remaining = input.target_budget - canonical_targets;
            if (remaining == 0) {
                incomplete = true;
                continue;
            }
            if (input.search && !input.search->allow(bound, planning_now_ns<SearchClock>())) {
                incomplete = true;
                break;
            }
            const auto expansion_started = planning_now_ns<SearchClock>();
            auto prepared                = session.prepare_expansion(queued.target);
            if (prepared.new_canonical_count() > remaining) {
                session.discard_expansion(std::move(prepared));
                incomplete = true;
                if (input.search) { input.search->stop = CaptureSearchStop::Canonical; }
                continue;
            }
            const auto children = session.commit_expansion(std::move(prepared));
            if (input.search) {
                input.search->assessment_estimate_ns =
                    std::max(input.search->assessment_estimate_ns,
                             planning_now_ns<SearchClock>() - expansion_started);
            }
            canonical_targets += children.new_canonical_count;
            mark_target(queued.ordinal, kTargetExpanded);
            for (const PressureTargetHandle child : children.children) {
                const PressureTargetGuidance guidance = session.guidance(child);
                if (guidance.candidate != candidate_id) {
                    throw std::logic_error("shared capture child changed candidate identity");
                }
                if (target_marked(guidance.stable_target_ordinal, kTargetDiscovered)) { continue; }
                mark_target(guidance.stable_target_ordinal, kTargetDiscovered);
                queue_.push_back(QueuedTarget{
                    .target  = child,
                    .ordinal = guidance.stable_target_ordinal,
                });
            }
        }

        if (input.complete) {
            *input.complete = proven || (!incomplete && cursor == queue_.size());
        }
        if (!incumbent || (input.allowance.cancellation &&
                           input.allowance.cancellation->load(std::memory_order_relaxed))) {
            return std::nullopt;
        }
        Result result{
            .pressure                = std::nullopt,
            .owner_outcomes          = std::move(incumbent->owner_outcomes),
            .checkpoint_outcomes     = std::move(incumbent->checkpoint_outcomes),
            .baseline_value          = incumbent->value.baseline_public,
            .target_value            = incumbent->value.target_public,
            .immediate_ns            = incumbent->value.immediate,
            .net_gain                = incumbent->value.gain,
            .stable_scenario_ordinal = input.stable_scenario_ordinal,
            .stable_target_ordinal   = incumbent->stable_target,
            .targets_evaluated       = targets_evaluated,
            .degradation_units       = incumbent->degradation_units,
            .dropped_checkpoints     = incumbent->dropped_checkpoints,
            .upper_bound             = bound,
        };
        if (input.global_best && !better(result, *input.global_best)) { return std::nullopt; }
        result.pressure = session.seal(std::move(*incumbent->assessed));
        if (!result.pressure) {
            throw std::logic_error("selected shared capture target could not be sealed");
        }
        return result;
    }

    static constexpr std::uint32_t kTargetBudget = 4096;

private:
    struct QueuedTarget {
        PressureTargetHandle target;
        std::uint32_t ordinal  = 0;
        bool construction_seed = false;
    };

    struct CombinedImpact {
        PlanningOwnerId owner;
        CheckpointRef checkpoint;
        std::uint64_t target_ns = 0;
    };

    struct TransitionValue {
        std::uint64_t baseline_public = 0;
        std::uint64_t target_public   = 0;
        std::uint64_t private_loss    = 0;
        std::uint64_t immediate       = 0;
        std::uint64_t gain            = 0;
        bool positive                 = false;
        bool saturated                = false;

        [[nodiscard]] friend constexpr bool operator==(TransitionValue,
                                                       TransitionValue) noexcept = default;
    };

    struct Incumbent {
        PressureTargetHandle target;
        std::optional<AssessedPressureTarget> assessed;
        TransitionValue value;
        std::uint32_t stable_target       = 0;
        std::uint32_t degradation_units   = 0;
        std::uint32_t dropped_checkpoints = 0;
        std::vector<PressureOwnerOutcome> owner_outcomes;
        std::vector<PressureCheckpointOutcome> checkpoint_outcomes;
    };

    static void validate(const Input& input) {
        if (input.capture == nullptr || !input.capture->publishes_shared ||
            input.target_budget == 0 || input.target_budget > kTargetBudget ||
            input.private_owners.size() != input.private_owner_ids.size() ||
            input.shared_owners.size() != input.shared_owner_ids.size()) {
            throw std::invalid_argument("shared capture planning input is malformed");
        }
    }

    static constexpr std::uint8_t kTargetDiscovered = BoundedTargetLedger::Discovered;
    static constexpr std::uint8_t kTargetAssessed   = BoundedTargetLedger::Assessed;
    static constexpr std::uint8_t kTargetExpanded   = BoundedTargetLedger::Expanded;

    [[nodiscard]] bool target_marked(std::uint32_t ordinal, std::uint8_t mark) const {
        return target_ledger_.contains(ordinal, mark);
    }

    void mark_target(std::uint32_t ordinal, std::uint8_t mark) {
        target_ledger_.mark(ordinal, mark);
    }

    [[nodiscard]] static const CheckpointPolicy*
    checkpoint_policy_for(std::span<const CheckpointPolicy> policies, PlanningOwnerId owner,
                          CheckpointRef checkpoint) noexcept {
        const auto found = std::find_if(policies.begin(), policies.end(), [&](const auto& policy) {
            return policy.owner == owner && policy.checkpoint == checkpoint;
        });
        return found == policies.end() ? nullptr : &*found;
    }

    [[nodiscard]] static std::uint64_t saturating_multiply(std::uint64_t value,
                                                           std::uint64_t multiplier) noexcept {
        return multiplier != 0 && value > std::numeric_limits<std::uint64_t>::max() / multiplier
                   ? std::numeric_limits<std::uint64_t>::max()
                   : value * multiplier;
    }

    [[nodiscard]] TransitionValue fold_target(const Input& input,
                                              const PressureTargetAssessment& assessment,
                                              const ContextMachineCostModel& machine_cost,
                                              bool optimistic = false) {
        impact_scratch_.clear();
        for (const PressureCheckpointRecoveryImpact& impact : assessment.checkpoint_impacts) {
            if (checkpoint_policy_for(input.checkpoint_policies, impact.owner, impact.checkpoint) ==
                nullptr) {
                throw std::logic_error("capture pressure impact has no portfolio checkpoint");
            }
            const auto found = std::find_if(
                impact_scratch_.begin(), impact_scratch_.end(), [&](const CombinedImpact& value) {
                    return value.owner == impact.owner && value.checkpoint == impact.checkpoint;
                });
            if (found == impact_scratch_.end()) {
                if (impact.target_recovery_work.empty()) {
                    throw std::logic_error("capture pressure impact has no supported recipe");
                }
                impact_scratch_.push_back(CombinedImpact{
                    .owner      = impact.owner,
                    .checkpoint = impact.checkpoint,
                    .target_ns =
                        price_checkpoint_recovery_work(machine_cost, impact.target_recovery_work),
                });
            } else {
                throw std::logic_error("capture pressure impact is duplicated");
            }
        }

        owner_scratch_.clear();
        for (const OwnerPolicy& policy : input.owner_policies) {
            owner_scratch_.push_back(ContextPortfolioOwnerPolicy{
                .owner                    = policy.owner,
                .private_retention_weight = policy.private_retention_weight,
                .explicit_shared_credit   = policy.explicit_shared_credit,
            });
        }
        const PlanningOwnerId candidate_owner = next_candidate_owner(input.owner_policies);
        const bool candidate_credit =
            has_shared_candidate_evidence(input.capture->shared_evidence,
                                          SharedCandidateEvidence::ExplicitBoundary) ||
            has_shared_candidate_evidence(input.capture->shared_evidence,
                                          SharedCandidateEvidence::RequestedAutomatic);
        owner_scratch_.push_back(ContextPortfolioOwnerPolicy{
            .owner                    = candidate_owner,
            .private_retention_weight = 0,
            .explicit_shared_credit   = candidate_credit,
        });

        checkpoint_scratch_.clear();
        for (const CheckpointPolicy& policy : input.checkpoint_policies) {
            std::uint64_t target_recovery =
                optimistic ? policy.optimistic_recovery_ns : policy.baseline_recovery_ns;
            if (policy.optimistic_recovery_ns > policy.baseline_recovery_ns) {
                throw std::logic_error("capture recovery lower bound exceeds baseline");
            }
            if (input.direct_shared_victim == policy.owner) {
                target_recovery = policy.rebuild_ns;
            } else {
                const auto impact = std::find_if(impact_scratch_.begin(), impact_scratch_.end(),
                                                 [&](const CombinedImpact& value) {
                                                     return value.owner == policy.owner &&
                                                            value.checkpoint == policy.checkpoint;
                                                 });
                if (impact != impact_scratch_.end()) {
                    target_recovery = impact->target_ns;
                    if (target_recovery < policy.optimistic_recovery_ns) {
                        throw std::logic_error("capture recovery violates Program lower bound");
                    }
                }
            }
            checkpoint_scratch_.push_back(ContextPortfolioCheckpointValue{
                .owner                = policy.owner,
                .demand_mask          = policy.demand_mask,
                .rebuild_ns           = policy.rebuild_ns,
                .baseline_recovery_ns = policy.baseline_recovery_ns,
                .target_recovery_ns   = target_recovery,
            });
        }
        checkpoint_scratch_.push_back(ContextPortfolioCheckpointValue{
            .owner                = candidate_owner,
            .demand_mask          = input.candidate_demand_mask,
            .rebuild_ns           = input.candidate_rebuild_ns,
            .baseline_recovery_ns = input.candidate_rebuild_ns,
            .target_recovery_ns   = price_checkpoint_recovery_work(
                machine_cost, input.capture->projected_recovery_work),
        });

        const ContextPortfolioValueResult portfolio =
            portfolio_value_.fold(owner_scratch_, checkpoint_scratch_);
        const std::uint64_t multiplier =
            static_cast<std::uint64_t>(input.blocked_runnable_requests) + 1U;
        const std::uint64_t target_immediate = saturating_multiply(
            price_materialization_machine_work(machine_cost, assessment.machine_work).immediate_ns,
            multiplier);
        const std::uint64_t baseline_immediate =
            saturating_multiply(input.private_baseline_immediate_ns, multiplier);
        const std::uint64_t immediate =
            target_immediate > baseline_immediate ? target_immediate - baseline_immediate : 0;
        bool saturated = portfolio.saturated ||
                         target_immediate == std::numeric_limits<std::uint64_t>::max() ||
                         baseline_immediate == std::numeric_limits<std::uint64_t>::max();
        std::uint64_t threshold  = portfolio.baseline_public_value;
        const auto add_threshold = [&](std::uint64_t increment) {
            if (increment > std::numeric_limits<std::uint64_t>::max() - threshold) {
                threshold = std::numeric_limits<std::uint64_t>::max();
                saturated = true;
            } else {
                threshold += increment;
            }
        };
        if (!optimistic) {
            add_threshold(portfolio.private_transition_loss);
            add_threshold(immediate);
        }
        const bool positive = !saturated && portfolio.target_public_value > threshold;
        return TransitionValue{
            .baseline_public = portfolio.baseline_public_value,
            .target_public   = portfolio.target_public_value,
            .private_loss    = portfolio.private_transition_loss,
            .immediate       = immediate,
            .gain            = positive ? portfolio.target_public_value - threshold : 0,
            .positive        = positive,
            .saturated       = saturated,
        };
    }

    [[nodiscard]] static PlanningOwnerId
    next_candidate_owner(std::span<const OwnerPolicy> policies) {
        std::uint32_t value = 0;
        for (const OwnerPolicy& policy : policies) {
            if (policy.owner.value == std::numeric_limits<std::uint32_t>::max()) {
                throw std::overflow_error("capture portfolio owner ID overflow");
            }
            value = std::max(value, policy.owner.value + 1U);
        }
        return PlanningOwnerId{.value = value};
    }

    [[nodiscard]] static bool better(const TransitionValue& value,
                                     const PressureTargetAssessment& assessment, const Input& input,
                                     const Incumbent& incumbent) noexcept {
        return std::tuple{
                   value.gain,
                   std::numeric_limits<std::uint32_t>::max() - assessment.degradation_units,
                   std::numeric_limits<std::uint32_t>::max() - assessment.dropped_checkpoints,
                   std::numeric_limits<std::uint32_t>::max() - input.stable_scenario_ordinal,
                   std::numeric_limits<std::uint32_t>::max() - assessment.stable_target_ordinal} >
               std::tuple{incumbent.value.gain,
                          std::numeric_limits<std::uint32_t>::max() - incumbent.degradation_units,
                          std::numeric_limits<std::uint32_t>::max() - incumbent.dropped_checkpoints,
                          std::numeric_limits<std::uint32_t>::max() - input.stable_scenario_ordinal,
                          std::numeric_limits<std::uint32_t>::max() - incumbent.stable_target};
    }

    ContextPortfolioValue portfolio_value_;
    std::vector<QueuedTarget> queue_;
    BoundedTargetLedger target_ledger_;
    std::vector<CombinedImpact> impact_scratch_;
    std::vector<ContextPortfolioOwnerPolicy> owner_scratch_;
    std::vector<ContextPortfolioCheckpointValue> checkpoint_scratch_;
};

} // namespace ninfer::runtime
