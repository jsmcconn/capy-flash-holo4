// Two-device qualification suite for the cross-device collectives declared in
// include/ninfer/ops/allreduce.h.
//
// Every case REQUIRES two CUDA devices driven from ONE process; with fewer than two visible
// devices the suite reports the repository's skip code (77) instead of failing.
//
// Oracles:
//   allreduce_sum - independent FP64 elementwise sum of the two represented BF16 inputs. The
//     Op's observable output is BF16, so the comparison allows one BF16 ulp of output storage
//     rounding (relative 3.95e-3 >= 2^-8); the oracle itself performs no rounding. The two ranks'
//     results must also be bit-identical.
//   allgather_rows - exact: the Op only relocates rows, so every destination byte is compared
//     bit-for-bit against the concatenated source halves.
//
// Both transports are qualified: the staged path eagerly and inside a two-device CUDA Graph, and
// the PeerMailbox exchange (both of its kernels) inside a graph replayed several times with fresh
// inputs; the three captured transports must also agree bit for bit.
//
// The direct peer pull (NINFER_TP_PEER_PULL=1, which requires direct P2P) is qualified against
// an exact oracle -- the definition fl32(a + b) rounded once to BF16, on every BF16 encoding class
// -- and bit for bit against the staged path, eagerly and captured, at the payloads it serves and
// the ones it leaves to the other routes, with mixed sizes, replays and ownership moves. Its
// captured paired copies (captured_peer_copy_pair) are qualified exactly: both directions, one
// direction, odd, unaligned and large sizes, operands a GPU stage produced just before the copy
// and overwrites just after it, replays, back-to-back launches and an in-place graph update.
//
// Selection: run with NINFER_TP_PEER_PULL=1 in the environment, every case runs on the peer-pull
// flags (every PeerEvents the suite creates carries them, and Core refuses unflagged captured peer
// copies); the case labels then end in "[NINFER_TP_PEER_PULL=1]". Without it, the staged default
// is the control and only the peer-pull cases select the flags.
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/peer_mailbox.h"
#include "ops/op_tester.h"
#include "ops/split_test_support.h"

#include "core/decode_graph.h"
#include "core/device.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// One BF16 ulp (2^-8 = 3.90625e-3) with the same small margin the residual_add suite uses. The
// only error the Op may introduce is the single round-to-nearest-even of a+b into BF16 storage,
// which is bounded by half an ulp; the criterion is deliberately not tighter than one ulp so a
// double-rounded tie cannot make the suite flaky.
constexpr PointwiseCriterion allreduce_sum_bf16_criterion() {
    return {/*absolute*/ 0.0, /*relative*/ 3.95e-3};
}

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

// Independent oracle: the complete formula in FP64 from the represented BF16 inputs.
std::vector<double> allreduce_sum_oracle(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<double> expected(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        expected[i] = static_cast<double>(a[i]) + static_cast<double>(b[i]);
    }
    return expected;
}

// The process selection (see the file comment).
bool peer_pull_selected() {
    const char* value = std::getenv("NINFER_TP_PEER_PULL");
    return value != nullptr && std::string_view(value) == "1";
}

// A case label naming the selection the case ran under.
std::string disclosed(const char* label) {
    return std::string(label) + (peer_pull_selected() ? " [NINFER_TP_PEER_PULL=1]" : "");
}

// `ne0` is the contiguous dimension and `ne1` the outer one, so a 1-D buffer passes ne1 == 1 and
// the real row-parallel residual passes {5120, 48}.
int run_allreduce_case(const char* case_label, std::int32_t ne0, std::int32_t ne1,
                       std::uint32_t seed, const ExecutionContext& ec,
                       const ops::PeerEvents& events) {
    const std::string label = disclosed(case_label);
    const std::size_t count = static_cast<std::size_t>(ne0) * static_cast<std::size_t>(ne1);
    std::vector<float> a(count), b(count);
    fill_uniform(a, seed, -8.0f, 8.0f);
    fill_uniform(b, seed + 1, -8.0f, 8.0f);
    round_to_bf16(a);
    round_to_bf16(b);

    const auto expected     = allreduce_sum_oracle(a, b);
    const auto a_bits       = encode_bf16(a);
    const auto b_bits       = encode_bf16(b);
    const std::size_t bytes = count * sizeof(std::uint16_t);

    set_device(ec, 0);
    GuardedDeviceBuffer buffer_0(bytes), staging_0(bytes);
    buffer_0.copy_from_host(a_bits.data(), bytes);
    staging_0.fill(0);

    set_device(ec, 1);
    GuardedDeviceBuffer buffer_1(bytes), staging_1(bytes);
    buffer_1.copy_from_host(b_bits.data(), bytes);
    staging_1.fill(0);

    const std::array<Tensor, 2> buffer{Tensor(buffer_0.data(), DType::BF16, {ne0, ne1}),
                                       Tensor(buffer_1.data(), DType::BF16, {ne0, ne1})};
    const std::array<Tensor, 2> staging{Tensor(staging_0.data(), DType::BF16, {ne0, ne1}),
                                        Tensor(staging_1.data(), DType::BF16, {ne0, ne1})};

    retire_staging(ec);
    ops::allreduce_sum(buffer, staging, ec, events);
    synchronize_both(ec);

    int failures = 0;
    set_device(ec, 0);
    failures += verify_pointwise((std::string(label) + " device 0").c_str(),
                                 from_device_bf16(buffer_0.data(), count), expected,
                                 allreduce_sum_bf16_criterion());
    failures += buffer_0.verify_guards("allreduce buffer device 0");
    failures += staging_0.verify_guards("allreduce staging device 0");
    set_device(ec, 1);
    failures += verify_pointwise((std::string(label) + " device 1").c_str(),
                                 from_device_bf16(buffer_1.data(), count), expected,
                                 allreduce_sum_bf16_criterion());
    failures += buffer_1.verify_guards("allreduce buffer device 1");
    failures += staging_1.verify_guards("allreduce staging device 1");
    return failures;
}

// `row_length` is ne[0] (the contiguous row) and the gathered axis is ne[1], so device 0 owns the
// leading `rows_0` rows and device 1 the trailing `rows_1`.
int run_allgather_case(const char* case_label, std::int32_t rows_0, std::int32_t rows_1,
                       std::int32_t row_length, std::uint32_t seed, const ExecutionContext& ec,
                       const ops::PeerEvents& events) {
    const std::string label  = disclosed(case_label);
    const std::int32_t rows  = rows_0 + rows_1;
    const std::size_t part_0 = static_cast<std::size_t>(rows_0) * row_length;
    const std::size_t part_1 = static_cast<std::size_t>(rows_1) * row_length;

    std::vector<float> source_0(part_0), source_1(part_1);
    fill_uniform(source_0, seed, -8.0f, 8.0f);
    fill_uniform(source_1, seed + 1, -8.0f, 8.0f);
    const auto bits_0 = encode_bf16(source_0);
    const auto bits_1 = encode_bf16(source_1);

    // Exact oracle: the gathered image is the concatenation of the two owned row ranges.
    std::vector<std::uint16_t> expected;
    expected.reserve(part_0 + part_1);
    expected.insert(expected.end(), bits_0.begin(), bits_0.end());
    expected.insert(expected.end(), bits_1.begin(), bits_1.end());

    const std::size_t full_bytes = expected.size() * sizeof(std::uint16_t);

    set_device(ec, 0);
    GuardedDeviceBuffer source_device_0(part_0 * sizeof(std::uint16_t));
    GuardedDeviceBuffer destination_0(full_bytes);
    source_device_0.copy_from_host(bits_0.data(), source_device_0.bytes());
    destination_0.fill(0xcd);

    set_device(ec, 1);
    GuardedDeviceBuffer source_device_1(part_1 * sizeof(std::uint16_t));
    GuardedDeviceBuffer destination_1(full_bytes);
    source_device_1.copy_from_host(bits_1.data(), source_device_1.bytes());
    destination_1.fill(0xcd);

    const std::array<Tensor, 2> destination{
        Tensor(destination_0.data(), DType::BF16, {row_length, rows}),
        Tensor(destination_1.data(), DType::BF16, {row_length, rows})};
    const std::array<Tensor, 2> part{
        Tensor(source_device_0.data(), DType::BF16, {row_length, rows_0}),
        Tensor(source_device_1.data(), DType::BF16, {row_length, rows_1})};

    retire_staging(ec);
    ops::allgather_rows(destination, part, ec, events);
    synchronize_both(ec);

    int failures = 0;
    set_device(ec, 0);
    failures +=
        verify_exact((std::string(label) + " device 0").c_str(),
                     from_device<std::uint16_t>(destination_0.data(), expected.size()), expected);
    failures += verify_exact("allgather source device 0 unchanged",
                             from_device<std::uint16_t>(source_device_0.data(), part_0), bits_0);
    failures += destination_0.verify_guards("allgather destination device 0");
    failures += source_device_0.verify_guards("allgather source device 0");
    set_device(ec, 1);
    failures +=
        verify_exact((std::string(label) + " device 1").c_str(),
                     from_device<std::uint16_t>(destination_1.data(), expected.size()), expected);
    failures += verify_exact("allgather source device 1 unchanged",
                             from_device<std::uint16_t>(source_device_1.data(), part_1), bits_1);
    failures += destination_1.verify_guards("allgather destination device 1");
    failures += source_device_1.verify_guards("allgather source device 1");
    return failures;
}

// Regression guard for the cross-call write-after-read hazard.
//
// kRounds alternating collectives are issued back-to-back on ONE set of buffers, staging, and
// events with NO host synchronization until the very end -- exactly what the tensor-parallel
// forward loop produces (128 all-reduces per decode token, plus one logit allgather per column)
// and what a captured graph replays. Under the earlier push-based
// design, call k+1's inbound write into the peer's staging was ordered only against the sender's
// own stream history, not against the receiver's call-k read of that staging; this loop is the
// pattern where that window opens.
//
// Two properties make a stale operand observable rather than benign:
//   * the chain value changes every round -- both devices start at 2^-40 and each all-reduce
//     doubles it, so after k rounds every element is exactly 2^(k-40). Every intermediate is a
//     power of two, hence exact in BF16 and in the FP32 accumulator, and an operand from the wrong
//     round yields a value that is not the expected power of two (3v instead of 2v, say);
//   * the interleaved all-gather reads the live all-reduce buffers as its parts, so it also
//     carries the round's value into its destination and is checked against the same expectation,
//     and its own cross-call ordering (the peer must finish reading part[r] before the next
//     all-reduce overwrites it) is exercised rather than merely occupying the streams.
//
// Deliberate skew: a large memset is queued on rank 0's stream first, with no host sync, so the
// two streams run genuinely out of step for the first rounds instead of in lockstep.
int run_chained_case(const char* case_label, const ExecutionContext& ec,
                     const ops::PeerEvents& events) {
    const std::string label          = disclosed(case_label);
    constexpr std::int32_t n         = 5120;
    constexpr int kRounds            = 64;
    constexpr int kStartExponent     = -40;
    constexpr std::size_t kSkewBytes = 256u << 20;
    constexpr int kSkewRepeats       = 8;

    const float start           = std::ldexp(1.0f, kStartExponent);
    const double expected_value = std::ldexp(1.0, kStartExponent + kRounds);
    const std::size_t count     = static_cast<std::size_t>(n);
    const std::size_t bytes     = count * sizeof(std::uint16_t);

    const std::vector<std::uint16_t> start_bits(count, f32_to_bf16(start));
    const std::vector<double> expected(count, expected_value);
    // Both halves of the gathered image come from the two all-reduce buffers, which hold the same
    // value once the round's all-reduce has run.
    const std::vector<double> expected_gathered(2 * count, expected_value);

    set_device(ec, 0);
    GuardedDeviceBuffer buffer_0(bytes), staging_0(bytes), gathered_0(2 * bytes);
    buffer_0.copy_from_host(start_bits.data(), bytes);
    staging_0.fill(0);
    gathered_0.fill(0xcd);
    DeviceBuffer skew(kSkewBytes);

    set_device(ec, 1);
    GuardedDeviceBuffer buffer_1(bytes), staging_1(bytes), gathered_1(2 * bytes);
    buffer_1.copy_from_host(start_bits.data(), bytes);
    staging_1.fill(0);
    gathered_1.fill(0xcd);

    const std::array<Tensor, 2> buffer{Tensor(buffer_0.data(), DType::BF16, {n}),
                                       Tensor(buffer_1.data(), DType::BF16, {n})};
    const std::array<Tensor, 2> staging{Tensor(staging_0.data(), DType::BF16, {n}),
                                        Tensor(staging_1.data(), DType::BF16, {n})};
    const std::array<Tensor, 2> gathered{Tensor(gathered_0.data(), DType::BF16, {n, 2}),
                                         Tensor(gathered_1.data(), DType::BF16, {n, 2})};
    const std::array<Tensor, 2> part{Tensor(buffer_0.data(), DType::BF16, {n, 1}),
                                     Tensor(buffer_1.data(), DType::BF16, {n, 1})};

    retire_staging(ec);

    set_device(ec, 0);
    for (int i = 0; i < kSkewRepeats; ++i) {
        cuda_check(cudaMemsetAsync(skew.p, i & 0xff, skew.bytes, ec.dev[0]->stream),
                   "cudaMemsetAsync skew");
    }

    for (int round = 0; round < kRounds; ++round) {
        ops::allreduce_sum(buffer, staging, ec, events);
        ops::allgather_rows(gathered, part, ec, events);
    }
    synchronize_both(ec);

    int failures = 0;
    set_device(ec, 0);
    failures +=
        verify_pointwise(label + " 64x device 0 buffer", from_device_bf16(buffer_0.data(), count),
                         expected, allreduce_sum_bf16_criterion());
    failures += verify_pointwise(label + " 64x device 0 gathered",
                                 from_device_bf16(gathered_0.data(), 2 * count), expected_gathered,
                                 allreduce_sum_bf16_criterion());
    failures += buffer_0.verify_guards(label + " buffer device 0");
    failures += staging_0.verify_guards(label + " staging device 0");
    failures += gathered_0.verify_guards(label + " gathered device 0");
    set_device(ec, 1);
    failures +=
        verify_pointwise(label + " 64x device 1 buffer", from_device_bf16(buffer_1.data(), count),
                         expected, allreduce_sum_bf16_criterion());
    failures += verify_pointwise(label + " 64x device 1 gathered",
                                 from_device_bf16(gathered_1.data(), 2 * count), expected_gathered,
                                 allreduce_sum_bf16_criterion());
    failures += buffer_1.verify_guards(label + " buffer device 1");
    failures += staging_1.verify_guards(label + " staging device 1");
    failures += gathered_1.verify_guards(label + " gathered device 1");
    return failures;
}

// The per-decode-token all-reduce payload is one 5120-element BF16 hidden row = 10 KiB, issued
// 128 times per token. Measure the full call: both peer transfers plus both local combines, with
// the host waiting for both devices to retire the work.
//
// The figure is SYNC-DOMINATED and is an upper bound, not the op's device cost: each iteration
// pays two host-side cudaStreamSynchronize round trips that the production path does not, because
// the forward loop issues 128 of these back-to-back and the decode path replays them inside a
// captured CUDA graph with no host sync at all. Read it as "one isolated, fully drained
// all-reduce". It is informative only: host-sync latency depends on the platform and its load.
int run_microbenchmark(const ExecutionContext& ec, const ops::PeerEvents& events) {
    constexpr std::int32_t n        = 5120;
    constexpr int kWarmupIterations = 50;
    constexpr int kTimedIterations  = 500;
    const std::size_t bytes         = static_cast<std::size_t>(n) * sizeof(std::uint16_t);

    set_device(ec, 0);
    GuardedDeviceBuffer buffer_0(bytes), staging_0(bytes);
    buffer_0.fill(0);
    staging_0.fill(0);
    set_device(ec, 1);
    GuardedDeviceBuffer buffer_1(bytes), staging_1(bytes);
    buffer_1.fill(0);
    staging_1.fill(0);

    const std::array<Tensor, 2> buffer{Tensor(buffer_0.data(), DType::BF16, {n}),
                                       Tensor(buffer_1.data(), DType::BF16, {n})};
    const std::array<Tensor, 2> staging{Tensor(staging_0.data(), DType::BF16, {n}),
                                        Tensor(staging_1.data(), DType::BF16, {n})};

    retire_staging(ec);
    for (int i = 0; i < kWarmupIterations; ++i) {
        ops::allreduce_sum(buffer, staging, ec, events);
        synchronize_both(ec);
    }

    std::vector<double> samples;
    samples.reserve(kTimedIterations);
    const auto started = std::chrono::steady_clock::now();
    for (int i = 0; i < kTimedIterations; ++i) {
        const auto call_started = std::chrono::steady_clock::now();
        ops::allreduce_sum(buffer, staging, ec, events);
        synchronize_both(ec);
        const std::chrono::duration<double, std::micro> call_elapsed =
            std::chrono::steady_clock::now() - call_started;
        samples.push_back(call_elapsed.count());
    }
    const std::chrono::duration<double, std::micro> elapsed =
        std::chrono::steady_clock::now() - started;
    const double mean_micros = elapsed.count() / kTimedIterations;
    std::sort(samples.begin(), samples.end());

    std::cout << "allreduce microbench: " << bytes << " B bf16 mean " << mean_micros << " us, p50 "
              << samples[samples.size() / 2] << " us, p99 " << samples[(samples.size() * 99) / 100]
              << " us, max " << samples.back() << " us over " << kTimedIterations << " iterations ("
              << disclosed("eager, host-sync dominated") << ")\n";

    int failures = 0;
    set_device(ec, 0);
    failures += buffer_0.verify_guards("microbench buffer device 0");
    failures += staging_0.verify_guards("microbench staging device 0");
    set_device(ec, 1);
    failures += buffer_1.verify_guards("microbench buffer device 1");
    failures += staging_1.verify_guards("microbench staging device 1");
    return failures;
}

// Captures `body` into ONE two-device graph: rank 1's stream is forked into rank 0's capture.
void capture_two_devices(const ExecutionContext& ec, const DecodeGraphPeerBridge& bridge,
                         DecodeGraphDefinition& definition, const std::function<void()>& body) {
    set_device(ec, 0);
    definition.capture(
        ec.dev[0]->stream,
        [&] {
            body();
            set_device(ec, 0);
        },
        DecodeGraphPeerCapture{.bridge = &bridge, .stream = ec.dev[1]->stream});
}

void launch_two_devices(const ExecutionContext& ec, DecodeGraphExecutable& executable) {
    set_device(ec, 0);
    executable.launch(ec.dev[0]->stream);
    synchronize_both(ec);
}

// Captured replay, the decode route. `sites` all-reduces of one [ne0, ne1] buffer per rank are
// captured once into a two-device graph and the executable is launched kReplays times, each with
// fresh random inputs and no host step between launches other than the input upload. After the
// first site both ranks hold bf16(a + b); every later site doubles it exactly, so the oracle is
// 2^(sites-1) * (a + b) in FP64 with the one-ulp criterion.
//
// Stale operands are made observable: a captured memset of kSkewBytes on rank 1 runs before the
// first site, so rank 0 reaches every exchange long before rank 1 has republished. An exchange
// that accepted the previous replay's publication (the ABA hazard of a flag that is not reset or
// advanced between launches) would combine the previous inputs and miss the oracle.
//
// kBackToBack further launches are then issued with no host synchronization between them, as a
// serving loop may, and must double the value 'sites' times each. One eager call afterwards
// checks that the staged path composes with the captured history on the same buffers and events.
//
// `mailbox` names the mailbox attached to `events`, or null. The node count proves the selected
// transport: the mailbox exchange is one kernel node per device per site, the staged path at
// least two per device per site.
int run_captured_case(const char* case_label, std::int32_t ne0, std::int32_t ne1, int sites,
                      bool expect_mailbox, const ExecutionContext& ec,
                      const ops::PeerEvents& events, const ops::PeerMailbox* mailbox) {
    const std::string label          = disclosed(case_label);
    constexpr int kReplays           = 4;
    constexpr int kBackToBack        = 3;
    constexpr std::size_t kSkewBytes = 64u << 20;

    const std::size_t count = static_cast<std::size_t>(ne0) * static_cast<std::size_t>(ne1);
    const std::size_t bytes = count * sizeof(std::uint16_t);

    set_device(ec, 0);
    GuardedDeviceBuffer buffer_0(bytes), staging_0(bytes);
    staging_0.fill(0);
    set_device(ec, 1);
    GuardedDeviceBuffer buffer_1(bytes), staging_1(bytes);
    staging_1.fill(0);
    DeviceBuffer skew(kSkewBytes);

    const std::array<Tensor, 2> buffer{Tensor(buffer_0.data(), DType::BF16, {ne0, ne1}),
                                       Tensor(buffer_1.data(), DType::BF16, {ne0, ne1})};
    const std::array<Tensor, 2> staging{Tensor(staging_0.data(), DType::BF16, {ne0, ne1}),
                                        Tensor(staging_1.data(), DType::BF16, {ne0, ne1})};
    retire_staging(ec);

    const DecodeGraphPeerBridge bridge(ec.dev[0]->device, ec.dev[1]->device);
    DecodeGraphDefinition definition;
    capture_two_devices(ec, bridge, definition, [&] {
        set_device(ec, 1);
        cuda_check(cudaMemsetAsync(skew.p, 0, skew.bytes, ec.dev[1]->stream), "skew memset");
        for (int site = 0; site < sites; ++site) {
            ops::allreduce_sum(buffer, staging, ec, events);
        }
    });

    int failures                = 0;
    const std::size_t nodes     = definition.node_count();
    const std::size_t exchanges = 1 + 2 * static_cast<std::size_t>(sites);
    if (expect_mailbox ? nodes != exchanges : nodes <= exchanges) {
        std::cerr << label << ": " << nodes << " graph nodes do not match the "
                  << (expect_mailbox ? "mailbox" : "staged") << " transport\n";
        ++failures;
    }

    DecodeGraphExecutable executable;
    executable.instantiate(definition);
    const double scale = std::ldexp(1.0, sites - 1);
    std::vector<float> a(count), b(count);
    std::vector<double> expected(count);
    for (int replay = 0; replay < kReplays; ++replay) {
        const std::uint32_t seed = 301u + 2u * static_cast<std::uint32_t>(replay);
        fill_uniform(a, seed, -8.0f, 8.0f);
        fill_uniform(b, seed + 1, -8.0f, 8.0f);
        round_to_bf16(a);
        round_to_bf16(b);
        const auto sum = allreduce_sum_oracle(a, b);
        for (std::size_t i = 0; i < count; ++i) { expected[i] = scale * sum[i]; }
        const auto a_bits = encode_bf16(a);
        const auto b_bits = encode_bf16(b);
        set_device(ec, 0);
        buffer_0.copy_from_host(a_bits.data(), bytes);
        set_device(ec, 1);
        buffer_1.copy_from_host(b_bits.data(), bytes);
        retire_staging(ec);

        launch_two_devices(ec, executable);

        const std::string replay_label = std::string(label) + " replay " + std::to_string(replay);
        set_device(ec, 0);
        const auto bits_0 = from_device<std::uint16_t>(buffer_0.data(), count);
        failures += verify_pointwise((replay_label + " device 0").c_str(),
                                     from_device_bf16(buffer_0.data(), count), expected,
                                     allreduce_sum_bf16_criterion());
        set_device(ec, 1);
        failures += verify_exact((replay_label + " device 1 equals device 0").c_str(),
                                 from_device<std::uint16_t>(buffer_1.data(), count), bits_0);
    }

    // Back-to-back launches: every site doubles the exact power-of-two multiple of bf16(a + b).
    const double back_to_back = std::ldexp(1.0, kBackToBack * sites);
    for (double& value : expected) { value *= back_to_back; }
    set_device(ec, 0);
    for (int launch = 0; launch < kBackToBack; ++launch) { executable.launch(ec.dev[0]->stream); }
    synchronize_both(ec);
    const std::string chained_label = std::string(label) + " back-to-back launches";
    set_device(ec, 0);
    const auto chained_0 = from_device<std::uint16_t>(buffer_0.data(), count);
    failures += verify_pointwise((chained_label + " device 0").c_str(),
                                 from_device_bf16(buffer_0.data(), count), expected,
                                 allreduce_sum_bf16_criterion());
    set_device(ec, 1);
    failures += verify_exact((chained_label + " device 1 equals device 0").c_str(),
                             from_device<std::uint16_t>(buffer_1.data(), count), chained_0);

    // Eager, staged, after the captured history: exactly doubles the last launch's result.
    for (double& value : expected) { value *= 2.0; }
    ops::allreduce_sum(buffer, staging, ec, events);
    synchronize_both(ec);
    const std::string eager_label = std::string(label) + " eager after replays";
    set_device(ec, 0);
    failures += verify_pointwise((eager_label + " device 0").c_str(),
                                 from_device_bf16(buffer_0.data(), count), expected,
                                 allreduce_sum_bf16_criterion());
    failures += buffer_0.verify_guards("captured buffer device 0");
    failures += staging_0.verify_guards("captured staging device 0");
    set_device(ec, 1);
    failures += verify_pointwise((eager_label + " device 1").c_str(),
                                 from_device_bf16(buffer_1.data(), count), expected,
                                 allreduce_sum_bf16_criterion());
    failures += buffer_1.verify_guards("captured buffer device 1");
    failures += staging_1.verify_guards("captured staging device 1");
    if (mailbox != nullptr && mailbox->hang_reported()) {
        std::cerr << label << ": a mailbox exchange reported a hang\n";
        ++failures;
    }
    return failures;
}

// Informative replay cost of the decode token's 128 all-reduces of one [5120] row, per transport:
// one graph of kSites all-reduces replayed kReplays times, reported per all-reduce. Inputs are
// zero, so the values stay bounded.
void run_captured_microbenchmark(const char* label, const ExecutionContext& ec,
                                 const ops::PeerEvents& events) {
    constexpr std::int32_t n = 5120;
    constexpr int kSites     = 128;
    constexpr int kWarmup    = 5;
    constexpr int kReplays   = 50;
    const std::size_t bytes  = static_cast<std::size_t>(n) * sizeof(std::uint16_t);

    set_device(ec, 0);
    GuardedDeviceBuffer buffer_0(bytes), staging_0(bytes);
    buffer_0.fill(0);
    staging_0.fill(0);
    set_device(ec, 1);
    GuardedDeviceBuffer buffer_1(bytes), staging_1(bytes);
    buffer_1.fill(0);
    staging_1.fill(0);
    const std::array<Tensor, 2> buffer{Tensor(buffer_0.data(), DType::BF16, {n}),
                                       Tensor(buffer_1.data(), DType::BF16, {n})};
    const std::array<Tensor, 2> staging{Tensor(staging_0.data(), DType::BF16, {n}),
                                        Tensor(staging_1.data(), DType::BF16, {n})};
    retire_staging(ec);

    const DecodeGraphPeerBridge bridge(ec.dev[0]->device, ec.dev[1]->device);
    DecodeGraphDefinition definition;
    capture_two_devices(ec, bridge, definition, [&] {
        for (int site = 0; site < kSites; ++site) {
            ops::allreduce_sum(buffer, staging, ec, events);
        }
    });
    DecodeGraphExecutable executable;
    executable.instantiate(definition);
    for (int i = 0; i < kWarmup; ++i) { launch_two_devices(ec, executable); }

    const auto started = std::chrono::steady_clock::now();
    set_device(ec, 0);
    for (int i = 0; i < kReplays; ++i) { executable.launch(ec.dev[0]->stream); }
    synchronize_both(ec);
    const std::chrono::duration<double, std::micro> elapsed =
        std::chrono::steady_clock::now() - started;
    std::cout << "allreduce captured replay (" << disclosed(label) << "): " << bytes << " B bf16, "
              << elapsed.count() / (static_cast<double>(kReplays) * kSites)
              << " us per all-reduce over " << kReplays << " replays of " << kSites
              << " (informative)\n";
}

// Transport identity. The same operands go through the captured staged path, the original
// mailbox kernel and the pipelined one; every transport must leave the same BITS on both ranks
// (the combine is one FP32 add and one round-to-nearest-even whichever kernel moved the operands).
// Three sites per graph exercise both mailbox slots; the operands are fresh random BF16 values.
std::vector<std::uint16_t> captured_bits(const char* label, std::int32_t ne0, std::int32_t ne1,
                                         const std::vector<std::uint16_t>& a_bits,
                                         const std::vector<std::uint16_t>& b_bits,
                                         const ExecutionContext& ec, const ops::PeerEvents& events,
                                         int& failures,
                                         const std::function<void(int)>& before_site = {}) {
    constexpr int kSites    = 3;
    const std::size_t count = static_cast<std::size_t>(ne0) * static_cast<std::size_t>(ne1);
    const std::size_t bytes = count * sizeof(std::uint16_t);
    set_device(ec, 0);
    GuardedDeviceBuffer buffer_0(bytes), staging_0(bytes);
    staging_0.fill(0);
    set_device(ec, 1);
    GuardedDeviceBuffer buffer_1(bytes), staging_1(bytes);
    staging_1.fill(0);
    const std::array<Tensor, 2> buffer{Tensor(buffer_0.data(), DType::BF16, {ne0, ne1}),
                                       Tensor(buffer_1.data(), DType::BF16, {ne0, ne1})};
    const std::array<Tensor, 2> staging{Tensor(staging_0.data(), DType::BF16, {ne0, ne1}),
                                        Tensor(staging_1.data(), DType::BF16, {ne0, ne1})};
    retire_staging(ec);
    const DecodeGraphPeerBridge bridge(ec.dev[0]->device, ec.dev[1]->device);
    DecodeGraphDefinition definition;
    capture_two_devices(ec, bridge, definition, [&] {
        for (int site = 0; site < kSites; ++site) {
            if (before_site) { before_site(site); }
            ops::allreduce_sum(buffer, staging, ec, events);
        }
    });
    DecodeGraphExecutable executable;
    executable.instantiate(definition);
    set_device(ec, 0);
    buffer_0.copy_from_host(a_bits.data(), bytes);
    set_device(ec, 1);
    buffer_1.copy_from_host(b_bits.data(), bytes);
    retire_staging(ec);
    launch_two_devices(ec, executable);
    set_device(ec, 0);
    auto bits_0 = from_device<std::uint16_t>(buffer_0.data(), count);
    failures += buffer_0.verify_guards((std::string(label) + " buffer device 0").c_str());
    set_device(ec, 1);
    failures += verify_exact((std::string(label) + " device 1 equals device 0").c_str(),
                             from_device<std::uint16_t>(buffer_1.data(), count), bits_0);
    failures += buffer_1.verify_guards((std::string(label) + " buffer device 1").c_str());
    return bits_0;
}

int run_transport_identity_case(const char* label, std::int32_t ne0, std::int32_t ne1,
                                std::uint32_t seed, const ExecutionContext& ec,
                                const ops::PeerEvents& staged, const ops::PeerEvents& legacy,
                                const ops::PeerEvents& pipelined) {
    const std::size_t count = static_cast<std::size_t>(ne0) * static_cast<std::size_t>(ne1);
    std::vector<float> a(count), b(count);
    fill_uniform(a, seed, -8.0f, 8.0f);
    fill_uniform(b, seed + 1, -8.0f, 8.0f);
    const auto a_bits      = encode_bf16(a);
    const auto b_bits      = encode_bf16(b);
    int failures           = 0;
    const std::string base = disclosed(label);
    const auto staged_bits =
        captured_bits((base + " staged").c_str(), ne0, ne1, a_bits, b_bits, ec, staged, failures);
    const auto legacy_bits =
        captured_bits((base + " legacy").c_str(), ne0, ne1, a_bits, b_bits, ec, legacy, failures);
    const auto pipelined_bits = captured_bits((base + " pipelined").c_str(), ne0, ne1, a_bits,
                                              b_bits, ec, pipelined, failures);
    failures +=
        verify_exact((base + " legacy mailbox equals staged").c_str(), legacy_bits, staged_bits);
    failures += verify_exact((base + " pipelined mailbox equals staged").c_str(), pipelined_bits,
                             staged_bits);
    return failures;
}

// L2 prefetch hints (PeerMailbox::set_l2_prefetch, q27/dexch). Exchanges whose launches carry
// prefetching blocks -- four ranges of a weight-sized buffer, one unaligned and one empty, more
// blocks than the exchange's own at the decode shape -- must leave the staged path's bits on both
// ranks and must not touch the prefetched memory; a hint is consumed by one exchange (site 1 sets
// none and must still match).
int run_prefetch_identity_case(const char* label, std::int32_t ne0, std::int32_t ne1,
                               std::uint32_t seed, const ExecutionContext& ec,
                               const ops::PeerEvents& staged, const ops::PeerEvents& pipelined,
                               ops::PeerMailbox& mailbox) {
    constexpr std::size_t kWeightBytes = 24u << 20;
    const std::size_t count = static_cast<std::size_t>(ne0) * static_cast<std::size_t>(ne1);
    std::vector<float> a(count), b(count);
    fill_uniform(a, seed, -8.0f, 8.0f);
    fill_uniform(b, seed + 1, -8.0f, 8.0f);
    const auto a_bits = encode_bf16(a);
    const auto b_bits = encode_bf16(b);
    std::array<std::unique_ptr<GuardedDeviceBuffer>, 2> weights;
    for (int rank = 0; rank < 2; ++rank) {
        set_device(ec, rank);
        weights[static_cast<std::size_t>(rank)] =
            std::make_unique<GuardedDeviceBuffer>(kWeightBytes);
        weights[static_cast<std::size_t>(rank)]->fill(0x5a);
    }
    const auto hint = [&](int site) {
        if (site == 1) { return; }
        for (int rank = 0; rank < 2; ++rank) {
            auto* base =
                static_cast<const std::uint8_t*>(weights[static_cast<std::size_t>(rank)]->data());
            const std::array<ops::PeerL2Range, 4> ranges{{{base, 6u << 20},
                                                          {base + (12u << 20) + 3, (5u << 20) + 7},
                                                          {base + (20u << 20), 8},
                                                          {base + (18u << 20), 1u << 20}}};
            mailbox.set_l2_prefetch(rank, ranges);
        }
    };
    int failures = 0;
    const std::string base(label);
    const auto staged_bits =
        captured_bits((base + " staged").c_str(), ne0, ne1, a_bits, b_bits, ec, staged, failures);
    const auto prefetch_bits = captured_bits((base + " prefetch").c_str(), ne0, ne1, a_bits, b_bits,
                                             ec, pipelined, failures, hint);
    failures += verify_exact((base + " prefetching mailbox equals staged").c_str(), prefetch_bits,
                             staged_bits);
    const std::vector<std::uint8_t> pattern(kWeightBytes, 0x5a);
    for (int rank = 0; rank < 2; ++rank) {
        set_device(ec, rank);
        const auto& weight = *weights[static_cast<std::size_t>(rank)];
        failures += verify_exact(
            (base + " prefetched memory unchanged, device " + std::to_string(rank)).c_str(),
            from_device<std::uint8_t>(weight.data(), kWeightBytes), pattern);
        failures += weight.verify_guards((base + " prefetched memory guards").c_str());
    }
    return failures;
}

// One request's and a full batch's exchanges interleaved in one graph through a mailbox with
// wide slots ([5120, 8] ordinary, [5120, 32] wide, three of each): every exchange must take the
// mailbox (two kernel nodes each), each set alternating its own slots, and both buffers' bits must
// equal the staged path's on the same operands.
std::array<std::vector<std::uint16_t>, 2> mixed_width_bits(const ExecutionContext& ec,
                                                           const ops::PeerEvents& events,
                                                           std::size_t& nodes, int& failures) {
    constexpr int kSites = 3;
    constexpr std::array<std::int32_t, 2> kColumns{8, 32};
    std::array<std::vector<std::uint16_t>, 2> out;
    std::array<std::array<GuardedDeviceBuffer*, 2>, 2> buffers{};
    std::vector<std::unique_ptr<GuardedDeviceBuffer>> owned;
    std::array<std::array<Tensor, 2>, 2> buffer{};
    std::array<std::array<Tensor, 2>, 2> staging{};
    for (int w = 0; w < 2; ++w) {
        const std::size_t bytes = std::size_t{5120} * kColumns[w] * sizeof(std::uint16_t);
        for (int rank = 0; rank < 2; ++rank) {
            set_device(ec, rank);
            owned.push_back(std::make_unique<GuardedDeviceBuffer>(bytes));
            buffers[w][rank] = owned.back().get();
            owned.push_back(std::make_unique<GuardedDeviceBuffer>(bytes));
            owned.back()->fill(0);
            buffer[w][rank]  = Tensor(buffers[w][rank]->data(), DType::BF16, {5120, kColumns[w]});
            staging[w][rank] = Tensor(owned.back()->data(), DType::BF16, {5120, kColumns[w]});
            std::vector<float> values(std::size_t{5120} * kColumns[w]);
            fill_uniform(values, 501u + 2u * static_cast<std::uint32_t>(w) + rank, -8.0f, 8.0f);
            const auto bits = encode_bf16(values);
            buffers[w][rank]->copy_from_host(bits.data(), bytes);
        }
    }
    retire_staging(ec);
    const DecodeGraphPeerBridge bridge(ec.dev[0]->device, ec.dev[1]->device);
    DecodeGraphDefinition definition;
    capture_two_devices(ec, bridge, definition, [&] {
        for (int site = 0; site < kSites; ++site) {
            ops::allreduce_sum(buffer[0], staging[0], ec, events);
            ops::allreduce_sum(buffer[1], staging[1], ec, events);
        }
    });
    nodes = definition.node_count();
    DecodeGraphExecutable executable;
    executable.instantiate(definition);
    launch_two_devices(ec, executable);
    for (int w = 0; w < 2; ++w) {
        const std::size_t count = std::size_t{5120} * kColumns[w];
        set_device(ec, 0);
        out[w] = from_device<std::uint16_t>(buffers[w][0]->data(), count);
        set_device(ec, 1);
        failures += verify_exact("mixed widths device 1 equals device 0",
                                 from_device<std::uint16_t>(buffers[w][1]->data(), count), out[w]);
    }
    return out;
}

int run_mixed_width_case(const ExecutionContext& ec, const ops::PeerEvents& staged) {
    int failures = 0;
    ops::PeerMailbox mailbox(ec, 5120 * 8 * sizeof(std::uint16_t), 2,
                             ops::PeerExchangeKernel::Pipelined, 5120 * 32 * sizeof(std::uint16_t));
    ops::PeerEvents events(ec);
    events.attach_mailbox(&mailbox);
    std::size_t mailbox_nodes = 0;
    std::size_t staged_nodes  = 0;
    const auto mailbox_bits   = mixed_width_bits(ec, events, mailbox_nodes, failures);
    const auto staged_bits    = mixed_width_bits(ec, staged, staged_nodes, failures);
    if (mailbox_nodes != 12 || staged_nodes <= 12) {
        std::cerr << "mixed widths: " << mailbox_nodes << " / " << staged_nodes
                  << " graph nodes do not match the mailbox / staged transports\n";
        ++failures;
    }
    failures +=
        verify_exact("mixed widths ordinary slots equal staged", mailbox_bits[0], staged_bits[0]);
    failures +=
        verify_exact("mixed widths wide slots equal staged", mailbox_bits[1], staged_bits[1]);
    if (mailbox.hang_reported()) {
        std::cerr << "mixed widths: a mailbox exchange reported a hang\n";
        ++failures;
    }
    return failures;
}

// A peer that never arrives: only rank 0 enqueues its half. The poller must give up, report the
// hang and return before a display watchdog (about 2 s on Windows WDDM) would reset the device.
int run_mailbox_hang_case(const ExecutionContext& ec, ops::PeerExchangeKernel kernel) {
    constexpr std::size_t kBytes      = 256;
    constexpr double kWatchdogSeconds = 2.0;
    ops::PeerMailbox lonely(ec, kBytes, 2, kernel);
    set_device(ec, 0);
    GuardedDeviceBuffer buffer_0(kBytes);
    buffer_0.fill(0);
    cuda_check(cudaDeviceSynchronize(), "mailbox hang setup");
    const int slot   = lonely.take_capture_slot();
    const auto start = std::chrono::steady_clock::now();
    lonely.enqueue_exchange_sum(0, slot, buffer_0.data(), kBytes, ec.dev[0]->stream);
    cuda_check(cudaStreamSynchronize(ec.dev[0]->stream), "mailbox hang exchange");
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "mailbox hang (" << ops::peer_exchange_kernel_name(kernel)
              << " kernel) reported after " << seconds << " s\n";
    int failures = 0;
    if (!lonely.hang_reported()) {
        std::cerr << "mailbox hang: a missing peer was not reported\n";
        ++failures;
    }
    if (seconds > kWatchdogSeconds) {
        std::cerr << "mailbox hang: the poller gave up after " << seconds
                  << " s, past the display watchdog\n";
        ++failures;
    }
    return failures;
}

// ---- Direct peer pull -------------------------------------------------------------------------

// NINFER_TP_PEER_PULL=1 for the scope's lifetime, then the caller's value again (unset when it
// was), so the process selection survives.
class ScopedPeerPullRequest {
public:
    ScopedPeerPullRequest() {
        const char* prior = std::getenv(kVariable);
        had_prior_        = prior != nullptr;
        if (had_prior_) { prior_ = prior; }
        ::setenv(kVariable, "1", 1);
    }

    ~ScopedPeerPullRequest() {
        if (had_prior_) {
            ::setenv(kVariable, prior_.c_str(), 1);
        } else {
            ::unsetenv(kVariable);
        }
    }

    ScopedPeerPullRequest(const ScopedPeerPullRequest&)            = delete;
    ScopedPeerPullRequest& operator=(const ScopedPeerPullRequest&) = delete;

private:
    static constexpr const char* kVariable = "NINFER_TP_PEER_PULL";
    bool had_prior_                        = false;
    std::string prior_;
};

// A PeerEvents with the direct peer pull requested: its constructor reads the variable.
ops::PeerEvents make_peer_pull_events(const ExecutionContext& ec) {
    const ScopedPeerPullRequest request;
    return ops::PeerEvents(ec);
}

// One launch with the Program's gates (models/qwen3_5/program/graph_execution.h).
void launch_gated(const ExecutionContext& ec, const DecodeGraphPeerBridge& bridge,
                  DecodeGraphExecutable& executable) {
    set_device(ec, 0);
    bridge.gate_launch(ec.dev[1]->stream, ec.dev[0]->stream);
    executable.launch(ec.dev[0]->stream);
    bridge.gate_peer_after_launch(ec.dev[1]->stream, ec.dev[0]->stream);
}

// Signed zeros, subnormals, infinities, NaNs, the largest finite values (whose sums overflow) and
// ordinary values: the first 256 elements pair each of these 16 encodings with each other one,
// the rest are hashed bit patterns over all BF16 encodings.
std::vector<std::uint16_t> encoding_inputs(std::size_t count, std::uint32_t seed, int rank) {
    constexpr std::uint16_t kSpecial[16] = {0x0000, 0x8000, 0x0001, 0x8001, 0x7f80, 0xff80,
                                            0x7fc1, 0x7f81, 0x7fff, 0xffff, 0x7f7f, 0xff7f,
                                            0x3f80, 0xbf80, 0x3b80, 0xbb80};
    std::vector<std::uint16_t> bits(count);
    for (std::size_t i = 0; i < count; ++i) {
        if (i < 256) {
            const std::size_t shift = (seed % 16) * (rank != 0 ? 7 : 3) + (rank != 0 ? i / 16 : 0);
            bits[i]                 = kSpecial[(i + shift) % 16];
            continue;
        }
        std::uint32_t x = static_cast<std::uint32_t>(i) * 1664525u + seed * 1013904223u +
                          static_cast<std::uint32_t>(rank) * 747796405u;
        x ^= x >> 16;
        bits[i] = static_cast<std::uint16_t>(x);
    }
    return bits;
}

// The Op's definition, independently: one FP32 addition of the represented BF16 inputs, rounded
// once to BF16 (round-to-nearest-even).
std::uint16_t bf16_sum_oracle(std::uint16_t a, std::uint16_t b) {
    return f32_to_bf16(bf16_to_f32(a) + bf16_to_f32(b));
}

bool bf16_is_nan(std::uint16_t bits) { return (bits & 0x7fffu) > 0x7f80u; }

// Exact, except that the definition leaves NaN payloads open: any NaN matches a NaN.
int verify_bf16_oracle(const std::string& label, const std::vector<std::uint16_t>& got,
                       const std::vector<std::uint16_t>& expected) {
    if (got.size() != expected.size()) {
        std::cerr << label << ": size mismatch\n";
        return 1;
    }
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (got[i] != expected[i] && !(bf16_is_nan(got[i]) && bf16_is_nan(expected[i]))) {
            std::cerr << label << ": oracle mismatch at index " << i << ": 0x" << std::hex << got[i]
                      << " expected 0x" << expected[i] << std::dec << '\n';
            return 1;
        }
    }
    return 0;
}

// One all-reduce's guarded buffer and staging on each rank.
struct AllreducePair {
    std::size_t count = 0;
    std::size_t bytes = 0;
    std::vector<std::unique_ptr<GuardedDeviceBuffer>> storage; // buffer, staging of rank 0, 1
    std::array<Tensor, 2> buffer{};
    std::array<Tensor, 2> staging{};

    AllreducePair(const ExecutionContext& ec, std::int32_t ne0, std::int32_t ne1)
        : count(static_cast<std::size_t>(ne0) * static_cast<std::size_t>(ne1)),
          bytes(count * sizeof(std::uint16_t)) {
        for (int rank = 0; rank < 2; ++rank) {
            set_device(ec, rank);
            storage.push_back(std::make_unique<GuardedDeviceBuffer>(bytes));
            storage.push_back(std::make_unique<GuardedDeviceBuffer>(bytes));
            storage.back()->fill(0);
            buffer[rank]  = Tensor(storage[2 * rank]->data(), DType::BF16, {ne0, ne1});
            staging[rank] = Tensor(storage[2 * rank + 1]->data(), DType::BF16, {ne0, ne1});
        }
    }

    void upload(const ExecutionContext& ec, const std::array<std::vector<std::uint16_t>, 2>& bits) {
        for (int rank = 0; rank < 2; ++rank) {
            set_device(ec, rank);
            storage[2 * rank]->copy_from_host(bits[rank].data(), bytes);
        }
    }

    // Rank 0's result, after checking that rank 1 holds the same bits and no guard was hit.
    std::vector<std::uint16_t> agreed(const ExecutionContext& ec, const std::string& label,
                                      int& failures) const {
        set_device(ec, 0);
        auto bits = from_device<std::uint16_t>(storage[0]->data(), count);
        set_device(ec, 1);
        failures += verify_exact((label + " device 1 equals device 0").c_str(),
                                 from_device<std::uint16_t>(storage[2]->data(), count), bits);
        for (int rank = 0; rank < 2; ++rank) {
            set_device(ec, rank);
            failures += storage[2 * rank]->verify_guards(label + " buffer");
            failures += storage[2 * rank + 1]->verify_guards(label + " staging");
        }
        return bits;
    }
};

// The peer pull against the staged path and the oracle, eagerly and in a two-device graph. A
// payload it does not serve runs the other routes through the same events: the same bits.
int run_peer_pull_identity_case(const char* case_label, std::int32_t ne0, std::int32_t ne1,
                                std::uint32_t seed, const ExecutionContext& ec,
                                const ops::PeerEvents& staged, const ops::PeerEvents& pull) {
    const std::string label = disclosed(case_label);
    const std::size_t count = static_cast<std::size_t>(ne0) * static_cast<std::size_t>(ne1);
    const std::array<std::vector<std::uint16_t>, 2> inputs{encoding_inputs(count, seed, 0),
                                                           encoding_inputs(count, seed, 1)};
    std::vector<std::uint16_t> expected(count);
    for (std::size_t i = 0; i < count; ++i) {
        expected[i] = bf16_sum_oracle(inputs[0][i], inputs[1][i]);
    }
    const std::array<const ops::PeerEvents*, 2> transports{&staged, &pull};
    int failures = 0;
    for (const bool captured : {false, true}) {
        const std::string mode = std::string(label) + (captured ? " captured" : " eager");
        std::array<std::vector<std::uint16_t>, 2> bits;
        for (std::size_t t = 0; t < 2; ++t) {
            AllreducePair pair(ec, ne0, ne1);
            pair.upload(ec, inputs);
            retire_staging(ec);
            const auto call = [&] {
                ops::allreduce_sum(pair.buffer, pair.staging, ec, *transports[t]);
            };
            if (captured) {
                const DecodeGraphPeerBridge bridge(ec.dev[0]->device, ec.dev[1]->device);
                DecodeGraphDefinition definition;
                capture_two_devices(ec, bridge, definition, call);
                DecodeGraphExecutable executable;
                executable.instantiate(definition);
                launch_two_devices(ec, executable);
            } else {
                call();
                synchronize_both(ec);
            }
            bits[t] = pair.agreed(ec, mode + (t == 0 ? " staged" : " peer pull"), failures);
        }
        failures += verify_exact((mode + " peer pull equals staged").c_str(), bits[1], bits[0]);
        failures += verify_bf16_oracle(mode + " peer pull oracle", bits[1], expected);
    }
    return failures;
}

// A decode round's mix on one PeerEvents: one graph of [5120] (two blocks of the peer pull),
// [5120, 6] (eight blocks), [5120, 16] (160 KiB: staged; with the flags, paired copies of twenty
// blocks) and [5120] again, launched with the
// Program's gates (models/qwen3_5/program/graph_execution.h), then two eager all-reduces and a
// second gated launch, with no host synchronization in between. A captured memset holds rank 1
// back and an eager one rank 0, so the ranks reach the exchanges out of step, and each block's
// epoch counter advances differently per size: a mispaired epoch combines the wrong operands.
// Fresh inputs of every encoding class each round. After a buffer's first site both ranks hold s,
// so each later site of that buffer yields oracle(s, s).
int run_peer_pull_mixed_case(const char* case_label, const ExecutionContext& ec,
                             const ops::PeerEvents& events) {
    const std::string label          = disclosed(case_label);
    constexpr int kRounds            = 3;
    constexpr std::size_t kSkewBytes = 64u << 20;
    AllreducePair small(ec, 5120, 1), medium(ec, 5120, 6), large(ec, 5120, 16);
    const std::array<AllreducePair*, 3> pairs{&small, &medium, &large};
    const std::array<const char*, 3> names{" [5120]", " [5120,6]", " [5120,16]"};
    const std::array<int, 3> sites{5, 3, 2}; // per round: graph, eager, graph
    set_device(ec, 0);
    DeviceBuffer skew_0(kSkewBytes);
    set_device(ec, 1);
    DeviceBuffer skew_1(kSkewBytes);
    retire_staging(ec);

    const DecodeGraphPeerBridge bridge(ec.dev[0]->device, ec.dev[1]->device);
    DecodeGraphDefinition definition;
    capture_two_devices(ec, bridge, definition, [&] {
        set_device(ec, 1);
        cuda_check(cudaMemsetAsync(skew_1.p, 0, skew_1.bytes, ec.dev[1]->stream), "skew memset");
        ops::allreduce_sum(small.buffer, small.staging, ec, events);
        ops::allreduce_sum(medium.buffer, medium.staging, ec, events);
        ops::allreduce_sum(large.buffer, large.staging, ec, events);
        ops::allreduce_sum(small.buffer, small.staging, ec, events);
    });
    DecodeGraphExecutable executable;
    executable.instantiate(definition);
    const auto gated_launch = [&] {
        set_device(ec, 0);
        bridge.gate_launch(ec.dev[1]->stream, ec.dev[0]->stream);
        executable.launch(ec.dev[0]->stream);
        bridge.gate_peer_after_launch(ec.dev[1]->stream, ec.dev[0]->stream);
    };

    int failures = 0;
    for (int round = 0; round < kRounds; ++round) {
        std::array<std::array<std::vector<std::uint16_t>, 2>, 3> inputs;
        for (std::size_t p = 0; p < 3; ++p) {
            const auto seed =
                701u + 16u * static_cast<std::uint32_t>(round) + 2u * static_cast<std::uint32_t>(p);
            inputs[p] = {encoding_inputs(pairs[p]->count, seed, 0),
                         encoding_inputs(pairs[p]->count, seed, 1)};
            pairs[p]->upload(ec, inputs[p]);
        }
        retire_staging(ec);
        gated_launch();
        set_device(ec, 0);
        cuda_check(cudaMemsetAsync(skew_0.p, round & 0xff, skew_0.bytes, ec.dev[0]->stream),
                   "skew memset");
        ops::allreduce_sum(small.buffer, small.staging, ec, events);
        ops::allreduce_sum(medium.buffer, medium.staging, ec, events);
        gated_launch();
        synchronize_both(ec);

        for (std::size_t p = 0; p < 3; ++p) {
            std::vector<std::uint16_t> expected(pairs[p]->count);
            for (std::size_t i = 0; i < expected.size(); ++i) {
                std::uint16_t value = bf16_sum_oracle(inputs[p][0][i], inputs[p][1][i]);
                for (int site = 1; site < sites[p]; ++site) {
                    value = bf16_sum_oracle(value, value);
                }
                expected[i] = value;
            }
            const std::string name =
                std::string(label) + " round " + std::to_string(round) + names[p];
            const auto bits = pairs[p]->agreed(ec, name, failures);
            failures += verify_bf16_oracle(name + " oracle", bits, expected);
        }
    }
    return failures;
}

// Ownership of the peer-pull flags. A peer-pull PeerEvents move-assigned over another (which then
// frees the other's flags) and move-constructed again keeps serving with its epochs; the
// moved-from instance is dead; construction and destruction leave the current device as found.
int run_peer_pull_ownership_case(const ExecutionContext& ec, bool peer_access,
                                 const ops::PeerEvents& staged) {
    const auto current_device = [] {
        int device = -1;
        cuda_check(cudaGetDevice(&device), "cudaGetDevice");
        return device;
    };
    int failures = 0;
    set_device(ec, 1);
    const int caller           = current_device();
    ops::PeerEvents keeper     = make_peer_pull_events(ec);
    const ops::PeerPull* flags = nullptr;
    {
        ops::PeerEvents donor = make_peer_pull_events(ec);
        flags                 = donor.peer_pull();
        keeper                = std::move(donor);
    }
    if (current_device() != caller) {
        std::cerr
            << "peer pull ownership: construction or destruction changed the current device\n";
        ++failures;
    }
    if ((flags != nullptr) != peer_access) {
        std::cerr << "peer pull ownership: flags " << (flags != nullptr ? "exist" : "are missing")
                  << " although direct P2P is " << (peer_access ? "enabled" : "unavailable")
                  << '\n';
        ++failures;
    }
    ops::PeerEvents moved(std::move(keeper));
    if (moved.peer_pull() != flags || keeper.peer_pull() != nullptr || keeper.live()) {
        std::cerr << "peer pull ownership: a move did not transfer the flags\n";
        ++failures;
    }
    failures +=
        run_peer_pull_identity_case("moved peer pull [5120]", 5120, 1, 801u, ec, staged, moved);
    failures +=
        run_peer_pull_identity_case("moved peer pull [5120,6]", 5120, 6, 803u, ec, staged, moved);
    return failures;
}

// Requested without direct P2P in both directions, the peer pull must refuse construction rather
// than keep native captured peer copies.
int run_peer_pull_requires_p2p_case(const ExecutionContext& ec) {
    try {
        const ops::PeerEvents events = make_peer_pull_events(ec);
    } catch (const std::runtime_error&) {
        std::cout << "peer pull: refused without direct P2P\n";
        return 0;
    }
    std::cerr << "peer pull: requested without direct P2P, construction succeeded\n";
    return 1;
}

std::vector<std::uint8_t> pattern_bytes(std::size_t count, std::uint32_t seed) {
    std::vector<std::uint8_t> bits(count);
    std::uint32_t x = seed * 2654435761u + 1u;
    for (std::uint8_t& bit : bits) {
        x   = x * 1664525u + 1013904223u;
        bit = static_cast<std::uint8_t>(x >> 24);
    }
    return bits;
}

// One captured paired copy's operands: rank r receives bytes[r] from source[1 - r] (on the peer)
// into destination[r], every operand `offset` bytes into its guarded buffer; a direction of 0
// bytes passes null operands.
struct PeerCopyOperands {
    std::array<std::size_t, 2> bytes{};
    std::size_t offset = 0;
    std::array<std::unique_ptr<GuardedDeviceBuffer>, 2> source;
    std::array<std::unique_ptr<GuardedDeviceBuffer>, 2> destination;
    std::array<std::vector<std::uint8_t>, 2> sent; // what source[r] holds
    std::array<const void*, 2> from{nullptr, nullptr};
    std::array<void*, 2> to{nullptr, nullptr};

    PeerCopyOperands(const ExecutionContext& ec, std::array<std::size_t, 2> received,
                     std::size_t shift)
        : bytes(received), offset(shift) {
        for (int rank = 0; rank < 2; ++rank) {
            const auto r = static_cast<std::size_t>(rank);
            set_device(ec, rank);
            source[r]      = std::make_unique<GuardedDeviceBuffer>(bytes[1 - r] + offset);
            destination[r] = std::make_unique<GuardedDeviceBuffer>(bytes[r] + offset);
        }
        for (std::size_t r = 0; r < 2; ++r) {
            if (bytes[r] == 0) { continue; }
            from[r] = static_cast<const std::uint8_t*>(source[1 - r]->data()) + offset;
            to[r]   = static_cast<std::uint8_t*>(destination[r]->data()) + offset;
        }
    }

    // Fresh source bytes, destinations filled with 0xcd, both devices retired.
    void upload(const ExecutionContext& ec, std::uint32_t seed) {
        for (int rank = 0; rank < 2; ++rank) {
            const auto r = static_cast<std::size_t>(rank);
            set_device(ec, rank);
            sent[r] = pattern_bytes(bytes[1 - r], seed + static_cast<std::uint32_t>(rank));
            source[r]->copy_from_host(sent[r].data(), sent[r].size(), offset);
            destination[r]->fill(0xcd);
        }
        retire_staging(ec);
    }

    // Exact: each destination holds the peer's bytes (its leading `offset` bytes untouched), each
    // source is unchanged, every guard intact.
    int verify(const ExecutionContext& ec, const std::string& label) const {
        int failures = 0;
        for (int rank = 0; rank < 2; ++rank) {
            const auto r           = static_cast<std::size_t>(rank);
            const std::string name = label + " device " + std::to_string(rank);
            set_device(ec, rank);
            std::vector<std::uint8_t> received(bytes[r] + offset);
            destination[r]->copy_to_host(received.data(), received.size());
            std::vector<std::uint8_t> expected(offset, 0xcd);
            expected.insert(expected.end(), sent[1 - r].begin(), sent[1 - r].end());
            failures += verify_exact((name + " received").c_str(), received, expected);
            std::vector<std::uint8_t> kept(sent[r].size());
            source[r]->copy_to_host(kept.data(), kept.size(), offset);
            failures += verify_exact((name + " source unchanged").c_str(), kept, sent[r]);
            failures += destination[r]->verify_guards(name + " destination");
            failures += source[r]->verify_guards(name + " source");
        }
        return failures;
    }
};

// Captured paired copies, exact. One graph -- a captured memset holding rank 1 back, then the pair
// -- must hold the memset and one kernel per device (no memcpy node), and is launched through the
// Program's gates kReplays times with fresh bytes, then kBackToBack times with no host step.
int run_peer_copy_pair_case(const char* case_label, std::size_t bytes0, std::size_t bytes1,
                            std::size_t offset, const ExecutionContext& ec,
                            const ops::PeerEvents& events) {
    constexpr int kReplays           = 4;
    constexpr int kBackToBack        = 3;
    constexpr std::size_t kSkewBytes = 64u << 20;
    const std::string label          = disclosed(case_label);
    PeerCopyOperands copy(ec, {bytes0, bytes1}, offset);
    set_device(ec, 1);
    DeviceBuffer skew(kSkewBytes);
    retire_staging(ec);

    const DecodeGraphPeerBridge bridge(ec.dev[0]->device, ec.dev[1]->device);
    DecodeGraphDefinition definition;
    bool served = false;
    capture_two_devices(ec, bridge, definition, [&] {
        set_device(ec, 1);
        cuda_check(cudaMemsetAsync(skew.p, 0, skew.bytes, ec.dev[1]->stream), "skew memset");
        served = ops::captured_peer_copy_pair(copy.from, copy.to, copy.bytes, ec, events);
    });
    int failures = 0;
    if (!served || definition.node_count() != 3) {
        std::cerr << label << ": the paired copy did not take the capture ("
                  << definition.node_count() << " graph nodes)\n";
        ++failures;
    }
    DecodeGraphExecutable executable;
    executable.instantiate(definition);
    for (int replay = 0; replay < kReplays; ++replay) {
        copy.upload(ec, 1101u + 4u * static_cast<std::uint32_t>(replay));
        launch_gated(ec, bridge, executable);
        synchronize_both(ec);
        failures += copy.verify(ec, label + " replay " + std::to_string(replay));
    }
    copy.upload(ec, 1131u);
    for (int launch = 0; launch < kBackToBack; ++launch) { launch_gated(ec, bridge, executable); }
    synchronize_both(ec);
    failures += copy.verify(ec, label + " back-to-back launches");
    return failures;
}

// In-place update, as the Program moves one profile's executable to another's: a graph of one
// paired copy is instantiated and launched, then replaced through DecodeGraphExecutable::update by
// a graph of the same shape over other buffers and sizes (another block count); the updated
// executable's copies must be exact, launched twice.
int run_peer_copy_update_case(const ExecutionContext& ec, const ops::PeerEvents& events) {
    const std::string label = disclosed("paired copy graph update");
    PeerCopyOperands first(ec, {10240, 10240}, 0);
    PeerCopyOperands second(ec, {1489920, 30722}, 0);
    retire_staging(ec);
    const DecodeGraphPeerBridge bridge(ec.dev[0]->device, ec.dev[1]->device);
    int failures       = 0;
    const auto capture = [&](PeerCopyOperands& copy, DecodeGraphDefinition& definition) {
        bool served = false;
        capture_two_devices(ec, bridge, definition, [&] {
            served = ops::captured_peer_copy_pair(copy.from, copy.to, copy.bytes, ec, events);
        });
        if (!served || definition.node_count() != 2) {
            std::cerr << label << ": the paired copy did not take the capture ("
                      << definition.node_count() << " graph nodes)\n";
            ++failures;
        }
    };
    DecodeGraphDefinition first_definition;
    DecodeGraphDefinition second_definition;
    capture(first, first_definition);
    capture(second, second_definition);
    DecodeGraphExecutable executable;
    executable.instantiate(first_definition);
    first.upload(ec, 1201u);
    launch_gated(ec, bridge, executable);
    synchronize_both(ec);
    failures += first.verify(ec, label + " before");
    executable.update(second_definition);
    for (int launch = 0; launch < 2; ++launch) {
        second.upload(ec, 1211u + 4u * static_cast<std::uint32_t>(launch));
        launch_gated(ec, bridge, executable);
        synchronize_both(ec);
        failures += second.verify(ec, label + " after, launch " + std::to_string(launch));
    }
    return failures;
}

// Source publication and read completion behind a GPU producer. One graph of kStages stages, each
// an all-reduce that rewrites both ranks' buffer (at [5120] the peer pull's reduction kernel and
// copy-back; at [5121] the staged combine kernel after a paired copy) immediately followed by a
// paired copy of each rank's buffer into a snapshot of the stage on the peer, and a one-sided copy
// of rank 1's buffer to rank 0; then a captured all-gather of the two buffers. A captured memset
// holds rank 1 back. After stage k both ranks hold s_k = oracle(s_(k-1), s_(k-1)), s_0 =
// oracle(a, b): a copy that read its source before the producer finished delivers s_(k-1), one
// whose source the next stage overwrote before the read finished delivers s_(k+1). Fresh inputs
// per replay through the Program's gates, then back-to-back launches.
int run_peer_copy_publication_case(const char* case_label, std::int32_t n,
                                   const ExecutionContext& ec, const ops::PeerEvents& events) {
    constexpr int kStages            = 5;
    constexpr int kReplays           = 4;
    constexpr int kBackToBack        = 2;
    constexpr std::size_t kSkewBytes = 64u << 20;
    const std::string label          = disclosed(case_label);
    const auto count                 = static_cast<std::size_t>(n);
    const std::size_t bytes          = count * sizeof(std::uint16_t);

    AllreducePair pair(ec, n, 1);
    std::vector<std::unique_ptr<GuardedDeviceBuffer>> owned;
    const auto allocate = [&](int rank, std::size_t size) {
        set_device(ec, rank);
        owned.push_back(std::make_unique<GuardedDeviceBuffer>(size));
        owned.back()->fill(0);
        return owned.back()->data();
    };
    std::array<std::array<void*, kStages>, 2> snapshot{}; // on rank r: the peer's s_k
    std::array<void*, kStages> one_sided{};               // on rank 0: rank 1's s_k
    for (int k = 0; k < kStages; ++k) {
        for (int rank = 0; rank < 2; ++rank) { snapshot[rank][k] = allocate(rank, bytes); }
        one_sided[k] = allocate(0, bytes);
    }
    std::array<Tensor, 2> gathered{};
    std::array<Tensor, 2> part{};
    for (int rank = 0; rank < 2; ++rank) {
        gathered[rank] = Tensor(allocate(rank, 2 * bytes), DType::BF16, {n, 2});
        part[rank]     = Tensor(pair.buffer[rank].data, DType::BF16, {n, 1});
    }
    set_device(ec, 1);
    DeviceBuffer skew(kSkewBytes);
    retire_staging(ec);

    const DecodeGraphPeerBridge bridge(ec.dev[0]->device, ec.dev[1]->device);
    DecodeGraphDefinition definition;
    bool served = true;
    capture_two_devices(ec, bridge, definition, [&] {
        set_device(ec, 1);
        cuda_check(cudaMemsetAsync(skew.p, 0, skew.bytes, ec.dev[1]->stream), "skew memset");
        for (int k = 0; k < kStages; ++k) {
            ops::allreduce_sum(pair.buffer, pair.staging, ec, events);
            served = ops::captured_peer_copy_pair({pair.buffer[1].data, pair.buffer[0].data},
                                                  {snapshot[0][k], snapshot[1][k]}, {bytes, bytes},
                                                  ec, events) &&
                     served;
            served =
                ops::captured_peer_copy_pair({pair.buffer[1].data, nullptr},
                                             {one_sided[k], nullptr}, {bytes, 0}, ec, events) &&
                served;
        }
        ops::allgather_rows(gathered, part, ec, events);
    });
    int failures = 0;
    if (!served) {
        std::cerr << label << ": a paired copy did not take the capture\n";
        ++failures;
    }
    DecodeGraphExecutable executable;
    executable.instantiate(definition);

    // The stage values of one launch whose ranks start from (a, b).
    const auto stages_of = [&](const std::vector<std::uint16_t>& a,
                               const std::vector<std::uint16_t>& b) {
        std::vector<std::vector<std::uint16_t>> s(kStages, std::vector<std::uint16_t>(count));
        for (std::size_t i = 0; i < count; ++i) { s[0][i] = bf16_sum_oracle(a[i], b[i]); }
        for (int k = 1; k < kStages; ++k) {
            for (std::size_t i = 0; i < count; ++i) {
                s[k][i] = bf16_sum_oracle(s[k - 1][i], s[k - 1][i]);
            }
        }
        return s;
    };
    const auto verify_launch = [&](const std::string& name,
                                   const std::vector<std::vector<std::uint16_t>>& s) {
        for (int k = 0; k < kStages; ++k) {
            const std::string stage = name + " stage " + std::to_string(k);
            for (int rank = 0; rank < 2; ++rank) {
                set_device(ec, rank);
                failures +=
                    verify_exact((stage + " device " + std::to_string(rank) + " snapshot").c_str(),
                                 from_device<std::uint16_t>(snapshot[rank][k], count), s[k]);
            }
            set_device(ec, 0);
            failures += verify_exact((stage + " one-sided").c_str(),
                                     from_device<std::uint16_t>(one_sided[k], count), s[k]);
        }
        const auto buffer = pair.agreed(ec, name, failures);
        failures += verify_exact((name + " buffer").c_str(), buffer, s.back());
        std::vector<std::uint16_t> image(s.back());
        image.insert(image.end(), s.back().begin(), s.back().end());
        for (int rank = 0; rank < 2; ++rank) {
            set_device(ec, rank);
            failures +=
                verify_exact((name + " gathered device " + std::to_string(rank)).c_str(),
                             from_device<std::uint16_t>(gathered[rank].data, 2 * count), image);
        }
    };

    std::vector<std::uint16_t> last;
    for (int replay = 0; replay < kReplays; ++replay) {
        std::array<std::vector<std::uint16_t>, 2> inputs;
        for (int rank = 0; rank < 2; ++rank) {
            std::vector<float> values(count);
            fill_uniform(values,
                         1301u + 2u * static_cast<std::uint32_t>(replay) +
                             static_cast<std::uint32_t>(rank),
                         -8.0f, 8.0f);
            inputs[static_cast<std::size_t>(rank)] = encode_bf16(values);
        }
        pair.upload(ec, inputs);
        retire_staging(ec);
        launch_gated(ec, bridge, executable);
        synchronize_both(ec);
        const auto s = stages_of(inputs[0], inputs[1]);
        verify_launch(label + " replay " + std::to_string(replay), s);
        last = s.back();
    }
    std::vector<std::vector<std::uint16_t>> s;
    for (int launch = 0; launch < kBackToBack; ++launch) {
        launch_gated(ec, bridge, executable);
        s    = stages_of(last, last);
        last = s.back();
    }
    synchronize_both(ec);
    verify_launch(label + " back-to-back launches", s);
    for (const auto& buffer : owned) { failures += buffer->verify_guards(label + " snapshot"); }
    return failures;
}

// Core's guard under NINFER_TP_PEER_PULL=1: inside a capture, a copy between two devices through
// memcpy_device_async enqueues nothing and reports cudaErrorNotSupported; a local copy is
// captured as before.
int run_core_capture_guard_case(const ExecutionContext& ec) {
    constexpr std::size_t kBytes = 4096;
    set_device(ec, 1);
    GuardedDeviceBuffer remote(kBytes);
    remote.fill(0x11);
    set_device(ec, 0);
    GuardedDeviceBuffer local(kBytes), copied(kBytes);
    local.fill(0x22);
    copied.fill(0);
    retire_staging(ec);
    cudaError_t foreign = cudaSuccess;
    cudaError_t own     = cudaErrorUnknown;
    int failures        = 0;
    {
        const ScopedPeerPullRequest request;
        DecodeGraphDefinition definition;
        set_device(ec, 0);
        definition.capture(ec.dev[0]->stream, [&] {
            foreign = memcpy_device_async(copied.data(), remote.data(), kBytes, ec.dev[0]->stream);
            own     = memcpy_device_async(copied.data(), local.data(), kBytes, ec.dev[0]->stream);
        });
        if (definition.node_count() != 1) {
            std::cerr << "core capture guard: " << definition.node_count()
                      << " graph nodes, expected the local copy alone\n";
            ++failures;
        }
        DecodeGraphExecutable executable;
        executable.instantiate(definition);
        executable.launch(ec.dev[0]->stream);
        synchronize_both(ec);
    }
    if (foreign != cudaErrorNotSupported || own != cudaSuccess) {
        std::cerr << "core capture guard: cross-device copy returned " << cudaGetErrorName(foreign)
                  << ", local copy " << cudaGetErrorName(own) << '\n';
        ++failures;
    }
    set_device(ec, 0);
    failures += verify_exact("core capture guard local copy",
                             from_device<std::uint8_t>(copied.data(), kBytes),
                             std::vector<std::uint8_t>(kBytes, 0x22));
    failures += copied.verify_guards("core capture guard destination");
    return failures;
}

// The direct peer pull and its captured paired copies. Requested without direct P2P, PeerEvents
// must refuse construction; with it, every case below runs on the flags.
int run_peer_pull_cases(const ExecutionContext& ec, const ops::PeerEvents& events,
                        bool peer_access) {
    if (!peer_access) { return run_peer_pull_requires_p2p_case(ec); }
    int failures                      = 0;
    const ops::PeerEvents pull_events = make_peer_pull_events(ec);
    std::cout << "peer pull: active\n";
    if (pull_events.peer_pull() == nullptr) {
        std::cerr << "peer pull: no flags although direct P2P is enabled\n";
        ++failures;
    }
    // Independent unrounded FP64 mathematical oracle, in addition to the exact
    // native/cast-reference comparisons over every encoding class below.
    failures += run_allreduce_case("peer pull FP64 [5120]", 5120, 1, 1001u, ec, pull_events);
    failures += run_allreduce_case("peer pull FP64 [5120,6]", 5120, 6, 1003u, ec, pull_events);
    // Served: the decode row (10 KiB), six tokens (60 KiB), the 64 KiB bound, one vector. Left to
    // the staged path (captured: the paired copies): 128 bytes past the bound, a 48-token chunk
    // (480 KiB), the 1024-token prefill chunk (10 MiB) and a payload that is not whole 16-byte
    // vectors.
    failures +=
        run_peer_pull_identity_case("peer pull [5120]", 5120, 1, 901u, ec, events, pull_events);
    failures +=
        run_peer_pull_identity_case("peer pull [5120,6]", 5120, 6, 903u, ec, events, pull_events);
    failures += run_peer_pull_identity_case("peer pull [4096,8] bound", 4096, 8, 905u, ec, events,
                                            pull_events);
    failures +=
        run_peer_pull_identity_case("peer pull [8,1] minimal", 8, 1, 907u, ec, events, pull_events);
    failures += run_peer_pull_identity_case("peer pull [4104,8] past the bound", 4104, 8, 909u, ec,
                                            events, pull_events);
    failures +=
        run_peer_pull_identity_case("peer pull [5120,48]", 5120, 48, 911u, ec, events, pull_events);
    failures += run_peer_pull_identity_case("peer pull [5120,1024]", 5120, 1024, 913u, ec, events,
                                            pull_events);
    failures += run_peer_pull_identity_case("peer pull [5121] partial vector", 5121, 1, 915u, ec,
                                            events, pull_events);
    // Reuse without host synchronization: interleaved with the event-ordered all-gather, replayed
    // with fresh inputs, back-to-back launches and an eager call after them.
    failures += run_chained_case("chained peer pull", ec, pull_events);
    failures +=
        run_captured_case("captured peer pull [5120]", 5120, 1, 5, false, ec, pull_events, nullptr);
    failures += run_captured_case("captured peer pull [5120,6]", 5120, 6, 5, false, ec, pull_events,
                                  nullptr);
    // The original odd-length staged case, now on the paired copies.
    failures += run_captured_case("captured paired copies [5121]", 5121, 1, 3, false, ec,
                                  pull_events, nullptr);
    failures += run_peer_pull_mixed_case("mixed staged", ec, events);
    failures += run_peer_pull_mixed_case("mixed peer pull", ec, pull_events);
    // An attached mailbox keeps the captured calls that fit its slot (one node per device and
    // site); the peer pull takes the wider ones. An attached HostExchange keeps the eager calls.
    ops::PeerMailbox pull_mailbox(ec, 5120 * 4 * sizeof(std::uint16_t));
    ops::PeerEvents pull_mailbox_events = make_peer_pull_events(ec);
    pull_mailbox_events.attach_mailbox(&pull_mailbox);
    failures += run_captured_case("captured mailbox before peer pull [5120]", 5120, 1, 3, true, ec,
                                  pull_mailbox_events, &pull_mailbox);
    failures += run_captured_case("captured peer pull past the mailbox slot [5120,6]", 5120, 6, 3,
                                  false, ec, pull_mailbox_events, &pull_mailbox);
    ops::HostExchange pull_exchange(ec);
    ops::PeerEvents pull_exchange_events = make_peer_pull_events(ec);
    pull_exchange_events.attach_host_exchange(&pull_exchange);
    failures += run_captured_case("captured peer pull, eager host exchange [5120]", 5120, 1, 3,
                                  false, ec, pull_exchange_events, nullptr);
    failures += run_peer_pull_ownership_case(ec, peer_access, events);

    // Captured paired copies: the decode row both ways, odd sizes, one direction of the logits
    // gather at six columns ([124160, 6] BF16) and of a large odd payload, unaligned operands
    // (byte route), a single byte.
    failures +=
        run_peer_copy_pair_case("paired copy [10240 B, 10240 B]", 10240, 10240, 0, ec, pull_events);
    failures += run_peer_copy_pair_case("paired copy odd [10242 B, 10241 B]", 10242, 10241, 0, ec,
                                        pull_events);
    failures += run_peer_copy_pair_case("paired copy rank 0 only [1489920 B]", 1489920, 0, 0, ec,
                                        pull_events);
    failures += run_peer_copy_pair_case("paired copy rank 1 only [3 MiB + 7 B]", 0,
                                        (std::size_t{3} << 20) + 7, 0, ec, pull_events);
    failures += run_peer_copy_pair_case("paired copy unaligned [5121 B, 33 B]", 5121, 33, 1, ec,
                                        pull_events);
    failures += run_peer_copy_pair_case("paired copy [1 B, 0 B]", 1, 0, 0, ec, pull_events);
    failures += run_peer_copy_update_case(ec, pull_events);
    // GPU-produced operands, the fused reduction and the paired copies on the same flags.
    failures +=
        run_peer_copy_publication_case("paired copy publication [5120]", 5120, ec, pull_events);
    failures +=
        run_peer_copy_publication_case("paired copy publication [5121]", 5121, ec, pull_events);

    run_captured_microbenchmark("peer pull", ec, pull_events);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count < 2) {
        std::cout << "SKIP: cross-device collectives require two CUDA devices, found "
                  << device_count << '\n';
        return 77;
    }

    const ExecutionContext ec({0, 1});
    const bool peer_access = ops::enable_peer_access(ec);
    std::cout << "peer access: "
              << (peer_access ? "enabled (direct P2P)"
                              : "unavailable (CUDA stages the device-to-device copies through "
                                "host memory)")
              << '\n';
    // The baseline follows the process selection (file comment): the staged control by default,
    // the peer-pull flags on every case with NINFER_TP_PEER_PULL=1.
    std::cout << "selection: "
              << (peer_pull_selected() ? "NINFER_TP_PEER_PULL=1 (peer pull, paired copies)"
                                       : "default (staged)")
              << '\n';
    if (peer_pull_selected() && !peer_access) {
        std::cerr << "NINFER_TP_PEER_PULL=1 requires direct P2P in both directions\n";
        std::cout << "FAIL allreduce\n";
        return 1;
    }
    const ops::PeerEvents events(ec);

    int failures = 0;
    // Real decode shape first: 5120 is the hidden dimension all-reduced 128 times per token.
    failures += run_allreduce_case("allreduce_sum [5120]", 5120, 1, 101u, ec, events);
    // The real row-parallel residual: a full 48-token prefill chunk, 2-D.
    failures += run_allreduce_case("allreduce_sum [5120,48]", 5120, 48, 102u, ec, events);
    failures += run_allreduce_case("allreduce_sum [4097]", 4097, 1, 103u, ec, events);
    failures += run_allreduce_case("allreduce_sum [1]", 1, 1, 104u, ec, events);
    failures += run_allreduce_case("allreduce_sum [3]", 3, 1, 105u, ec, events);
    failures += run_allreduce_case("allreduce_sum [17,3]", 17, 3, 106u, ec, events);

    failures += run_allgather_case("allgather_rows [5120,1024]", 512, 512, 5120, 201u, ec, events);
    // Gathered logits: 248320 vocabulary rows for one token, split by vocabulary half.
    failures +=
        run_allgather_case("allgather_rows [1,248320]", 124160, 124160, 1, 202u, ec, events);
    failures += run_allgather_case("allgather_rows [5120,3] uneven", 2, 1, 5120, 203u, ec, events);
    failures += run_allgather_case("allgather_rows [7,2] minimal", 1, 1, 7, 204u, ec, events);

    failures += run_chained_case("chained", ec, events);

    // Captured transports. The mailbox slot holds the MTP-3 verification activation [5120, 4];
    // the [5120, 8] case exceeds it and must stay staged inside the same kind of graph.
    ops::PeerMailbox mailbox(ec, 5120 * 4 * sizeof(std::uint16_t));
    ops::PeerEvents mailbox_events(ec);
    mailbox_events.attach_mailbox(&mailbox);
    // The original exchange kernel (NINFER_TP_MAILBOX_LEGACY=1) on a mailbox of its own.
    ops::PeerMailbox legacy_mailbox(ec, 5120 * 4 * sizeof(std::uint16_t), 2,
                                    ops::PeerExchangeKernel::Legacy);
    ops::PeerEvents legacy_events(ec);
    legacy_events.attach_mailbox(&legacy_mailbox);
    failures += run_captured_case("captured staged [5120]", 5120, 1, 5, false, ec, events, nullptr);
    failures +=
        run_captured_case("captured staged [5120,4]", 5120, 4, 5, false, ec, events, nullptr);
    failures += run_captured_case("captured mailbox [5120]", 5120, 1, 5, true, ec, mailbox_events,
                                  &mailbox);
    failures += run_captured_case("captured mailbox [5120,4]", 5120, 4, 5, true, ec, mailbox_events,
                                  &mailbox);
    failures += run_captured_case("captured mailbox [8,1] minimal", 8, 1, 3, true, ec,
                                  mailbox_events, &mailbox);
    failures += run_captured_case("captured oversized [5120,8] stays staged", 5120, 8, 3, false, ec,
                                  mailbox_events, &mailbox);
    // Unaligned vector count: 5121 elements are not whole 16-byte vectors.
    failures += run_captured_case("captured [5121] stays staged", 5121, 1, 3, false, ec,
                                  mailbox_events, &mailbox);

    failures += run_captured_case("captured legacy mailbox [5120]", 5120, 1, 5, true, ec,
                                  legacy_events, &legacy_mailbox);
    failures += run_captured_case("captured legacy mailbox [5120,4]", 5120, 4, 5, true, ec,
                                  legacy_events, &legacy_mailbox);
    failures += run_captured_case("captured legacy mailbox [8,1] minimal", 8, 1, 3, true, ec,
                                  legacy_events, &legacy_mailbox);

    // Bit identity of the three captured transports, at the decode and MTP-3 verify shapes, a
    // payload whose last warp lane is partial (65 vectors) and a single vector.
    failures += run_transport_identity_case("identity [5120]", 5120, 1, 401u, ec, events,
                                            legacy_events, mailbox_events);
    failures += run_transport_identity_case("identity [5120,4]", 5120, 4, 403u, ec, events,
                                            legacy_events, mailbox_events);
    failures += run_transport_identity_case("identity [520,1]", 520, 1, 405u, ec, events,
                                            legacy_events, mailbox_events);
    failures += run_transport_identity_case("identity [8,1]", 8, 1, 407u, ec, events, legacy_events,
                                            mailbox_events);

    failures += run_prefetch_identity_case("prefetch identity [5120]", 5120, 1, 421u, ec, events,
                                           mailbox_events, mailbox);
    failures += run_prefetch_identity_case("prefetch identity [5120,4]", 5120, 4, 423u, ec, events,
                                           mailbox_events, mailbox);

    // A slot of a full batch's width: DFlash2 K=7 at four requests, [5120, 8 x 4] = 320 KiB (160
    // blocks of the pipelined kernel), carrying every narrower exchange too; one column more stays
    // staged. The original kernel on a slot of the same width, for the identity check. Then the
    // Program's layout: one request's slots plus wide slots, both widths in one graph.
    ops::PeerMailbox batch_mailbox(ec, 5120 * 32 * sizeof(std::uint16_t));
    ops::PeerEvents batch_events(ec);
    batch_events.attach_mailbox(&batch_mailbox);
    ops::PeerMailbox legacy_batch_mailbox(ec, 5120 * 32 * sizeof(std::uint16_t), 2,
                                          ops::PeerExchangeKernel::Legacy);
    ops::PeerEvents legacy_batch_events(ec);
    legacy_batch_events.attach_mailbox(&legacy_batch_mailbox);
    failures += run_captured_case("captured batch mailbox [5120,32]", 5120, 32, 5, true, ec,
                                  batch_events, &batch_mailbox);
    failures += run_captured_case("captured batch mailbox [5120,16]", 5120, 16, 3, true, ec,
                                  batch_events, &batch_mailbox);
    failures += run_captured_case("captured batch mailbox [5120,1]", 5120, 1, 3, true, ec,
                                  batch_events, &batch_mailbox);
    failures += run_captured_case("captured oversized [5120,33] stays staged", 5120, 33, 3, false,
                                  ec, batch_events, &batch_mailbox);
    failures += run_transport_identity_case("identity [5120,32]", 5120, 32, 409u, ec, events,
                                            legacy_batch_events, batch_events);
    failures += run_transport_identity_case("identity [5120,24]", 5120, 24, 411u, ec, events,
                                            legacy_batch_events, batch_events);
    failures += run_mixed_width_case(ec, events);

    failures += run_mailbox_hang_case(ec, ops::PeerExchangeKernel::Pipelined);
    failures += run_mailbox_hang_case(ec, ops::PeerExchangeKernel::Legacy);

    failures += run_core_capture_guard_case(ec);
    failures += run_peer_pull_cases(ec, events, peer_access);

    failures += run_microbenchmark(ec, events);
    run_captured_microbenchmark("staged", ec, events);
    run_captured_microbenchmark("mailbox", ec, mailbox_events);
    run_captured_microbenchmark("legacy mailbox", ec, legacy_events);

    std::cout << (failures ? "FAIL" : "OK") << " allreduce\n";
    return failures ? 1 : 0;
}
