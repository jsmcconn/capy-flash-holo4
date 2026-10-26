#include "models/qwen3_5/program/planning/host_pressure_seed.h"
#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

using ninfer::models::qwen3_5::detail::host_eviction_prefix;

static void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

int main() {
    try {
        const auto unlimited = [] { return true; };
        // Two eligible dormant owners are ordered cheap then expensive. The
        // protected source is absent from that list and remains independently held.
        unsigned probes = 0;
        auto cheap      = host_eviction_prefix(
            1,
            [&](std::size_t n) {
                ++probes;
                return n >= 1;
            },
            unlimited);
        require(cheap && *cheap == 1, "Host-full seed did not retain a non-maximal cheap eviction");
        require(probes <= 2, "one-owner cover performed an unbounded search");
        // Only the all-owner target could fit: the caller's non-maximal limit
        // excludes it, so the mandatory fallback remains explicit.
        require(!host_eviction_prefix(
                    1, [](std::size_t n) { return n >= 2; }, unlimited),
                "Host seed disguised an all-owner fallback");
        // Independent reference oracle: each allocation is freed only after all
        // its owners leave. Individually additive freed-byte estimates are wrong.
        const std::array<std::vector<unsigned>, 3> references{{{0, 1}, {2}, {3}}};
        const std::array<unsigned, 3> bytes{600, 700, 1000};
        auto aliases = host_eviction_prefix(
            3,
            [&](std::size_t n) {
                unsigned free = 0;
                for (std::size_t page = 0; page < references.size(); ++page) {
                    bool released = true;
                    for (auto owner : references[page]) released &= owner < n;
                    if (released) free += bytes[page];
                }
                return free >= 500;
            },
            unlimited);
        require(aliases && *aliases == 2,
                "joint alias release did not retain the smallest feasible prefix");
        for (std::size_t needed = 1; needed < 192; ++needed) {
            unsigned calls = 0;
            auto cover     = host_eviction_prefix(
                191,
                [&](std::size_t n) {
                    ++calls;
                    return n >= needed;
                },
                unlimited);
            require(cover && *cover == needed && calls <= 17,
                    "Host seed lost its logarithmic bound or minimal prefix");
        }
        require(!host_eviction_prefix(
                    0, [](std::size_t) { return true; }, unlimited),
                "empty owner set produced a target");
        unsigned calls = 0;
        auto bounded   = host_eviction_prefix(
            31,
            [&](std::size_t n) {
                ++calls;
                return n >= 5;
            },
            [&] { return calls < 4; });
        require(bounded && *bounded == 8 && calls == 4,
                "budget expiry lost the already-proved feasible upper bound");
        require(!host_eviction_prefix(
                    31, [](std::size_t) { return true; }, [] { return false; }),
                "exhausted seed budget accepted an untested target");
        std::cout << "Host pressure seed controls passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
