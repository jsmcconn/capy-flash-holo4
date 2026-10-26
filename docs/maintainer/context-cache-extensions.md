# Context-cache extensions

This branch is based on upstream v0.7.5. Its commits separate independent
implementation changes and keep their behavioral tests with the relevant code.
Upstream model support, numerical formats and serving defaults remain the base
contract; these extensions do not establish new model-quality or throughput claims.

## Physical capacity and Host storage

An explicit physical KV pool may exceed logical context times concurrency to
retain immutable pages and copy-on-write tails. Logical request limits and full
runtime memory checks still apply. Automatic capacity selection is unchanged.

Host pressure demotions can reserve several first-fit extents. Allocation proofs
account for release order, descriptor ownership, aliases, last references and
writers. Transaction tests cover byte-preserving transfers and reservation
release on cancellation; these checks do not claim an observed production fault.

## Retention and planning

`--retain-dormant-shared` keeps a rebuild-cost prior after explicit credit expires.
It is off by default and does not guarantee that a particular prefix stays cached.
`--trace-shared-lifecycle` emits optional metadata diagnostics without prompt text.
See [serving options](../serving.md#server-options) for the public contract.

Portfolio evaluation indexes dense owner identifiers directly while preserving
opaque, sparse and shuffled identifiers. An independent integer oracle covers
both layouts.

Pressure planning constructs a proved preserving seed before optional refinement.
Capture scenarios share an assessment/time budget and optimistic ordering; a
feasible construction opportunity is considered before optional combinatorial
search. The budget limits optional work, not every operation in an admission.
Native fixtures cover null seeds, cancellation, exact ownership and budget stops.

## Prompt-cache policy

A full set of four explicit markers is preserved instead of losing a marker to
an absent implicit target. `prompt_cache_options.before_images: true` adds
optional boundaries before media turns within the same write budget. Explicit
mode disables these automatic boundaries. Chat and Responses image-bearing tool
outputs use the same policy; protocol tests cover precedence and translation.
See [OpenAI prompt caching](../serving.md#openai-prompt-caching).

## Optional transport

`NINFER_TP_PEER_PULL=1` enables the supported captured peer-pull route and paired
captured-copy synchronization. It requires peer access and supported shapes.
The derived primitive retains its [vLLM license and attribution](../../third_party/vllm/README.ninfer.md).
Independent numerical, graph-capture and lifetime tests live with the operation.
A collective microbenchmark does not establish whole-engine performance.
See [transport diagnostics](../capy-flash/diagnostics.md).

## Validation

Run the affected CMake test targets described in [tests](../../tests/README.md),
plus `python3 tools/check_docs.py` and the normal pre-commit hooks. Transport and
artifact tests require their declared devices and artifacts; a skipped GPU check
is not transport qualification. Changes to context reuse or numerical execution
also require representative model-level checks before selecting a serving build.
Workstation configuration and deployment records are intentionally external to
this source branch.
