# Vendored vLLM custom all-reduce synchronization

NInfer vendors one header of the vLLM custom all-reduce:

- upstream: <https://github.com/vllm-project/vllm>
- commit: `6b6a6c5c98e4a23c77bb07d344c515f48c157193`
- file: `csrc/custom_collective_common.cuh` (sha256
  `42ea189bad64b6e49fdf9c733d402bfea9f62bfd8ae2b3dee9fb437b3904adb9`)
- license: Apache License 2.0; see `LICENSE`

`custom_collective_common.cuh` is the unchanged upstream file: the per-block start/end GPU flag
barriers (`barrier_at_start`, `barrier_at_end`), the `Signal`/`RankData`/`RankSignals` layouts and
the packed BF16 reduction (`packed_reduce`). `src/ops/common/allreduce.cu` includes it and carries
local derivatives, each marked in that file:

- `peer_pull_reduce_1stage`, a modified copy of upstream's `cross_device_reduce_1stage`
  (`csrc/custom_all_reduce.cuh`, same commit): the two operand pointers arrive as a kernel argument
  instead of through a device-resident registration table; it uses the three derivatives below and
  upstream's non-final (release/acquire) `barrier_at_end<2, false>` instead of the relaxed final one.
- `barrier_at_start_publishing`, derived from `barrier_at_start`: the same start plane and
  `_flag` counter, but each signalling thread executes a system-scope fence and stores the peer's
  start flag with `st.release.sys`, and waits with `ld.acquire.sys`, where upstream's volatile
  accesses publish nothing. Its flag accesses carry a `"memory"` clobber (upstream's
  `st_flag_release`/`ld_flag_acquire` asm does not).
- `packed_reduce_cg`, derived from `packed_reduce`: each 16-byte operand is loaded with
  `ld.global.cg` (`__ldcg`); the upcast, the addition order and the downcast are upstream's own
  functions.
- `peer_copy_pair`, a byte copy between `barrier_at_start_publishing` and
  `barrier_at_end<2, false>` on the same `Signal` blocks (`ops::captured_peer_copy_pair`).

Nothing else from vLLM (IPC registration, the communicator, the two-stage kernel) is used.
