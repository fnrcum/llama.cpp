# llama-sycl-fable — Change & Test Log

Maintainer of record: Fable (AI agent), session of 2026-07-06.
Target hardware: 2× Intel Arc Pro B70 (Battlemage / Xe2, 32 GB GDDR6, ~608 GB/s, 256 XMX engines) on the AI server.
Serving workload: `llama-server`, Ornith-1.0-35B-A3B (MoE, MXFP4 experts + Q8_0), 240k context,
5 parallel slots, unified KV cache (`-np 5 --kv-unified`), f16 KV (no KV quantization).

Purpose of this file: exact record of what was changed, why, how it was tested, and how to
extend or roll back. Future additions should append a new dated section at the bottom.

---

## 1. Repository state

- Repo: `/data/llama.cpp` on the AI server, branch **`fable-sycl`**.
- Base: upstream `ggml-org/llama.cpp` master @ `20a04b220` (2026-07-06).
- Merged PRs:
  - `db33d7f9c` — merge of **PR #25173** (DSpark speculative decoding: new `dspark` draft arch +
    `draft-dspark` spec type on top of the merged DFlash drafter). NOTE: inactive at runtime for
    Ornith — requires a matching DSpark draft model (`-md` + `--spec-type draft-dspark`); published
    drafts exist only for Qwen3-4B/8B/14B.
  - `19e4851aa` — merge of **PR #25222** (SYCL flash attention via oneDNN Graph SDPA on the XMX
    engines, f16 KV, Xe2-gated, prefill only, `Q->ne[1] >= 32`). The previously deployed
    `llama-sycl-fast:2026-07-02` image already contained this PR; it is carried forward with
    `-DGGML_SYCL_DNN=ON` made explicit.

### Custom commits (the actual Fable changes)

#### `e22b9cf72` — sycl: extend fused MoE MMVQ GEMV to multi-token decode batches

**Problem.** `ggml_sycl_mul_mat_id_mmvq_fused()` (the fast fused MoE expert-GEMV path) bailed out
unless `src1->ne[2] == 1 && ids->ne[1] == 1`, i.e. **single-token decode only**. With 2+ users
generating concurrently (or any speculative/MTP verify step), every `MUL_MAT_ID` fell into the
generic path in `ggml_sycl_mul_mat_id()`, which per op does: device→host copy of `ids` + a
**blocking `stream->wait()`**, a host-side counting sort, and (for decode-sized batches) a cascade
of tiny per-row GEMV launches. At 48+ layers × 3 expert matmuls per layer this serialized the
whole decode. Measured symptom: single-user gen ~35 t/s but 4-user aggregate only ~10.5 t/s.

**Change.** Generalized the fused path to process all `(token, expert-slot)` routed pairs of a
decode-sized batch in a single kernel launch:

- `ggml/src/ggml-sycl/mmvq.cpp`
  - `mul_mat_vec_q_moe` (AoS kernel) and `mul_mat_vec_q_moe_reorder` (SoA/reorder kernel):
    workgroup dim 1 now enumerates `n_ids * n_tokens` pairs; each pair decomposes into
    `token = pair / n_ids`, `slot = pair % n_ids`, reads its expert id from
    `ids_dev[token*ids_s1 + slot*ids_s0]`, its activation row at
    `(token*ne11 + (ne11==1 ? 0 : slot)) * src1_qrow_stride` (handles both the shared-row
    gate/up case `ne11==1` and the per-slot down-proj case `ne11==n_ids`), and writes
    `dst + slot*dst_slot_stride + token*dst_token_stride`.
  - `launch_mul_mat_vec_q_moe{,_reorder}` and the two public dispatchers
    `ggml_sycl_mul_mat_vec_q_id{,_reorder}` gained `n_ids, n_tokens, ne11, ids_s0, ids_s1`
    and split `dst_row_stride` into `dst_slot_stride` / `dst_token_stride`.
- `ggml/src/ggml-sycl/mmvq.hpp` — updated signatures + docs.
- `ggml/src/ggml-sycl/ggml-sycl.cpp`
  - `ggml_sycl_mul_mat_id_mmvq_fused()`: accepts `1 <= ne12 <= GGML_SYCL_MOE_FUSED_MAX_TOKENS`
    (env, default **16**), requires `ids->ne[1] == ne12`, quantizes all `ne11*ne12` src1 rows to
    Q8_1 in one call, passes ids strides in elements.
  - Call site in `ggml_sycl_mul_mat_id()`: the `ne12 == 1` guard removed — the fused function
    gates itself; larger batches (prefill) still use the counting-sort + batched-GEMM path,
    which amortizes expert weight reads better at high token counts.

**Tuning knob.** `GGML_SYCL_MOE_FUSED_MAX_TOKENS` (default 16). Decode batches larger than this
fall back to the sort path. If running MTP/speculative decoding with 5 slots, raise to ~48
(verify batches are `n_users × (n_draft+1)` tokens). Correctness validated up to 64 (see tests).
Crossover vs the sort path above ~16–64 tokens has NOT been benchmarked — measure before raising
further.

#### `61bec8ff9` — add `Dockerfile.fable`

Runtime image build. Copies **prebuilt** `build/bin` from `/data/llama.cpp` (no in-image
rebuild), base `intel/oneapi-basekit:2025.3.2-0-devel-ubuntu24.04` (same as the build container,
so oneAPI runtime libs match). `.dockerignore` limits context to `build/bin`.

**Gotcha (cost one deploy cycle):** `LD_LIBRARY_PATH` must **append** to the basekit's path
(`ENV LD_LIBRARY_PATH=/app/llama.cpp/build/bin:${LD_LIBRARY_PATH}`). Overwriting it makes
`llama-server` fail at startup with `libsvml.so: cannot open shared object file` (icx runtime
libs live under `/opt/intel/oneapi/...` and are found via the inherited path).

---

## 2. Build recipe

Built inside a throwaway container (`fable-build`, since removed) from the same basekit image,
with `/data/llama.cpp` bind-mounted at `/src`:

```bash
apt-get install -y ninja-build git build-essential   # basekit image lacks ninja/git
source /opt/intel/oneapi/setvars.sh
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE='-O3 -DNDEBUG' \
  -DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DGGML_SYCL_DNN=ON \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build build -j20      # ~2.5 min
docker build -t llama-sycl-fable:latest -t llama-sycl-fable:2026-07-06 -f Dockerfile.fable .
```

Deliberately **not** used:
- `-DGGML_SYCL_DEVICE_ARCH=xe2` (AOT): skips the `-ze-intel-greater-than-4GB-buffer-required`
  link flag (see `ggml/src/ggml-sycl/CMakeLists.txt:167`), which this deployment needs for the
  multi-GB KV buffers at 240k context. JIT (default) matches the old image's behavior.
- SYCL graphs (`GGML_SYCL_GRAPH`): compiled in by default but runtime-disabled by default
  (`GGML_SYCL_DISABLE_GRAPH=1`), and `check_graph_compatibility()` rejects graphs containing
  `MUL_MAT_ID` anyway — irrelevant for MoE models.

---

## 3. Deployment changes

Container `llama-sycl-1` (B70 #2, `renderD130`, port 8081) now runs `llama-sycl-fable:latest`.
`llama-sycl-0` (port 8080) intentionally left on `llama-sycl-fast:latest` (old image).

Flag/env deltas vs the previous deployment (everything else byte-identical):

| Setting | Old | New | Why |
|---|---|---|---|
| image | `llama-sycl-fast:latest` | `llama-sycl-fable:latest` | this work |
| `GGML_SYCL_DISABLE_OPT=1` | set | **removed** | re-enables the SYCL weight-reorder optimization for dense (attention) GEMVs in decode; verified safe with this model (MXFP4/Q8_0 experts are unaffected by reorder — only Q4_0/Q8_0/Q3_K–Q6_K dense mats and Q4_K/Q5_K/Q6_K experts get reordered) |
| `-ub` | 4096 | 4096 (kept) | `-ub 8192` tested: +5–8% prefill at 16k/48k but −8% 5-user gen; not worth it (see tests) |

Rollback: `docker rm -f llama-sycl-1`, then re-run the original command with
`llama-sycl-fast:latest` (image still present, untouched).

---

## 4. Tests performed

### 4.1 Kernel correctness — `test-backend-ops`

Official ggml op test, run on the B70 (`SYCL0`), op `MUL_MAT_ID` (the modified op):

- Default gate (`GGML_SYCL_MOE_FUSED_MAX_TOKENS=16`): **790/790 PASS** — covers the new
  multi-token fused path (test cases include `n=32` token batches... those >16 exercise the
  sort-path fallback; `n≤16` exercise the new kernel) across all quant types incl. `mxfp4`.
- `GGML_SYCL_MOE_FUSED_MAX_TOKENS=64`: **790/790 PASS** — forces the `n=32` cases through the
  new fused kernel, validating the multi-token indexing at larger batch sizes.

### 4.2 End-to-end output correctness — greedy comparison

`/data/bench-fable/correctness.py`: 3 fixed prompts, `temperature=0, top_k=1`, 120 tokens,
old deployment vs new deployment:

- 2 of 3 outputs **token-identical**.
- 1 of 3 (thermodynamics prompt) identical for the first ~2 sentences, then diverges into an
  equally coherent paraphrase (similarity 0.674). Expected: kernel changes alter FP accumulation
  order; greedy decoding amplifies eventual near-tie flips. No incoherence/garbage observed
  (the known failure mode of a bad reorder/oneDNN path is repeated single tokens — not seen).

### 4.3 Performance benchmark — `/data/bench-fable/bench.py`

Deterministic seeded prompts (tokenize→slice→detokenize to exact token counts),
`cache_prompt=false`, 200 generated tokens per request, via HTTP `/completion` timings.
Scenarios: single-user 2k/16k/48k prompt; 4 users × 8k concurrent; 5 users × 24k concurrent
(all 5 slots busy, 120k total prompt tokens). Results JSON archived in
`/data/bench-fable/results-*.json` (`baseline`, `fable-v1`, `fable-v1-warm`, `fable-ub8192`).

Baseline (old image + old env, live container before swap) vs final deployment:

| scenario | prefill t/s old→new | gen t/s old→new | wall s old→new |
|---|---|---|---|
| single 2k   | 1425 → 1431 | 35.6 → **47.2** (+33%) | 7.9 → 7.6 |
| single 16k  | 1762 → 1754 | 34.0 → **46.3** (+36%) | 15.3 → 13.7 |
| conc 4×8k (agg)  | 794 → **1070** (+35%) | 10.5 → **17.3** (+64%) | 40.3 → 29.9 |
| single 48k  | 1599 → 1598 | 30.9 → **44.6** (+44%) | 37.5 → 35.6 |
| conc 5×24k (agg) | 997 → **1065** (+7%) | 4.6 → **6.0** (+30%) | 120.3 → 112.6 |

Notes:
- First request after container start is slower (JIT kernel compilation); warm numbers above.
- `-ub 8192` variant (`results-fable-ub8192.json`): 16k prefill 1832, 48k prefill 1737
  (+5–8%), but 5×24k gen 5.52 vs 6.03 (−8%) and slower TTFT on short prompts → rejected.

### 4.4 What was NOT tested (known gaps)

- Perplexity run (`llama-perplexity`) on this exact build (PR #25222's own PPL validation was
  relied upon for the oneDNN FA path; greedy-output comparison used instead).
- The fused-path crossover between 16 and 64 tokens (gate raise beyond 16 unmeasured for perf).
- DSpark (PR #25173) at runtime — no compatible draft model for Ornith.
- Multi-GPU / tensor-parallel operation (single-device per container by design; note the
  oneDNN FA path adds a forced sync when `device_count > 1`).

---

## 5. Artifacts & where things live

| Artifact | Location (AI server) |
|---|---|
| Source tree, branch `fable-sycl` | `/data/llama.cpp` |
| Runtime image | `llama-sycl-fable:latest` == `llama-sycl-fable:2026-07-06` |
| Old image (rollback) | `llama-sycl-fast:latest` (2026-07-02) |
| Dockerfile | `/data/llama.cpp/Dockerfile.fable` |
| Bench harness | `/data/bench-fable/bench.py` |
| Correctness harness | `/data/bench-fable/correctness.py` |
| Result archives | `/data/bench-fable/results-{baseline,fable-v1,fable-v1-warm,fable-ub8192}.json`, `ref-{baseline,fable,final}.json` |

Runtime env knobs added/relevant to this build:

| Env | Default | Meaning |
|---|---|---|
| `GGML_SYCL_MOE_FUSED_MAX_TOKENS` | 16 | max decode-batch tokens routed through the new fused MoE GEMV; larger batches use the sort path |
| `GGML_SYCL_FA_ONEDNN` | 1 | kill-switch for the PR #25222 oneDNN XMX prefill FA (`0` = old tile kernel) |
| `GGML_SYCL_DISABLE_OPT` | 0 | set to `1` to restore the old no-reorder behavior |

---

## 6. Future work / ideas (unranked)

- Raise + benchmark `GGML_SYCL_MOE_FUSED_MAX_TOKENS` (needed ~48 for MTP/spec with 5 slots).
- Multi-column (ncols>1) variant of the fused MoE kernel: share expert-weight reads when the
  same expert serves several tokens in a batch (would push the crossover vs the sort path up).
- Decode-side FA: TILE kernel is XVE-bound; investigate an XMX/oneDNN path for `Q->ne[1] < 32`
  or a wider vec kernel for the 5-token unified-KV decode shape.
- Remove the blocking `stream->wait()` + host counting sort in the big-batch `mul_mat_id`
  path (device-side sort or persistent mapping buffer).
- Try `-DGGML_SYCL_DEVICE_ARCH=xe2` AOT once the >4GB-buffer link-flag interaction is resolved
  upstream (faster container cold start; perf likely neutral).
- Upstream the multi-token fused MoE kernel as a PR to ggml-org/llama.cpp.

---

*Append new dated sections below this line for future changes.*
