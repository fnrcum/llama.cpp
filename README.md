# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

> [!IMPORTANT]
> This is the **fable fork** of llama.cpp (`fnrcum/llama.cpp`). It tracks upstream
> `ggml-org/llama.cpp` master and adds SYCL (Intel Arc / Battlemage), CUDA and
> KV-cache-quantization work on top. See [Fable fork changes](#fable-fork-changes)
> for what is different from upstream and how to use the prebuilt Docker images.

## Fable fork changes

This fork is upstream master plus a set of performance and experimentation patches,
maintained for serving on Intel Arc (Battlemage) and NVIDIA GPUs. If you pull this
code or use the `ghcr.io/fnrcum/llama-*-fable` Docker images, this is what you get
on top of stock llama.cpp:

- **SYCL flash attention via oneDNN (XMX engines)** - f16 prefill path for Xe2 /
  Battlemage GPUs, large prefill speedups at long context (x1.2 at 512 tokens up to
  x4+ at 80k tokens).
- **SYCL fused MoE decode** - the fused MMVQ expert-GEMV path now covers multi-token
  decode batches, which greatly improves multi-user generation throughput on MoE
  models (e.g. Arc Pro B70).
- **SYCL Gemma flash-attention tile tuning** - `ncols2=8` tile shape for DV=512
  GQA-8 decode so the KV cache is read once.
- **TurboQuant / RotorQuant KV cache types** - experimental 3- and 4-bit KV cache
  quantization types `tbq3_0`, `tbq4_0`, `planar3_0`, `iso3_0`, `planar4_0`,
  `iso4_0` (CPU + CUDA kernels). These are fork-specific: GGUF files or KV caches
  using them are not compatible with upstream llama.cpp.
- **DSpark speculative decoding** - a `dspark` draft architecture and
  `draft-dspark` speculative type (see [docs/speculative.md](docs/speculative.md)).
- **Backported open upstream PRs** - SYCL fixes merged here before landing
  upstream: [#25608](https://github.com/ggml-org/llama.cpp/pull/25608) (UE4M3/NVFP4
  scale decode fix - NVFP4 GGUFs no longer produce gibberish on Intel Arc) and
  [#25550](https://github.com/ggml-org/llama.cpp/pull/25550) (XIELU unary op).
- **Fork CI / release pipeline** - trimmed GitHub Actions to CPU/CUDA/SYCL/server
  checks plus a release workflow that publishes the Docker images and a SYCL binary
  tarball; upstream-only pipelines (Apple, Android, ROCm, MUSA, CANN, etc.) were
  removed.

Details, benchmarks and rollback notes for every change live in
[FABLE-CHANGES.md](FABLE-CHANGES.md).

### Per-commit changes vs upstream master

| Commit | Change |
| --- | --- |
| `8ec11a142` | SYCL: F16 flash attention on the XMX engines via the oneDNN graph API (prefill; x1.21 at p=512, x4.26 at p=80k on Qwen3.6-27B-Q8_0) |
| `7eebf5c57` | SYCL: review fixes for the oneDNN FA path (pp512 +32% with fa=1, perplexity delta 0.11%) |
| `425582fbd` | SYCL: oneDNN FA rev 3.0 - gate to Battlemage (Xe2) only, other archs fall back; multi-GPU sync fix |
| `96c5be9dd` | spec: add DSpark speculative decoding (`dspark` draft arch + `draft-dspark` spec type) |
| `8c548e7a8` / `1ba891a03` | spec: draft block size is read in the dflash/dspark implementations |
| `8241e1a83` | spec: PR-25222 revision v2, addressed audits |
| `47932db0e` | docs: DSpark section in `docs/speculative.md` |
| `e22b9cf72` | SYCL: extend fused MoE MMVQ GEMV to multi-token decode batches (multi-user generation speedup on Arc B70) |
| `a5c4c5425` | Add TurboQuant/RotorQuant KV cache types (`tbq3_0`, `tbq4_0`, `planar3_0`, `iso3_0`, `planar4_0`, `iso4_0`): CPU quantizers + CUDA set-rows/cpy/flash-attention kernels |
| `117df6229` | tbq3_0: fix broken 3-bit bit-packing (UB shift, dropped 3rd byte) and stack-smashing norm test |
| `463568f72` | tbq3_0: measured error thresholds in `test-quantize-fns` |
| `92f2cfc3f` | SYCL: fattn-tile `ncols2=8` for DV=512 (Gemma full-attention GQA-8 decode reads KV once) |
| `2513c8eb2` | tests: adapt patch-added `test_cpy` calls to the new upstream signature |
| `0a9ef27f2` | SYCL: fix UE4M3 (NVFP4 scale) decode - unsigned, exp=0xF valid (backport of upstream PR #25608; fixes gibberish from NVFP4 GGUFs on Arc) |
| `910e652c3` | SYCL: XIELU unary op support (backport of upstream PR #25550) |
| `61bec8ff9` / `a8cc188af` | `Dockerfile.fable` for building the `llama-sycl-fable` image from prebuilt binaries |
| `9080878ed` | CI: fable release workflow (GHCR images + binary tarball + GitHub release) |
| `496b3ad29` / `1b7ca4cb4` | docs: `FABLE-CHANGES.md` change and test log |

On top of these, master carries periodic merge commits from `ggml-org/llama.cpp`
master; run `git log --no-merges upstream/master..master` for an always-current
list. Last upstream sync: `b980114a6` (2026-07-13, upstream master @ `91c631b21`),
which also renumbered the fork KV cache type enums to make room for upstream's
`Q2_0` (see [FABLE-CHANGES.md](FABLE-CHANGES.md) session 2026-07-13). The
backported PRs and this sync were regression-tested on Arc Pro B70 with Gemma 4
26B QAT (text + vision, parallel slots) and Ornith-1.0-35B-A3B (MoE, long-context
prefill) - no regressions.

### Docker: CUDA image

The CUDA image is built from upstream's `.devops/cuda.Dockerfile` with the fork's
sources and runs `llama-server` by default. It requires an NVIDIA driver plus the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
on the host:

```sh
docker pull ghcr.io/fnrcum/llama-cuda-fable:latest

docker run --rm --gpus all \
    -v /path/to/models:/models \
    -p 8080:8080 \
    ghcr.io/fnrcum/llama-cuda-fable:latest \
    -m /models/my_model.gguf -c 32768 -ngl 99 --host 0.0.0.0 --port 8080
```

The fork's KV cache quantization types can be enabled with e.g.
`--cache-type-k tbq4_0 --cache-type-v tbq4_0` (requires flash attention, `-fa 1`).

### Docker: SYCL image (Intel Arc / Battlemage)

The SYCL image bundles the Intel oneAPI runtime, Level Zero and the compute
runtime; the host only needs a working i915/xe kernel driver. It also runs
`llama-server` by default. GPU access is passed through `/dev/dri`:

```sh
docker pull ghcr.io/fnrcum/llama-sycl-fable:latest

docker run --rm \
    --device /dev/dri \
    -v /path/to/models:/models \
    -p 8080:8080 \
    ghcr.io/fnrcum/llama-sycl-fable:latest \
    -m /models/my_model.gguf -c 32768 -ngl 99 --host 0.0.0.0 --port 8080
```

Notes:

- The oneDNN XMX flash-attention prefill path is active automatically on
  Battlemage (Xe2) GPUs with `-fa 1`; other Intel archs fall back to the regular
  SYCL kernels.
- For multi-GPU hosts add `--device /dev/dri` (all render nodes are exposed) and
  select devices with `ZE_AFFINITY_MASK` or `GGML_SYCL_DEVICE`.
- Every [release](https://github.com/fnrcum/llama.cpp/releases) also ships a
  `llama-<version>-sycl-linux-x64.tar.gz` binary tarball for running outside
  Docker (needs the oneAPI runtime + Level Zero on the host).

## Recent API changes

- [Changelog for `libllama` API](https://github.com/ggml-org/llama.cpp/issues/9289)
- [Changelog for `llama-server` REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

## Hot topics

- **Hugging Face cache migration: models downloaded with `-hf` are now stored in the standard Hugging Face cache directory, enabling sharing with other HF tools.**
- **[guide : using the new WebUI of llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/16938)**
- [guide : running gpt-oss with llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/15396)
- [[FEEDBACK] Better packaging for llama.cpp to support downstream consumers 🤗](https://github.com/ggml-org/llama.cpp/discussions/15313)
- Support for the `gpt-oss` model with native MXFP4 format has been added | [PR](https://github.com/ggml-org/llama.cpp/pull/15091) | [Collaboration with NVIDIA](https://blogs.nvidia.com/blog/rtx-ai-garage-openai-oss) | [Comment](https://github.com/ggml-org/llama.cpp/discussions/15095)
- Multimodal support arrived in `llama-server`: [#12898](https://github.com/ggml-org/llama.cpp/pull/12898) | [documentation](./docs/multimodal.md)
- VS Code extension for FIM completions: https://github.com/ggml-org/llama.vscode
- Vim/Neovim plugin for FIM completions: https://github.com/ggml-org/llama.vim
- Hugging Face Inference Endpoints now support GGUF out of the box! https://github.com/ggml-org/llama.cpp/discussions/9669
- Hugging Face GGUF editor: [discussion](https://github.com/ggml-org/llama.cpp/discussions/9268) | [tool](https://huggingface.co/spaces/CISCai/gguf-editor)
- WebGPU support is now available in the browser, see a blog/demo introducing it [here](https://reeselevine.github.io/llamas-on-the-web/).

----

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon [In Progress]](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
