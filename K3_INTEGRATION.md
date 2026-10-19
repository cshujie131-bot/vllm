# K3 plugin interface integration

Public base: 2cf0a6915ce544dc493a0990f2ea38d81601128a (vLLM 0.28.0).

This source branch exposes the interfaces used by the accepted K3 plugin:

- Model/vision component factories reuse native loaders, processors, metadata,
  model forwards and execution skeletons. Default components are unchanged.
- Runner class selection, fixed-input copy hook and Graph descriptor metadata
  let FL own its workspace and intermediate-tensor lifetime policy.
- CUSTOM MLA selection uses the registered plugin backend. Only a custom backend
  explicitly declaring support for missing CUDA capability can accept it; native
  fallback and dtype/dimension checks remain intact.
- PP exposes transport initialization and receive-completion hooks. The native
  communicator, queue and state filtering remain unchanged. FL implements the
  host-packed transport; this repository contains no K3 transport implementation.

Nine existing Python files change. There is no request-scheduler algorithm
change and no runtime source rewriting. The PP default AST matches the public
base after inlining the two hooks; other interfaces retain the previously
verified default-preserving changes. This fork is a pinned integration source,
not a claim that the interfaces have been merged upstream. Upstream submission
requires human review under this repository's contribution policy.

Build the empty backend against the matching vendor Torch. For a shallow source
checkout, the documented build-only VLLM_VERSION_OVERRIDE may be set to
0.28.0+empty; installed Git provenance separately identifies this exact source.
AI assistance was used for this integration.
