# Competition model recovery

Search completed 2026-09-06 across `C:\Users\JingyuWang` and repository history. Historical raw-head 320/416 models were found under the ignored local directory `tmp/rk3566_profile/`. No split-output 320/416 models were found there or on the reachable `lubancat` board; a read-only board search returned no `.rknn` files. The `opi5` host remained unreachable.

The raw models were not copied over the tracked deployment artifacts. New split-output candidates were built from the historical 320/416 ONNX and are documented in `docs/competition_model_build.md`.
