# calx-mill

A generic substrate-throughput modeler: occupancy and throughput projection for
*any* compute substrate, from an Intel 8088 to an AVX-512 core to a GPU SM.

`calx-mill` generalizes the Roofline model (Williams, Waterman, Patterson 2009)
from one dimension (arithmetic intensity: compute vs memory) into a multi-pipe,
multi-resource model with an explicit **concurrency** dimension. "Occupancy"
stops being a GPU word and becomes *concurrency saturation* — the same quantity
for an 8088's prefetch queue, an AVX-512 core's ROB, and an SM's warp count.

## Two halves

- **`concurrency`** — the substrate-generic occupancy: how many concurrent work
  units fit, a `min` of the register limit, the local-store limit, and the hard
  ceiling.
- **`project`** — the per-pipe-max (PPM) throughput projection, floored by the
  scheduler issue cap and the memory byte budget.

The core knows nothing of CUDA, x86, or the 8088; a `Substrate` is just its
resource axes. Substrate-specific **adapters** (an NVIDIA-SM adapter parsing
`ptxas -v` + `ncu`, etc.) build a `Substrate` and feed it to the core.

## Correctness

The arithmetic is [Kani](https://model-checking.github.io/kani/)-verified,
universal over substrate specs: the concurrency is bounded by the ceiling, the
allocation rounding is the tightest valid multiple, occupancy stays in `[0,100]`,
and fewer per-unit resources never lowers concurrency (monotonicity). Run the
proofs with:

```
cargo kani
```

The empirical anchor (the runtime `cudaOccupancy*` API, `ncu` achieved occupancy
on a *compiled* kernel) stays outside Kani — measured truth, not proven.

## The NVIDIA adapter

`src/nvidia/` ports the tu102 reference measurement pipeline to Rust: the
measured op-table ingest and PPM projection (`project.py`), the SASS op census
(`sass_census.py`), the timed-loop purity gate (`check_sass.py`), the
deterministic table generator (`mk_table.py`), the absolute projection gate
(`verify_projection.py`), and parsers for `ptxas -v` resource usage and
`ncu --csv` metric exports. Differential tests hold every port to
byte-for-byte output parity against goldens generated from the Python
(`tests/fixtures/README.md` records the provenance; the ncu fixture is
documented-format, pending a live capture).

The `calx-mill` binary exposes the pipeline: `project`, `census`,
`check-sass` (incl. `--census-match`), `mk-table`, `verify-projection`,
`ptxas`, `ncu`. Run `calx-mill --help` for the surface. `cargo build
--release` is the whole build.

## Status

Core, Kani proofs, the three-substrate falsifier (8088 / AVX-512 / TU102),
and the NVIDIA adapter's parity half (checked-in data only). The
predicted-vs-measured close against live hardware is the next lane.

## License

Released into the public domain ([Unlicense](./LICENSE)).
