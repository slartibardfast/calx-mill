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

## Status

Stand-up scaffold: the core types, the `concurrency` + `project` functions, the
three-substrate test falsifier (8088 / AVX-512 / TU102), and the Kani harnesses.
Adapters and the measurement pipeline land next.

## License

Released into the public domain ([Unlicense](./LICENSE)).
