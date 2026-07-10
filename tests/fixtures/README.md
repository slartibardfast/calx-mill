# Parity fixtures and goldens

Inputs and expected outputs for the differential tests that hold the NVIDIA
adapter to the reference Python tools (`reference/tu102/tools/` in the
`connollydavid/tu102` lineage). Every golden was produced by running the
Python on this machine (Python 3.14.6); no GPU was involved anywhere - the
bench binaries below were *compiled* (nvcc needs no GPU) and disassembled,
never executed.

## tu102/ - inputs copied from the tu102 reference repo

- `table/tu102_ops.csv`: the committed measured op table, verbatim. Doubles
  as the byte-identical expected output of the `mk_table` port.
- `table/priors_t4.csv`, `table/na_sm75.csv`: the priors and explicit-absence
  tables `mk_table` consumes, verbatim.
- `data/results/t5820-2xrtx6000/*.csv`: the checked-in measurement CSVs,
  verbatim (the raw inputs of `mk_table` and `verify_projection`).
- `bench/proj/*.sass`, `bench/alu/*.sass`: `cuobjdump -sass` disassemblies of
  the bench binaries compiled as below (real SASS, no GPU, never executed).

## goldens/ - Python tool outputs

Generated from a scratch copy of the tu102 repo with the bench binaries
compiled by `/opt/cuda/bin/nvcc` (13.3.73, `V13.3.73`):

```
nvcc -O2 -arch=sm_75 -lineinfo -DTU102_GIT_SHA=\"golden\" \
     -o bench/proj/anchors.bin bench/proj/anchors.cu -lnvidia-ml
# likewise fa_mini.bin, inject.bin, alu/alu.bin, alu/pipes.bin
```

with `CUOBJDUMP` in the tools pointed at `/opt/cuda/bin/cuobjdump` (the only
edit; the tools hardcode `/opt/cuda-13.3`).

- `census_full_<bin>.csv`: `python3 tools/sass_census.py bench/<d>/<bin>.bin --full -o <out>`
- `census_base_anchors.csv`: same without `--full`
- `project_ffma_anchor.txt`: `python3 tools/project.py census_full_anchors.csv --kernel ffma_anchor --warps 8`
- `project_stream_anchor.txt`: `python3 tools/project.py census_full_anchors.csv --kernel stream_anchor --warps 8 --mem-class dram`
- `project_fa_mini_dp4a.txt`: `python3 tools/project.py census_full_fa_mini.csv --kernel fa_mini_kernelILi1 --warps 8 --mem-class l1`
- `check_sass_<bin>.txt`: `python3 tools/check_sass.py bench/<d>/<bin>.bin`
  (exit 0 everywhere except pipes.bin, exit 1 with one FAIL row; fa_mini.bin
  prints its exemption)
- `census_match.txt`: `python3 tools/check_sass.py --census-match
  bench/proj/anchors.bin:ffma_anchor bench/proj/anchors.bin:capmix_anchor`
  (exit 1, worst 49.2pp)
- `verify_projection.txt`: `python3 tools/verify_projection.py` (exit 1: 6
  gate failures - the fixture SASS is this host's nvcc, not the rig's; the
  golden binds Rust to Python on identical inputs)

## ptxas/ - real `ptxas -v` captures (nvcc needs no GPU)

- `ptxas_v_fixture.txt`: `nvcc -O2 -arch=sm_75 -Xptxas -v -c ptxas_fix.cu`
  stderr for a two-kernel probe (one plain kernel, one with 8448 B smem and
  a barrier).
- `ptxas_v_anchors.txt`: the same over `bench/proj/anchors.cu` (6 kernels).

## ncu/ - documented-format fixture, PENDING REAL-DATA VALIDATION

- `atomics_metrics.csv`: hand-built in the documented `ncu --csv` shape
  (quoted fields, "Kernel Name"/"Metric Name"/"Metric Unit"/"Metric Value"
  columns, thousands separators). No raw ncu export is checked in anywhere
  under reference/; the metric values are transcribed from
  `data/ncu-atomics-20260610/NOTES.md` (the one recorded ncu run). The parser
  must be re-validated against a live `ncu --csv` capture when the GPU lane
  runs the measured close.

The SASS these censuses came from was emitted by nvcc 13.3.73 on this host,
not by the rig that produced the published measurements; the goldens bind the
Rust to the Python *on these inputs*, which is the parity claim (the
measured-vs-predicted close against live hardware is a separate, GPU-owning
step).
