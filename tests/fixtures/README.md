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
- `project_ffma_anchor.txt`: `python3 tools/project.py census_full_anchors.csv --kernel ffma_anchor --warps 8`
- `project_stream_anchor.txt`: `python3 tools/project.py census_full_anchors.csv --kernel stream_anchor --warps 8 --mem-class dram`
- `project_fa_mini_dp4a.txt`: `python3 tools/project.py census_full_fa_mini.csv --kernel fa_mini_kernelILi1 --warps 8 --mem-class l1`

The SASS these censuses came from was emitted by nvcc 13.3.73 on this host,
not by the rig that produced the published measurements; the goldens bind the
Rust to the Python *on these inputs*, which is the parity claim (the
measured-vs-predicted close against live hardware is a separate, GPU-owning
step).
