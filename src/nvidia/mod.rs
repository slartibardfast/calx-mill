//! The NVIDIA-SM adapter: parses the measured tu102 op table, SASS op mixes,
//! `ptxas -v` resource usage, and `ncu` metric exports, and folds them into
//! demands for the substrate-generic core. The core never sees a CUDA type;
//! everything CUDA-shaped stays on this side of the seam.
//!
//! The parsers are Rust ports of the tu102 reference Python tools
//! (`project.py`, `sass_census.py`, `check_sass.py`, `mk_table.py`), held to
//! byte-for-byte output parity by differential tests against goldens generated
//! from the Python (see `tests/fixtures/README.md`).

pub mod check;
pub mod csvio;
pub mod mktable;
pub mod ncu;
pub mod ptxas;
pub mod pattern;
pub mod projection;
pub mod pyfmt;
pub mod sass;
pub mod table;
pub mod verify;
