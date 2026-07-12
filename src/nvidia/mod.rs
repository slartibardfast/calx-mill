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

/// Map a ggml/GGUF element dtype name to its streamed byte width, for building
/// a [`crate::Precision`] (call/0022). The accumulator width and pipe are the
/// caller's choice (an op decides its accumulate dtype and whether it runs on
/// the tensor or CUDA-core pipe); this fixes only the data-width half from the
/// dtype. Block-quantized weights (Q4_0 etc.) are given their effective bytes
/// per element (bpw/8) rounded up to at least 1, since the projection floors on
/// streamed bytes.
pub fn dtype_bytes(dtype: &str) -> u32 {
    match dtype {
        "F32" | "f32" => 4,
        "F16" | "f16" | "BF16" | "bf16" => 2,
        "Q8_0" | "q8_0" | "Q8_1" | "q8_1" | "I8" | "i8" => 1,
        // block-quantized ~4.1-4.5 bpw -> 1 byte/elem effective (ceil).
        "Q4_0" | "q4_0" | "Q4_0_AR16" | "q4_0_ar16" | "Q4_K" | "q4_k" => 1,
        _ => 4, // unknown: assume full precision (conservative for bandwidth)
    }
}
