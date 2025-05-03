// text_normalizer/src/lib.rs
use pyo3::prelude::*;

/// Minimal text normalization: lowercase and trim whitespace.
#[pyfunction]
fn normalize_text(text: &str) -> String {
    text.trim().to_lowercase()
}

// --- ADDED BATCH FUNCTION ---
/// Normalizes a batch of texts: lowercase and trim whitespace.
/// More efficient for multiple strings due to fewer Python-Rust calls.
#[pyfunction]
fn normalize_text_batch(texts: Vec<String>) -> Vec<String> {
    texts.into_iter() // Use iterator for efficient processing
        .map(|text| text.trim().to_lowercase()) // Apply the same logic
        .collect() // Collect results into a new Vec<String>
}
// --- END ADDED BATCH FUNCTION ---

#[pymodule]
fn text_normalizer(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(normalize_text, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_text_batch, m)?)?;
    Ok(())
}
