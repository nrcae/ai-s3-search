// text_normalizer/src/lib.rs
use pyo3::prelude::*;

/// Minimal text normalization: lowercase and trim whitespace.
#[pyfunction]
fn normalize_text(text: &str) -> String {
    text.trim().to_lowercase()
}

#[pymodule]
fn text_normalizer(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(normalize_text, m)?)?;
    Ok(())
}
