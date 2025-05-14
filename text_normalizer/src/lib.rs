// text_normalizer/src/lib.rs
use pyo3::prelude::*;
use regex::Regex;

/// Minimal text normalization: lowercase and trim whitespace.
#[pyfunction]
fn normalize_text(text: &str) -> String {
    text.trim().to_lowercase()
}

/// Normalizes a batch of texts: lowercase and trim whitespace.
/// More efficient for multiple strings due to fewer Python-Rust calls.
#[pyfunction]
fn normalize_text_batch(texts: Vec<String>) -> Vec<String> {
    texts.into_iter() // Use iterator for efficient processing
        .map(|text| text.trim().to_lowercase()) // Apply the same logic
        .collect() // Collect results into a new Vec<String>
}

#[pyfunction]
fn chunk_text_rust(text: &str, size: usize, overlap: usize) -> PyResult<Vec<String>> {
    let sentence_re = Regex::new(r"[.!?]\s+").unwrap(); // Split at [.!?] followed by whitespace
    let sentences: Vec<&str> = sentence_re.split(text).collect(); // Split the text
    
    let mut chunks: Vec<String> = Vec::new();
    let mut current_chunk_words: Vec<&str> = Vec::new();
    let mut current_word_count = 0;

    for sentence in sentences {
        let sentence_words: Vec<&str> = sentence.split_whitespace().collect();
        let sentence_word_len = sentence_words.len();

        if current_word_count + sentence_word_len > size && !current_chunk_words.is_empty() {
            chunks.push(current_chunk_words.join(" "));
            
            // Reset for overlap (just a basic word-based overlap, for minimal effort)
            let overlap_start = std::cmp::max(0, current_chunk_words.len() as i32 - overlap as i32) as usize;
            current_chunk_words = current_chunk_words[overlap_start..].to_vec();

            current_word_count = current_chunk_words.len(); // Recalculate word count
        }
        current_chunk_words.extend(sentence_words);
        current_word_count += sentence_word_len;
    }

    if !current_chunk_words.is_empty() {
        chunks.push(current_chunk_words.join(" "));
    }
    Ok(chunks)
}

#[pymodule]
fn text_normalizer(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(normalize_text, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_text_batch, m)?)?;
    m.add_function(wrap_pyfunction!(chunk_text_rust, m)?)?;
    Ok(())
}