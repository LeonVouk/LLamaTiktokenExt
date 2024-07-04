use pyo3::prelude::*;
use rayon::prelude::*;
use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

fn get_stats_simple(ids: &[usize]) -> HashMap<(usize, usize), usize> {
    let mut counts = HashMap::new();
    for window in ids.windows(2) {
        if let [a, b] = window {
            *counts.entry((*a, *b)).or_insert(0) += 1;
        }
    }
    counts
}

#[pyfunction]
fn get_stats(ids: Vec<usize>) -> HashMap<(usize, usize), usize> {
    let counts = DashMap::new();
    ids.par_windows(2).for_each(|pair| {
        if let [a, b] = pair {
            *counts.entry((*a, *b)).or_insert(0) += 1;
        }
    });
    counts.into_iter().collect()
}

#[pyfunction]
fn merge(ids: Vec<usize>, pair: (usize, usize), idx: usize) -> Vec<usize> {
    let mut new_ids = Vec::with_capacity(ids.len());
    let mut i = 0;
    while i < ids.len() {
        if i < ids.len() - 1 && ids[i] == pair.0 && ids[i + 1] == pair.1 {
            new_ids.push(idx);
            i += 2;
        } else {
            new_ids.push(ids[i]);
            i += 1;
        }
    }
    new_ids
}

#[pyfunction]
fn parallel_train_loop(
    text_chunks: Vec<Vec<usize>>,
    num_merges: usize,
    init_vocab_size: usize,
    vocab: HashMap<usize, Vec<u8>>,
    merges: HashMap<(usize, usize), usize>,
) -> PyResult<(HashMap<(usize, usize), usize>, HashMap<usize, Vec<u8>>)> {
    let mut ids = text_chunks;
    let merges = Arc::new(Mutex::new(merges));
    let vocab = Arc::new(Mutex::new(vocab));

    for i in 0..num_merges {
        let stats = DashMap::new();

        // Collect stats in parallel
        ids.par_iter().for_each(|chunk_ids| {
            get_stats_simple(chunk_ids).into_iter().for_each(|(k, v)| {
                *stats.entry(k).or_insert(0) += v;
            });
        });

        let pair = match stats.iter().max_by_key(|entry| *entry.value()).map(|entry| *entry.key()) {
            Some(pair) => pair,
            None => break,
        };

        let idx = init_vocab_size + i;

        // Parallelize the merge operation safely
        ids = ids.into_par_iter().map(|chunk_ids| {
            merge(chunk_ids, pair, idx)
        }).collect::<Vec<_>>();

        // Lock and update merges
        {
            let mut merges_lock = merges.lock().unwrap();
            merges_lock.insert(pair, idx);
        }

        // Extract values and update vocab
        let (bytes0, bytes1) = {
            let vocab_lock = vocab.lock().unwrap();
            (
                vocab_lock.get(&pair.0).cloned(),
                vocab_lock.get(&pair.1).cloned(),
            )
        };

        if let (Some(bytes0), Some(bytes1)) = (bytes0, bytes1) {
            let mut vocab_lock = vocab.lock().unwrap();
            vocab_lock.insert(idx, [bytes0.as_slice(), bytes1.as_slice()].concat());
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing vocabulary entries"));
        }
    }

    Ok((Arc::try_unwrap(merges).unwrap().into_inner().unwrap(), Arc::try_unwrap(vocab).unwrap().into_inner().unwrap()))
}

#[pymodule]
fn ext_llama(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_stats, m)?)?;
    m.add_function(wrap_pyfunction!(merge, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_train_loop, m)?)?;
    Ok(())
}

