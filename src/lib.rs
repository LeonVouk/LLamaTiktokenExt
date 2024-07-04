use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Mutex;


fn get_stats(ids: Vec<usize>) -> HashMap<(usize, usize), usize> {
    let counts = Mutex::new(HashMap::new());
    ids.par_windows(2).for_each(|pair| {
        if let [a, b] = pair {
            let mut counts = counts.lock().unwrap();
            *counts.entry((*a, *b)).or_insert(0) += 1;
        }
    });
    let counts = counts.into_inner().unwrap();
    counts
}


fn merge(ids: Vec<usize>, pair: (usize, usize), idx: usize) -> Vec<usize> {
    let mut new_ids = Vec::new();
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
fn train_loop(
    text_chunks: Vec<Vec<usize>>,
    num_merges: usize,
    init_vocab_size: usize,
    mut vocab: HashMap<usize, Vec<u8>>,
    mut merges: Vec<(usize, usize)>
) -> (Vec<(usize, usize)>, HashMap<usize, Vec<usize>>) {
    let mut ids = text_chunks;
    println!("Training starting.");
    for i in 0..num_merges {
        println!("Iteration: {}", i);
        let stats = Mutex::new(HashMap::new());

        ids.par_iter().for_each(|chunk_ids| {
            get_stats(chunk_ids.clone()).iter().for_each(|(k, v)| {
                let mut stats = stats.lock().unwrap();
                *stats.entry(*k).or_insert(0) += v;
            });
        });

        let stats = stats.into_inner().unwrap();
        let pair = match stats.iter().max_by_key(|entry| entry.1) {
            Some(pair) => *pair.0,
            None => break,
        };

        let idx = init_vocab_size + i;
        ids = ids
            .into_par_iter()
            .map(|chunk_ids| merge(chunk_ids, pair, idx))
            .collect();

        merges.push(pair);
        vocab.insert(idx, [vocab[&pair.0].as_slice(), vocab[&pair.1].as_slice()].concat());
    }

    let vocab_as_usize = vocab.into_iter().map(|(k, v)| (k, v.into_iter().map(|b| b as usize).collect())).collect();
    (merges, vocab_as_usize)
}


#[pymodule]
fn ext_llama(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(train_loop, m)?)?;
    Ok(())
}
