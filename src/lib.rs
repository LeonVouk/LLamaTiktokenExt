use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Once;
use std::sync::atomic::{fence, Ordering};

// Thread initialization
static INIT_RAYON: Once = Once::new();

fn init_rayon_pool() {
    // Use 25% of the CPUs by default for safety. Increase with care if extra performance is needed -> Getting segmentation faults at 30% on my local environment
    let fraction = 0.25_f64;
    let threads = std::cmp::max(1, (fraction * (num_cpus::get() as f64)).ceil() as usize);
    println!("System starting with {} threads by default. Change at src.lib.init_rayon_pool if not acceptable", threads);

    INIT_RAYON.call_once(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .unwrap();
    });
}

fn get_stats(ids: &[usize]) -> HashMap<(usize, usize), usize> {
    // Memory fence to ensure ordering, might help with the damn segmentation faults
    fence(Ordering::SeqCst);

    let result = ids
        .par_windows(2)
        .fold(
            HashMap::new,
            |mut local_counts, pair| {
                if let [a, b] = pair {
                    *local_counts.entry((*a, *b)).or_insert(0) += 1;
                }
                local_counts
            },
        )
        .reduce(
            HashMap::new,
            |mut acc, local_counts| {
                for (k, v) in local_counts {
                    *acc.entry(k).or_insert(0) += v;
                }
                acc
            },
        );

    // Memory fence to ensure ordering, might help with the damn segmentation faults
    fence(Ordering::SeqCst);
    result
}

fn merge(ids: Vec<usize>, pair: (usize, usize), idx: usize) -> Vec<usize> {
    // Memory fence to ensure ordering, might help with the damn segmentation faults
    fence(Ordering::SeqCst);

    let mut new_ids = Vec::with_capacity(ids.len());
    let mut i = 0;
    while i < ids.len() {
        if i + 1 < ids.len() && ids[i] == pair.0 && ids[i + 1] == pair.1 {
            new_ids.push(idx);
            i += 2;
        } else {
            new_ids.push(ids[i]);
            i += 1;
        }
    }

    // Memory fence to ensure ordering, might help with the damn segmentation faults
    fence(Ordering::SeqCst);
    new_ids
}


#[pyfunction]
fn train_loop(
    mut ids: Vec<Vec<usize>>,
    num_merges: usize,
    init_vocab_size: usize,
    mut vocab: HashMap<usize, Vec<u8>>,
    mut merges: Vec<(usize, usize)>,
    resume_step: usize,
    steps_per_call: usize,
) -> (Vec<(usize, usize)>, HashMap<usize, Vec<usize>>, Vec<Vec<usize>>, usize) {
    eprintln!("[DEBUG] Entering train_loop");
    eprintln!("[DEBUG] num_merges: {}, init_vocab_size: {}, initial merges length: {}", num_merges, init_vocab_size, merges.len());
    eprintln!("[DEBUG] text_chunks length: {}", ids.len());
    eprintln!("[DEBUG] resume_step: {}, steps_per_call: {}", resume_step, steps_per_call);

    init_rayon_pool();

    let end_step = (resume_step + steps_per_call).min(num_merges);
    let mut last_step = resume_step.saturating_sub(1);

    for i in resume_step..end_step {
        eprintln!("[DEBUG] Iteration: {}", i);

        let stats = ids
            .par_iter()
            .map(|chunk_ids| get_stats(chunk_ids.as_slice()))
            .reduce(HashMap::new, |mut acc, local_counts| {
                for (k, v) in local_counts {
                    *acc.entry(k).or_insert(0) += v;
                }
                acc
            });

        let pair = match stats.iter().max_by_key(|entry| entry.1) {
            Some(pair) => *pair.0,
            None => {
                eprintln!("[DEBUG] No pairs found, breaking out of loop.");
                break;
            }
        };

        let idx = init_vocab_size + i;

        // For some reason this is faster... don't @me
        let left_token = match vocab.get(&pair.0) {
            Some(v) => v,
            None => {
                eprintln!("[ERROR] Vocab does not contain token {}, stopping.", pair.0);
                break;
            }
        };
        let right_token = match vocab.get(&pair.1) {
            Some(v) => v,
            None => {
                eprintln!("[ERROR] Vocab does not contain token {}, stopping.", pair.1);
                break;
            }
        };

        ids = ids
            .into_par_iter()
            .map(|chunk_ids| merge(chunk_ids, pair, idx))
            .collect();

        let new_token = [left_token.as_slice(), right_token.as_slice()].concat();
        vocab.insert(idx, new_token);
        merges.push(pair);

        last_step = i;
    }

    let vocab_as_usize: HashMap<usize, Vec<usize>> = vocab
        .into_iter()
        .map(|(k, v)| (k, v.into_iter().map(|b| b as usize).collect()))
        .collect();

    // If no merges were done this call, last_step might be resume_step-1, so we saturate add 1 to avoid underflow issues.
    let current_step = last_step.saturating_add(1);

    eprintln!("[DEBUG] Exiting train_loop at step: {}", current_step);

    (merges, vocab_as_usize, ids, current_step)
}

#[pymodule]
fn ext_llama(_py: Python, m: &PyModule) -> PyResult<()> {
    eprintln!("[DEBUG] Initializing Python module ext_llama");
    m.add_function(wrap_pyfunction!(train_loop, m)?)?;
    Ok(())
}
