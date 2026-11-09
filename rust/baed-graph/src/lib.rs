use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub mod cc;
pub mod ego;

type EgoOut<'py> = (
    Bound<'py, PyArray1<bool>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
);

/// Extract k-hop ego-graphs of `centers` from a directed CSR graph (out-edges).
///
/// Returns flat arrays to avoid allocating one Python object per subgraph:
///   kept        bool[n_centers]  - False if the ego-graph exceeded `max_nodes`
///   node_ptr    i64[n_kept + 1]  - nodes of kept ego-graph j are nodes[node_ptr[j]:node_ptr[j+1]]
///   nodes       i64[]            - sorted global node ids
///   edge_ptr    i64[n_kept + 1]
///   edges       i64[2 * n_edges] - flattened (u, v) pairs, u <= v, global ids
#[pyfunction]
#[pyo3(signature = (indptr, indices, centers, hops, max_nodes=None))]
fn ego_subgraphs<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    centers: PyReadonlyArray1<'py, i64>,
    hops: usize,
    max_nodes: Option<usize>,
) -> PyResult<EgoOut<'py>> {
    let indptr = indptr.as_slice()?;
    let indices = indices.as_slice()?;
    let centers = centers.as_slice()?;
    let n = indptr.len().saturating_sub(1) as i64;
    if indptr.is_empty() || *indptr.last().unwrap() as usize != indices.len() {
        return Err(PyValueError::new_err("indptr[-1] must equal len(indices)"));
    }
    if indices.iter().chain(centers.iter()).any(|&x| x < 0 || x >= n) {
        return Err(PyValueError::new_err("node id out of range"));
    }

    let results = py.detach(|| ego::ego_many(indptr, indices, centers, hops, max_nodes));

    let mut kept = Vec::with_capacity(results.len());
    let mut node_ptr = vec![0i64];
    let mut edge_ptr = vec![0i64];
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    for r in results {
        match r {
            None => kept.push(false),
            Some(e) => {
                kept.push(true);
                nodes.extend_from_slice(&e.nodes);
                node_ptr.push(nodes.len() as i64);
                for (u, v) in e.edges {
                    edges.push(u);
                    edges.push(v);
                }
                edge_ptr.push((edges.len() / 2) as i64);
            }
        }
    }
    Ok((
        kept.into_pyarray(py),
        node_ptr.into_pyarray(py),
        nodes.into_pyarray(py),
        edge_ptr.into_pyarray(py),
        edges.into_pyarray(py),
    ))
}

/// Largest connected component of every graph in a PyG-style batch.
///
/// `node_ptr` (B+1) delimits the nodes of each graph, `src`/`dst` are global edge ids.
/// Returns (nodes, node_ptr, edges[2*E] local ids, edge_ptr) - see `cc.rs` for the exact semantics.
#[pyfunction]
fn largest_components<'py>(
    py: Python<'py>,
    node_ptr: PyReadonlyArray1<'py, i64>,
    src: PyReadonlyArray1<'py, i64>,
    dst: PyReadonlyArray1<'py, i64>,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)> {
    let node_ptr = node_ptr.as_slice()?;
    let src = src.as_slice()?;
    let dst = dst.as_slice()?;
    let n = *node_ptr.last().unwrap_or(&0);
    if src.len() != dst.len() || src.iter().chain(dst.iter()).any(|&x| x < 0 || x >= n) {
        return Err(PyValueError::new_err("bad edge list"));
    }
    let o = py.detach(|| cc::largest_components(node_ptr, src, dst));
    let mut flat = Vec::with_capacity(o.edges.len() * 2);
    for (u, v) in o.edges {
        flat.push(u);
        flat.push(v);
    }
    Ok((o.nodes.into_pyarray(py), o.node_ptr.into_pyarray(py), flat.into_pyarray(py), o.edge_ptr.into_pyarray(py)))
}

/// Number of rayon worker threads (set RAYON_NUM_THREADS to change it).
#[pyfunction]
fn num_threads() -> usize {
    rayon::current_num_threads()
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ego_subgraphs, m)?)?;
    m.add_function(wrap_pyfunction!(largest_components, m)?)?;
    m.add_function(wrap_pyfunction!(num_threads, m)?)?;
    Ok(())
}
