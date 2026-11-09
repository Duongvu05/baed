//! k-hop ego-graph extraction on a directed CSR graph.
//!
//! Semantics match the original networkx code in BAED:
//!   nodes = nx.single_source_shortest_path_length(G_directed, c, cutoff=k)
//!   sub   = nx.Graph(G_directed.subgraph(nodes))
//! i.e. BFS follows *out*-edges only, and the ego-graph keeps every directed
//! edge between kept nodes, merged into an undirected simple graph
//! (self-loops kept, parallel / reciprocal edges merged).

use rayon::prelude::*;

/// One extracted ego-graph: sorted node ids and undirected edges (u <= v), sorted.
pub struct Ego {
    pub nodes: Vec<i64>,
    pub edges: Vec<(i64, i64)>,
}

#[inline]
fn neighbors<'a>(indptr: &'a [i64], indices: &'a [i64], u: i64) -> &'a [i64] {
    let u = u as usize;
    &indices[indptr[u] as usize..indptr[u + 1] as usize]
}

/// Extract one ego-graph. Returns `None` if it has more than `max_nodes` nodes.
pub fn ego_one(
    indptr: &[i64],
    indices: &[i64],
    center: i64,
    hops: usize,
    max_nodes: Option<usize>,
) -> Option<Ego> {
    // BFS by layers. Small ego-graphs dominate, so a sorted Vec + binary search
    // beats a hash set and needs no per-thread O(N) scratch memory.
    let mut seen: Vec<i64> = vec![center];
    let mut frontier: Vec<i64> = vec![center];
    for _ in 0..hops {
        let mut next: Vec<i64> = Vec::new();
        for &u in &frontier {
            next.extend_from_slice(neighbors(indptr, indices, u));
        }
        next.sort_unstable();
        next.dedup();
        seen.sort_unstable();
        next.retain(|v| seen.binary_search(v).is_err());
        if next.is_empty() {
            break;
        }
        seen.extend_from_slice(&next);
        frontier = next;
    }
    seen.sort_unstable();
    if let Some(m) = max_nodes {
        if seen.len() > m {
            return None;
        }
    }
    let mut edges: Vec<(i64, i64)> = Vec::new();
    for &u in &seen {
        for &v in neighbors(indptr, indices, u) {
            if seen.binary_search(&v).is_ok() {
                edges.push(if u <= v { (u, v) } else { (v, u) });
            }
        }
    }
    edges.sort_unstable();
    edges.dedup();
    Some(Ego { nodes: seen, edges })
}

/// Extract ego-graphs for all centers in parallel. Order of the output follows `centers`.
pub fn ego_many(
    indptr: &[i64],
    indices: &[i64],
    centers: &[i64],
    hops: usize,
    max_nodes: Option<usize>,
) -> Vec<Option<Ego>> {
    centers
        .par_iter()
        .map(|&c| ego_one(indptr, indices, c, hops, max_nodes))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // 0 -> 1 -> 2 -> 3, 2 -> 0, 4 -> 0 (4 is not reachable from 0)
    fn graph() -> (Vec<i64>, Vec<i64>) {
        let adj: Vec<Vec<i64>> = vec![vec![1], vec![2], vec![3, 0], vec![], vec![0]];
        let mut indptr = vec![0i64];
        let mut indices = vec![];
        for row in adj {
            indices.extend(row);
            indptr.push(indices.len() as i64);
        }
        (indptr, indices)
    }

    #[test]
    fn two_hop_follows_out_edges() {
        let (p, i) = graph();
        let e = ego_one(&p, &i, 0, 2, None).unwrap();
        assert_eq!(e.nodes, vec![0, 1, 2]);
        // 0-1, 1-2, and 2->0 merged as 0-2
        assert_eq!(e.edges, vec![(0, 1), (0, 2), (1, 2)]);
    }

    #[test]
    fn max_nodes_filters() {
        let (p, i) = graph();
        assert!(ego_one(&p, &i, 0, 2, Some(2)).is_none());
        assert!(ego_one(&p, &i, 3, 2, Some(1)).is_some());
    }

    #[test]
    fn self_loop_kept() {
        let p = vec![0i64, 1];
        let i = vec![0i64];
        let e = ego_one(&p, &i, 0, 1, None).unwrap();
        assert_eq!(e.edges, vec![(0, 0)]);
    }
}
