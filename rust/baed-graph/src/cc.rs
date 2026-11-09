//! Largest connected component of every graph in a batch (undirected view).
//!
//! Matches the original Python post-processing of generated graphs:
//!   g = to_networkx(data, to_undirected=True)
//!   g = g.subgraph(max(nx.connected_components(g), key=len))
//!   from_networkx(g)
//! nx.connected_components yields components in order of their smallest node,
//! and max() keeps the first maximal one, so ties go to the component with the
//! smallest node id. Kept nodes stay in ascending order (local ids 0..k-1) and
//! every undirected edge is emitted in both directions (a self-loop once),
//! sorted by (src, dst).

fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

pub struct CcOut {
    pub nodes: Vec<i64>,     // global ids of kept nodes, per graph ascending
    pub node_ptr: Vec<i64>,  // B + 1
    pub edges: Vec<(i64, i64)>, // local ids
    pub edge_ptr: Vec<i64>,  // B + 1
}

pub fn largest_components(node_ptr: &[i64], src: &[i64], dst: &[i64]) -> CcOut {
    let n_total = *node_ptr.last().unwrap_or(&0) as usize;
    let b = node_ptr.len().saturating_sub(1);
    // graph id of every node
    let mut gid = vec![0usize; n_total];
    for g in 0..b {
        for v in node_ptr[g] as usize..node_ptr[g + 1] as usize {
            gid[v] = g;
        }
    }
    let mut parent: Vec<usize> = (0..n_total).collect();
    for (&u, &v) in src.iter().zip(dst.iter()) {
        let (ru, rv) = (find(&mut parent, u as usize), find(&mut parent, v as usize));
        if ru != rv {
            // keep the smaller id as root so the root is the component's minimum node
            let (lo, hi) = if ru < rv { (ru, rv) } else { (rv, ru) };
            parent[hi] = lo;
        }
    }
    let mut size = vec![0usize; n_total];
    for v in 0..n_total {
        let r = find(&mut parent, v);
        size[r] += 1;
    }
    // per graph: root of the largest component (ties -> smallest root = smallest node)
    let mut best = vec![usize::MAX; b];
    for v in 0..n_total {
        if parent[v] == v {
            let g = gid[v];
            if best[g] == usize::MAX || size[v] > size[best[g]] || (size[v] == size[best[g]] && v < best[g]) {
                best[g] = v;
            }
        }
    }
    let mut local = vec![-1i64; n_total];
    let mut out = CcOut { nodes: vec![], node_ptr: vec![0], edges: vec![], edge_ptr: vec![0] };
    let mut per_graph_edges: Vec<Vec<(i64, i64)>> = vec![Vec::new(); b];
    for g in 0..b {
        let mut k = 0i64;
        for v in node_ptr[g] as usize..node_ptr[g + 1] as usize {
            if find(&mut parent, v) == best[g] {
                local[v] = k;
                k += 1;
                out.nodes.push(v as i64);
            }
        }
        out.node_ptr.push(out.nodes.len() as i64);
    }
    for (&u, &v) in src.iter().zip(dst.iter()) {
        let (u, v) = (u as usize, v as usize);
        if local[u] >= 0 && local[v] >= 0 {
            let (a, c) = (local[u], local[v]);
            let e = &mut per_graph_edges[gid[u]];
            e.push((a, c));
            if a != c {
                e.push((c, a));
            }
        }
    }
    for mut e in per_graph_edges {
        e.sort_unstable();
        e.dedup();
        out.edges.extend(e);
        out.edge_ptr.push(out.edges.len() as i64);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_largest_and_breaks_ties_by_min_node() {
        // graph 0: nodes 0..4, comps {0},{1,2},{3,4} -> tie between {1,2} and {3,4} -> {1,2}
        // graph 1: nodes 5..7, edges 5-6, 6-7 -> all
        let ptr = vec![0, 5, 8];
        let src = vec![1, 4, 5, 7];
        let dst = vec![2, 3, 6, 6];
        let o = largest_components(&ptr, &src, &dst);
        assert_eq!(o.nodes, vec![1, 2, 5, 6, 7]);
        assert_eq!(o.node_ptr, vec![0, 2, 5]);
        assert_eq!(o.edges, vec![(0, 1), (1, 0), (0, 1), (1, 0), (1, 2), (2, 1)]);
        assert_eq!(o.edge_ptr, vec![0, 2, 6]);
    }

    #[test]
    fn edgeless_graph_keeps_first_node() {
        let o = largest_components(&[0, 3], &[], &[]);
        assert_eq!(o.nodes, vec![0]);
        assert!(o.edges.is_empty());
    }

    #[test]
    fn self_loop_once() {
        let o = largest_components(&[0, 1], &[0], &[0]);
        assert_eq!(o.edges, vec![(0, 0)]);
    }
}
