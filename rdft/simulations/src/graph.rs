/// Graph types for particle simulation.
///
/// All graphs are stored as adjacency lists. For lattices we generate
/// the adjacency at construction time — the memory overhead is modest
/// and it keeps the hot loop simple (no coordinate arithmetic).

use rand::prelude::*;
use rand::rngs::SmallRng;

#[derive(Clone)]
pub struct Graph {
    pub adj: Vec<Vec<u32>>,
    pub name: String,
}

impl Graph {
    pub fn num_nodes(&self) -> usize {
        self.adj.len()
    }

    pub fn neighbors(&self, node: u32) -> &[u32] {
        &self.adj[node as usize]
    }

    pub fn degree(&self, node: u32) -> usize {
        self.adj[node as usize].len()
    }

    /// Pick a uniformly random neighbor.
    #[inline]
    pub fn random_neighbor(&self, node: u32, rng: &mut impl Rng) -> u32 {
        let nbrs = &self.adj[node as usize];
        nbrs[rng.gen_range(0..nbrs.len())]
    }

    // ----------------------------------------------------------------
    // Constructors
    // ----------------------------------------------------------------

    /// d-dimensional hypercubic lattice with periodic boundary conditions.
    /// Total nodes = size^dim.
    pub fn hypercubic(dim: usize, size: usize) -> Self {
        let num_nodes = size.pow(dim as u32);
        let mut adj = vec![Vec::with_capacity(2 * dim); num_nodes];

        // Precompute strides: stride[i] = size^i
        let mut strides = vec![1usize; dim];
        for i in 1..dim {
            strides[i] = strides[i - 1] * size;
        }

        for node in 0..num_nodes {
            // Decode coordinates
            let mut tmp = node;
            let mut coords = vec![0usize; dim];
            for d in 0..dim {
                coords[d] = tmp % size;
                tmp /= size;
            }

            for d in 0..dim {
                // +1 in dimension d
                let plus = {
                    let mut c = coords.clone();
                    c[d] = (c[d] + 1) % size;
                    encode_coords(&c, &strides)
                };
                // -1 in dimension d
                let minus = {
                    let mut c = coords.clone();
                    c[d] = (c[d] + size - 1) % size;
                    encode_coords(&c, &strides)
                };
                adj[node].push(plus as u32);
                adj[node].push(minus as u32);
            }
        }

        Graph {
            adj,
            name: format!("{}D lattice (L={})", dim, size),
        }
    }

    /// Sierpinski carpet at given iteration level.
    /// d_s ~ 1.86 (Watanabe 1985).
    pub fn sierpinski_carpet(level: usize) -> Self {
        let size = 3usize.pow(level as u32);

        fn is_removed(mut x: usize, mut y: usize, mut sz: usize) -> bool {
            while sz > 1 {
                let third = sz / 3;
                if x >= third && x < 2 * third && y >= third && y < 2 * third {
                    return true;
                }
                x %= third;
                y %= third;
                sz = third;
            }
            false
        }

        // Build list of valid sites
        let mut site_map = std::collections::HashMap::new();
        let mut sites = Vec::new();
        for x in 0..size {
            for y in 0..size {
                if !is_removed(x, y, size) {
                    site_map.insert((x, y), sites.len());
                    sites.push((x, y));
                }
            }
        }

        let n = sites.len();
        let mut adj = vec![Vec::new(); n];
        let deltas: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

        for &(x, y) in &sites {
            let i = site_map[&(x, y)];
            for (dx, dy) in deltas {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx >= 0 && nx < size as i32 && ny >= 0 && ny < size as i32 {
                    if let Some(&j) = site_map.get(&(nx as usize, ny as usize)) {
                        adj[i].push(j as u32);
                    }
                }
            }
        }

        Graph {
            adj,
            name: format!("Sierpinski carpet (level {})", level),
        }
    }

    /// Uniform random labelled tree on n nodes (Prüfer sequence).
    /// d_s -> 4/3 for large n (Destri-Donetti 2002).
    pub fn random_tree(n: usize, seed: u64) -> Self {
        if n <= 1 {
            return Graph {
                adj: vec![Vec::new(); n],
                name: "Random tree (n=1)".into(),
            };
        }

        let mut rng = SmallRng::seed_from_u64(seed);

        // Prüfer sequence of length n-2
        let prufer: Vec<u32> = (0..n - 2).map(|_| rng.gen_range(0..n as u32)).collect();

        // Decode Prüfer sequence to tree edges
        let mut degree = vec![1u32; n];
        for &p in &prufer {
            degree[p as usize] += 1;
        }

        let mut adj = vec![Vec::new(); n];
        for &p in &prufer {
            // Find smallest leaf (degree == 1) not yet used
            for i in 0..n {
                if degree[i] == 1 {
                    adj[i].push(p);
                    adj[p as usize].push(i as u32);
                    degree[i] -= 1;
                    degree[p as usize] -= 1;
                    break;
                }
            }
        }

        // Connect last two nodes with degree 1
        let remaining: Vec<usize> = (0..n).filter(|&i| degree[i] == 1).collect();
        if remaining.len() == 2 {
            adj[remaining[0]].push(remaining[1] as u32);
            adj[remaining[1]].push(remaining[0] as u32);
        }

        Graph {
            adj,
            name: format!("Random tree (n={})", n),
        }
    }

    /// Barabási-Albert preferential attachment network.
    /// d_s >= 4 (mean-field regime for BRW).
    pub fn barabasi_albert(n: usize, m: usize, seed: u64) -> Self {
        assert!(m >= 1 && m < n, "Need 1 <= m < n");
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut adj = vec![Vec::new(); n];

        // Start with complete graph on m+1 nodes
        for i in 0..=m {
            for j in 0..=m {
                if i != j {
                    adj[i].push(j as u32);
                }
            }
        }

        // Repeated targets list for preferential attachment
        // Each node appears once per edge endpoint
        let mut targets: Vec<u32> = Vec::new();
        for i in 0..=m {
            for _ in 0..m {
                targets.push(i as u32);
            }
        }

        for new_node in (m + 1)..n {
            // Pick m distinct targets weighted by degree
            let mut chosen = Vec::with_capacity(m);
            let mut attempts = 0;
            while chosen.len() < m && attempts < m * 10 {
                let idx = rng.gen_range(0..targets.len());
                let t = targets[idx];
                if !chosen.contains(&t) {
                    chosen.push(t);
                }
                attempts += 1;
            }

            for &t in &chosen {
                adj[new_node].push(t);
                adj[t as usize].push(new_node as u32);
                targets.push(t);
                targets.push(new_node as u32);
            }
        }

        Graph {
            adj,
            name: format!("Barabási-Albert (n={}, m={})", n, m),
        }
    }

    /// Build graph from edge list (undirected — each edge added both ways).
    pub fn from_edge_list(n: usize, edges: &[(u32, u32)]) -> Self {
        let mut adj = vec![Vec::new(); n];
        for &(u, v) in edges {
            adj[u as usize].push(v);
            adj[v as usize].push(u);
        }
        Graph {
            adj,
            name: format!("Custom graph (n={})", n),
        }
    }

    /// Complete graph on n nodes.
    pub fn complete(n: usize) -> Self {
        let mut adj = vec![Vec::with_capacity(n - 1); n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    adj[i].push(j as u32);
                }
            }
        }
        Graph {
            adj,
            name: format!("Complete graph K_{}", n),
        }
    }
}

fn encode_coords(coords: &[usize], strides: &[usize]) -> usize {
    coords.iter().zip(strides).map(|(&c, &s)| c * s).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_1d_lattice() {
        let g = Graph::hypercubic(1, 10);
        assert_eq!(g.num_nodes(), 10);
        // Node 0 should have neighbors 1 and 9 (periodic)
        let nbrs = g.neighbors(0);
        assert_eq!(nbrs.len(), 2);
        assert!(nbrs.contains(&1));
        assert!(nbrs.contains(&9));
    }

    #[test]
    fn test_2d_lattice() {
        let g = Graph::hypercubic(2, 5);
        assert_eq!(g.num_nodes(), 25);
        // Every node should have degree 4
        for i in 0..25 {
            assert_eq!(g.degree(i), 4);
        }
    }

    #[test]
    fn test_sierpinski() {
        let g = Graph::sierpinski_carpet(1);
        // Level 1: 9 - 1 = 8 sites
        assert_eq!(g.num_nodes(), 8);
    }

    #[test]
    fn test_random_tree() {
        let g = Graph::random_tree(100, 42);
        assert_eq!(g.num_nodes(), 100);
        // Tree has n-1 edges -> sum of degrees = 2(n-1) = 198
        let total_degree: usize = (0..100).map(|i| g.degree(i as u32)).sum();
        assert_eq!(total_degree, 198);
    }

    #[test]
    fn test_ba() {
        let g = Graph::barabasi_albert(100, 3, 42);
        assert_eq!(g.num_nodes(), 100);
    }
}
