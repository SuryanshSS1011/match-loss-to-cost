"""
Data generation script for network traffic forecasting experiment.

Generates:
1. Network topology (Watts-Strogatz small-world graph)
2. Routing matrix (shortest path routing)
3. Synthetic traffic matrix time series with daily patterns
4. Link load time series
"""

import os
import pickle
import numpy as np
import networkx as nx

from .config import CONFIG, DATA_DIR
from .utils import set_all_seeds


def create_topology(num_nodes: int, k: int, p: float, seed: int) -> tuple:
    """
    Create a connected Watts-Strogatz small-world graph and extract directed links.

    Args:
        num_nodes: Number of nodes in the graph
        k: Each node is connected to k nearest neighbors in ring topology
        p: Probability of rewiring each edge
        seed: Random seed for reproducibility

    Returns:
        G: NetworkX graph object
        links: List of directed link tuples (u, v)
    """
    G = nx.connected_watts_strogatz_graph(num_nodes, k, p, seed=seed)

    # Convert undirected edges to directed links (each edge becomes two directed links)
    links = []
    for u, v in G.edges():
        links.append((u, v))
        links.append((v, u))

    # Sort for consistent ordering
    links = sorted(links)

    return G, links


def compute_routing_matrix(G: nx.Graph, links: list, od_pairs: list) -> np.ndarray:
    """
    Compute routing matrix using shortest path routing.

    Args:
        G: NetworkX graph
        links: List of directed link tuples
        od_pairs: List of OD pair tuples (source, destination)

    Returns:
        R: Routing matrix of shape (num_links, num_od) with binary entries
    """
    num_links = len(links)
    num_od = len(od_pairs)

    # Create link index mapping for fast lookup
    link_to_idx = {link: idx for idx, link in enumerate(links)}

    R = np.zeros((num_links, num_od), dtype=np.float32)

    for k, (s, t) in enumerate(od_pairs):
        # Get shortest path
        path = nx.shortest_path(G, s, t)

        # Mark all links along the path
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            link_idx = link_to_idx.get((u, v))
            if link_idx is not None:
                R[link_idx, k] = 1.0

    return R


def generate_traffic_matrix(T: int, num_od: int, config: dict, rng: np.random.Generator) -> np.ndarray:
    """
    Generate synthetic traffic matrix time series with daily patterns and bursts.

    Args:
        T: Number of time steps
        num_od: Number of OD pairs
        config: Configuration dictionary
        rng: NumPy random generator

    Returns:
        TM: Traffic matrix of shape (T, num_od)
    """
    time_step_minutes = config['time_step_minutes']
    base_min = config['base_traffic_min']
    base_max = config['base_traffic_max']
    amp_min = config['amplitude_min']
    amp_max = config['amplitude_max']
    noise_factor = config['noise_factor']
    burst_prob = config['burst_prob']
    burst_factor_min = config['burst_factor_min']
    burst_factor_max = config['burst_factor_max']

    # Sample parameters for each OD pair
    base = rng.uniform(base_min, base_max, size=num_od)
    amplitude = rng.uniform(amp_min, amp_max, size=num_od)
    phase = rng.uniform(0, 2 * np.pi, size=num_od)

    # Generate time in days
    t_steps = np.arange(T)
    t_days = t_steps * time_step_minutes / (60 * 24)  # Convert to days

    # Generate traffic matrix
    TM = np.zeros((T, num_od), dtype=np.float32)

    for k in range(num_od):
        # Sinusoidal pattern with daily period
        pattern = base[k] * (1 + amplitude[k] * np.sin(2 * np.pi * t_days + phase[k]))

        # Add Gaussian noise
        noise = rng.normal(0, noise_factor * base[k], size=T)

        TM[:, k] = pattern + noise

    # Add random bursts
    burst_mask = rng.random((T, num_od)) < burst_prob
    burst_factors = rng.uniform(burst_factor_min, burst_factor_max, size=(T, num_od))
    TM += burst_mask * burst_factors * base[np.newaxis, :]

    # Ensure non-negative
    TM = np.maximum(TM, 0)

    return TM


def compute_link_loads(R: np.ndarray, TM: np.ndarray) -> np.ndarray:
    """
    Compute link loads from traffic matrix using routing matrix.

    L[t] = R @ TM[t]

    Args:
        R: Routing matrix of shape (num_links, num_od)
        TM: Traffic matrix of shape (T, num_od)

    Returns:
        L: Link load matrix of shape (T, num_links)
    """
    # L[t] = R @ TM[t].T, computed efficiently as TM @ R.T
    L = TM @ R.T
    return L


def split_data(T: int, train_frac: float, val_frac: float) -> dict:
    """
    Compute train/val/test split indices.

    Args:
        T: Total number of time steps
        train_frac: Fraction for training
        val_frac: Fraction for validation

    Returns:
        Dictionary with split indices
    """
    T_train = int(T * train_frac)
    T_val = int(T * val_frac)
    T_test = T - T_train - T_val

    return {
        'T_train': T_train,
        'T_val': T_val,
        'T_test': T_test,
        'train_end': T_train,
        'val_end': T_train + T_val,
    }


def save_data(G: nx.Graph, links: list, od_pairs: list, R: np.ndarray,
              TM: np.ndarray, L: np.ndarray, split_info: dict, data_dir: str) -> None:
    """Save all generated data to disk."""
    os.makedirs(data_dir, exist_ok=True)

    # Save topology
    np.savez(os.path.join(data_dir, 'topology.npz'),
             edges=np.array(list(G.edges())),
             links=np.array(links),
             num_nodes=G.number_of_nodes())

    # Save graph object separately (for networkx functions)
    with open(os.path.join(data_dir, 'graph.pkl'), 'wb') as f:
        pickle.dump(G, f)

    # Save routing matrix
    np.savez(os.path.join(data_dir, 'routing_matrix.npz'),
             R=R,
             od_pairs=np.array(od_pairs))

    # Save traffic data
    np.savez(os.path.join(data_dir, 'traffic_data.npz'),
             TM=TM,
             L=L,
             **split_info)


def main():
    """Generate and save all synthetic data."""
    print("=" * 50)
    print("Generating Synthetic Network Traffic Data")
    print("=" * 50)

    # Set random seeds
    seed = CONFIG['random_seed']
    set_all_seeds(seed)
    rng = np.random.default_rng(seed)

    # Create topology
    print("\n1. Creating network topology...")
    G, links = create_topology(
        num_nodes=CONFIG['num_nodes'],
        k=CONFIG['watts_strogatz_k'],
        p=CONFIG['watts_strogatz_p'],
        seed=seed
    )
    num_links = len(links)
    print(f"   - Nodes: {G.number_of_nodes()}")
    print(f"   - Undirected edges: {G.number_of_edges()}")
    print(f"   - Directed links: {num_links}")

    # Define OD pairs
    nodes = list(G.nodes())
    od_pairs = [(i, j) for i in nodes for j in nodes if i != j]
    num_od = len(od_pairs)
    print(f"   - OD pairs: {num_od}")

    # Compute routing matrix
    print("\n2. Computing routing matrix...")
    R = compute_routing_matrix(G, links, od_pairs)
    print(f"   - Shape: {R.shape}")
    print(f"   - Sparsity: {100 * (1 - R.sum() / R.size):.1f}%")

    # Generate traffic matrix
    print("\n3. Generating traffic matrix time series...")
    T = CONFIG['total_time_steps']
    TM = generate_traffic_matrix(T, num_od, CONFIG, rng)
    print(f"   - Shape: {TM.shape}")
    print(f"   - Time steps: {T} ({CONFIG['days']} days at {CONFIG['time_step_minutes']}-min intervals)")
    print(f"   - Traffic range: [{TM.min():.2f}, {TM.max():.2f}]")

    # Compute link loads
    print("\n4. Computing link loads...")
    L = compute_link_loads(R, TM)
    print(f"   - Shape: {L.shape}")
    print(f"   - Load range: [{L.min():.2f}, {L.max():.2f}]")

    # Split data
    print("\n5. Splitting data...")
    split_info = split_data(T, CONFIG['train_frac'], CONFIG['val_frac'])
    print(f"   - Train: {split_info['T_train']} steps ({CONFIG['train_frac']*100:.0f}%)")
    print(f"   - Val:   {split_info['T_val']} steps ({CONFIG['val_frac']*100:.0f}%)")
    print(f"   - Test:  {split_info['T_test']} steps ({CONFIG['test_frac']*100:.0f}%)")

    # Save data
    print("\n6. Saving data to disk...")
    save_data(G, links, od_pairs, R, TM, L, split_info, DATA_DIR)
    print(f"   - Saved to: {DATA_DIR}")

    print("\n" + "=" * 50)
    print("Data generation complete!")
    print("=" * 50)

    return G, links, od_pairs, R, TM, L, split_info


if __name__ == '__main__':
    main()
