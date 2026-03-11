#!/usr/bin/env python3
"""Precompute and cache graph structures for faster training."""

import argparse
import hashlib
import pickle
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from ase.db import connect
from ase.neighborlist import neighbor_list
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mp_data_pipeline.config import DB_PATH
from mp_data_pipeline.ml.splits import load_split


def compute_graph_for_structure(args):
    """Worker function to compute graph for a single structure."""
    db_path, mp_id, cutoff, max_neighbors = args
    
    try:
        db = connect(str(db_path))
        row = db.get(mp_id=mp_id)
        atoms = row.toatoms()
        
        # Compute neighbor list
        i, j, d = neighbor_list("ijd", atoms, cutoff)
        
        # Limit neighbors per atom
        edge_index = []
        edge_dist = []
        
        for atom_idx in range(len(atoms)):
            mask = i == atom_idx
            neighbors_i = j[mask]
            distances_i = d[mask]
            
            if len(neighbors_i) > max_neighbors:
                # Keep closest neighbors
                sorted_indices = np.argsort(distances_i)[:max_neighbors]
                neighbors_i = neighbors_i[sorted_indices]
                distances_i = distances_i[sorted_indices]
            
            for neighbor_j, dist in zip(neighbors_i, distances_i):
                edge_index.append([atom_idx, neighbor_j])
                edge_dist.append(dist)
        
        edge_index = np.array(edge_index, dtype=np.int64).T if edge_index else np.zeros((2, 0), dtype=np.int64)
        edge_dist = np.array(edge_dist, dtype=np.float32) if edge_dist else np.zeros(0, dtype=np.float32)
        
        return mp_id, (edge_index, edge_dist)
    
    except Exception as e:
        print(f"Error processing {mp_id}: {e}", flush=True)
        return mp_id, None


def main():
    parser = argparse.ArgumentParser(description="Precompute graph structures")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="ASE DB path")
    parser.add_argument(
        "--split",
        type=Path,
        default=PROJECT_ROOT / "data" / "splits" / "split_iid_seed42.json",
        help="Split JSON path",
    )
    parser.add_argument("--cutoff", type=float, default=6.0, help="Cutoff radius")
    parser.add_argument("--max-neighbors", type=int, default=24, help="Max neighbors per atom")
    parser.add_argument("--workers", type=int, default=None, help="Number of workers (default: CPU count)")
    args = parser.parse_args()
    
    # Load split
    split = load_split(args.split)
    all_mp_ids = split["train"] + split["val"] + split["test"]
    
    print(f"Precomputing graphs for {len(all_mp_ids)} structures...", flush=True)
    print(f"Cutoff: {args.cutoff}, Max neighbors: {args.max_neighbors}", flush=True)
    
    # Create cache key
    cache_key = hashlib.md5(
        f"{args.db}_{args.cutoff}_{args.max_neighbors}".encode()
    ).hexdigest()
    cache_file = PROJECT_ROOT / "data" / "cache" / f"graphs_{cache_key}.pkl"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if cache exists
    if cache_file.exists():
        print(f"Cache already exists: {cache_file}", flush=True)
        print("Delete it if you want to recompute.", flush=True)
        return
    
    # Prepare worker arguments
    n_workers = args.workers or min(cpu_count(), 16)
    print(f"Using {n_workers} workers...", flush=True)
    
    worker_args = [
        (args.db, mp_id, args.cutoff, args.max_neighbors)
        for mp_id in all_mp_ids
    ]
    
    # Compute graphs in parallel with progress bar
    start = time.time()
    graphs = {}
    
    print("Starting parallel computation...", flush=True)
    with Pool(n_workers) as pool:
        # Use imap_unordered for progress tracking
        results_iter = pool.imap_unordered(compute_graph_for_structure, worker_args, chunksize=100)
        
        # Collect results with progress bar
        failed = 0
        for mp_id, graph_data in tqdm(results_iter, total=len(all_mp_ids), desc="Computing graphs"):
            if graph_data is not None:
                graphs[mp_id] = graph_data
            else:
                failed += 1
    
    elapsed = time.time() - start
    print(f"✅ Computed {len(graphs)} graphs in {elapsed:.1f}s ({len(graphs)/elapsed:.1f} graphs/sec)", flush=True)
    if failed > 0:
        print(f"⚠️ Failed: {failed} structures", flush=True)
    
    # Save to cache
    print(f"Saving to cache: {cache_file}", flush=True)
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'mp_ids': all_mp_ids,
                'cutoff': args.cutoff,
                'max_neighbors': args.max_neighbors,
                'graphs': graphs
            }, f)
        print(f"✅ Cache saved ({cache_file.stat().st_size / 1024 / 1024:.1f} MB)", flush=True)
    except Exception as e:
        print(f"❌ Failed to save cache: {e}", flush=True)
        raise


if __name__ == "__main__":
    main()
