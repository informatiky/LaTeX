#!/usr/bin/env python3
"""
chung_lu_sweep.py

Sweeps the power-law exponent gamma for a Chung-Lu random graph. For every
gamma value, the weight sequence w_i = c * i^(-p), p = 1/(gamma-1), i0 = 0,
is built using the LARGEST admissible scaling constant c (Fasino, Tonetto &
Tudisco 2021, Theorem 3.1). One Chung-Lu graph is generated per gamma value
(networkx's expected-degree-graph algorithm), and a row of statistics is
appended to a CSV file. Every step is logged to both the console and a log
file.

Usage example:
    python chung_lu_sweep.py --n 500 --gamma-start 2.0 --gamma-stop 3.0 \
        --gamma-step 0.1

Every run automatically creates a folder:
    ./Batch Generations/<timestamp>_n<n>_gamma<start>-<stop>_step<step>_rep<r>_seed<s>/
containing results.csv and sweep.log (filenames can be overridden with
--output / --log-file; the batch root can be overridden with --batch-root).

Notes:
    - i0 is NOT exposed as a parameter; it is fixed at 0, so c always
      coincides with w_max for the generated sequence.
    - --anchor controls how c is chosen per gamma value:
        max   (default) c = c_max(gamma) for every gamma (original behavior).
        c     c is fixed at c_max(n, gamma_start) for every gamma in the
              sweep. Since c_max(n, gamma) is monotonically increasing in
              gamma, this value is guaranteed admissible for every gamma
              >= gamma_start.
        w_min w_min is fixed at the value it takes when c = c_max(n,
              gamma_start); for every other gamma, c is then DERIVED from
              this fixed w_min via c = w_min * n^p. This is NOT the same
              sequence as anchor=c -- fixing w_min instead of c makes the
              derived c shrink sharply as gamma increases (see chat).
              Admissibility is still checked and logged every iteration.
    - "Exact max clique size" is computed via networkx's Bron-Kerbosch
      enumeration (nx.find_cliques). This is exact but can become slow for
      large or dense graphs; a warning is logged before each computation
      above a configurable node-count threshold.
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

# Above this many nodes, log a heads-up that exact clique search may be slow.
CLIQUE_SIZE_WARNING_THRESHOLD = 15000


def build_batch_folder_name(args: argparse.Namespace) -> str:
    """
    Construct a folder name from the current timestamp and the submitted
    sweep parameters, e.g.:
    20260822-141525_n500_gamma2.000-3.000_step0.100_rep1_seed42
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parts = [
        timestamp,
        f"n{args.n}",
        f"gamma{args.gamma_start:.3f}-{args.gamma_stop:.3f}",
        f"step{args.gamma_step:.3f}",
        f"anchor-{args.anchor}",
        f"rep{args.replicates}",
        f"seed{args.seed}",
    ]
    return "_".join(parts)


def setup_logging(log_path: Path) -> None:
    """Configure logging to both a file and the console."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    )

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(console_handler)


def compute_c_max(n: int, gamma: float) -> float:
    """
    Exact discrete maximum admissible c for w_1^2 <= sum(w_i).
    """
    p = 1.0 / (gamma - 1.0)
    i = np.arange(1, n + 1, dtype=float)
    return float(np.sum(i ** (-p)))


def w_min_from_c(n: int, gamma: float, c: float) -> float:
    """w_min = w_n = c * n^(-p), the smallest weight in the sequence."""
    p = 1.0 / (gamma - 1.0)
    return c * n ** (-p)


def c_from_w_min(n: int, gamma: float, w_min: float) -> float:
    """Inverse of w_min_from_c: c = w_min * n^p."""
    p = 1.0 / (gamma - 1.0)
    return w_min * n ** p


def build_weight_sequence(n: int, gamma: float, c: float) -> np.ndarray:
    """Weight sequence w_i = c * i^(-p) for i = 1..n (i0 = 0)."""
    p = 1.0 / (gamma - 1.0)
    i = np.arange(1, n + 1, dtype=float)
    return c * i ** (-p)


def generate_chung_lu_graph(weights: np.ndarray, seed: int) -> nx.Graph:
    """
    Generate a simple Chung-Lu graph from an expected-degree sequence using
    networkx's expected_degree_graph (Miller-Hagberg fast algorithm), which
    implements p_ij = min(1, w_i*w_j / S). Self-loops are excluded.
    """
    return nx.expected_degree_graph(weights.tolist(), seed=seed, selfloops=False)


def exact_max_clique_size(G: nx.Graph) -> int:
    """Exact maximum clique size via Bron-Kerbosch enumeration."""
    if G.number_of_nodes() == 0:
        return 0
    return max((len(c) for c in nx.find_cliques(G)), default=0)


def degree_stats(G: nx.Graph) -> dict:
    degrees = np.array([d for _, d in G.degree()], dtype=float)
    if degrees.size == 0:
        return dict(degree_mean=np.nan, degree_median=np.nan,
                     degree_std=np.nan, degree_min=np.nan, degree_max=np.nan)
    return dict(
        degree_mean=float(np.mean(degrees)),
        degree_median=float(np.median(degrees)),
        degree_std=float(np.std(degrees, ddof=1)) if degrees.size > 1 else 0.0,
        degree_min=float(np.min(degrees)),
        degree_max=float(np.max(degrees)),
    )


def run_sweep(n: int, gammas: np.ndarray, seed: int, replicates: int,
              anchor: str) -> pd.DataFrame:
    logging.info(
        "Starting sweep: n=%d, gamma values=%d, replicates per gamma=%d, "
        "base seed=%d, anchor=%s",
        n, len(gammas), replicates, seed, anchor,
    )

    gamma_ref = float(gammas[0])
    c_ref = None
    w_min_ref = None
    if anchor in ("c", "w_min"):
        c_ref = compute_c_max(n, gamma_ref)
        w_min_ref = w_min_from_c(n, gamma_ref, c_ref)
        logging.info(
            "Anchor reference computed at gamma_ref=%.4f (lowest gamma in sweep): "
            "c_ref=%.6f, w_min_ref=%.8f",
            gamma_ref, c_ref, w_min_ref,
        )

    rows = []
    for gamma_idx, gamma in enumerate(gammas):
        logging.info("---- gamma = %.4f (%d/%d) ----", gamma, gamma_idx + 1, len(gammas))

        p = 1.0 / (gamma - 1.0)
        logging.info("Computed p = 1/(gamma-1) = %.6f", p)

        if anchor == "max":
            c = compute_c_max(n, gamma)
            logging.info("Computed maximum admissible c (i0=0): c_max = %.6f", c)
        elif anchor == "c":
            c = c_ref
            logging.info("Using fixed c (anchored at gamma_ref=%.4f): c = %.6f",
                         gamma_ref, c)
        elif anchor == "w_min":
            c = c_from_w_min(n, gamma, w_min_ref)
            logging.info(
                "Using fixed w_min (anchored at gamma_ref=%.4f, w_min=%.8f): "
                "derived c = %.6f",
                gamma_ref, w_min_ref, c,
            )
        else:
            raise ValueError(f"Unknown anchor mode: {anchor}")

        weights = build_weight_sequence(n, gamma, c)
        S = float(np.sum(weights))
        w_max = float(weights[0])
        w_min = float(weights[-1])
        epsilon = 1.0 - w_max / np.sqrt(S)
        logging.info(
            "Weight sequence built: W (sum of weights) = %.4f, w_max = %.4f, "
            "w_min = %.4f, epsilon = %.6f",
            S, w_max, w_min, epsilon,
        )
        if w_max ** 2 > S:
            logging.warning(
                "Admissibility check FAILED (w_max^2=%.4f > S=%.4f) -- "
                "this should not happen when using c_max; please verify.",
                w_max ** 2, S,
            )
        else:
            logging.info("Admissibility check passed (w_max^2 <= S).")

        for rep in range(replicates):
            rep_seed = seed + gamma_idx * 1000 + rep
            logging.info("Generating Chung-Lu graph (replicate %d/%d, seed=%d)...",
                         rep + 1, replicates, rep_seed)
            t0 = time.time()
            G = generate_chung_lu_graph(weights, seed=rep_seed)
            gen_time = time.time() - t0

            num_nodes = G.number_of_nodes()
            num_edges = G.number_of_edges()
            density = nx.density(G)
            logging.info(
                "Graph generated in %.3fs: nodes=%d, edges=%d, density=%.6f",
                gen_time, num_nodes, num_edges, density,
            )

            dstats = degree_stats(G)
            logging.info(
                "Degree stats: mean=%.3f, median=%.3f, std=%.3f, min=%.0f, max=%.0f",
                dstats["degree_mean"], dstats["degree_median"], dstats["degree_std"],
                dstats["degree_min"], dstats["degree_max"],
            )

            if num_nodes > CLIQUE_SIZE_WARNING_THRESHOLD:
                logging.warning(
                    "n=%d exceeds warning threshold (%d) for exact clique search; "
                    "this may take a while.",
                    num_nodes, CLIQUE_SIZE_WARNING_THRESHOLD,
                )
            t0 = time.time()
            clique_size = exact_max_clique_size(G)
            clique_time = time.time() - t0
            logging.info(
                "Exact max clique size = %d (computed in %.3fs)",
                clique_size, clique_time,
            )

            rows.append(dict(
                gamma=gamma,
                n_target=n,
                replicate=rep,
                seed=rep_seed,
                anchor=anchor,
                gamma_ref=gamma_ref,
                p_exponent=p,
                c_used=c,
                W_sum_weights=S,
                w_max=w_max,
                w_min=w_min,
                epsilon=epsilon,
                num_nodes=num_nodes,
                num_edges=num_edges,
                density=density,
                exact_max_clique_size=clique_size,
                degree_mean=dstats["degree_mean"],
                degree_median=dstats["degree_median"],
                degree_std=dstats["degree_std"],
                degree_min=dstats["degree_min"],
                degree_max=dstats["degree_max"],
                graph_gen_time_s=gen_time,
                clique_time_s=clique_time,
            ))

    logging.info("Sweep complete: %d rows generated.", len(rows))
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chung-Lu gamma sweep with maximum admissible c (i0=0)."
    )
    parser.add_argument("--n", type=int, required=True,
                         help="Number of nodes.")
    parser.add_argument("--gamma-start", type=float, required=True,
                         help="First gamma value (inclusive).")
    parser.add_argument("--gamma-stop", type=float, required=True,
                         help="Last gamma value (inclusive).")
    parser.add_argument("--gamma-step", type=float, default=0.1,
                         help="Step size between gamma values (default: 0.1).")
    parser.add_argument("--replicates", type=int, default=1,
                         help="Number of graph replicates per gamma value (default: 1).")
    parser.add_argument("--seed", type=int, default=42,
                         help="Base random seed (default: 42).")
    parser.add_argument("--anchor", choices=["max", "c", "w_min"], default="max",
                         help="How c is chosen per gamma: 'max' = c_max(gamma) "
                              "every time (default); 'c' = c fixed at its "
                              "gamma_start value for the whole sweep; 'w_min' "
                              "= w_min fixed at its gamma_start value, c "
                              "derived per gamma. See module docstring.")
    parser.add_argument("--batch-root", type=Path, default=Path("./Batch Generations"),
                         help="Root folder under which a timestamped batch "
                              "subfolder is created (default: ./Batch Generations).")
    parser.add_argument("--output", type=Path, default=Path("results.csv"),
                         help="CSV filename, placed inside the batch subfolder "
                              "unless given as an absolute path (default: results.csv).")
    parser.add_argument("--log-file", type=Path, default=Path("sweep.log"),
                         help="Log filename, placed inside the batch subfolder "
                              "unless given as an absolute path (default: sweep.log).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    batch_folder_name = build_batch_folder_name(args)
    batch_dir = args.batch_root / batch_folder_name
    batch_dir.mkdir(parents=True, exist_ok=True)

    # If --output / --log-file were given as absolute paths, Path's "/"
    # operator will honor that and ignore batch_dir; otherwise they are
    # placed inside the batch folder.
    output_path = batch_dir / args.output
    log_path = batch_dir / args.log_file

    setup_logging(log_path)

    logging.info("chung_lu_sweep.py started")
    logging.info("Batch folder: %s", batch_dir.resolve())
    logging.info(
        "Parameters: n=%d, gamma in [%.4f, %.4f] step %.4f, replicates=%d, "
        "seed=%d, anchor=%s",
        args.n, args.gamma_start, args.gamma_stop, args.gamma_step,
        args.replicates, args.seed, args.anchor,
    )

    if args.n < 2:
        logging.error("n must be at least 2.")
        sys.exit(1)
    if args.gamma_step <= 0:
        logging.error("gamma-step must be positive.")
        sys.exit(1)
    if args.gamma_start <= 1.0:
        logging.error("gamma-start must be > 1 (p = 1/(gamma-1) requires this).")
        sys.exit(1)

    n_steps = int(round((args.gamma_stop - args.gamma_start) / args.gamma_step)) + 1
    if n_steps < 1:
        logging.error("gamma-stop must be >= gamma-start.")
        sys.exit(1)
    gammas = args.gamma_start + args.gamma_step * np.arange(n_steps)
    gammas = gammas[gammas <= args.gamma_stop + 1e-9]
    logging.info("Gamma values to sweep: %s", np.round(gammas, 4).tolist())

    df = run_sweep(args.n, gammas, args.seed, args.replicates, args.anchor)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info("Results written to %s", output_path.resolve())
    logging.info("Log written to %s", log_path.resolve())
    logging.info("chung_lu_sweep.py finished successfully")


if __name__ == "__main__":
    main()