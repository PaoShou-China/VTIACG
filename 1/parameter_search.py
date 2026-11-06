"""Bayesian search for the virtual-target controller parameters.
Run from this folder with python parameter_search.py.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from tqdm import tqdm

from simulation import our_guidance_data, parameter_list

search_space = [
    Real(0, 0.2, name="k1"),
    Real(0.1, 0.6, name="k2"),
    Real(0, 0.2, name="k3"),
    Real(1, 6, name="N"),
]
_worker_pool = None
_evaluation_count = 0


def _silence_worker_stdout():
    """Suppress simulation diagnostics once per worker process."""
    sys.stdout = open(os.devnull, "w")


def _evaluate_case(job):
    parameter, k1, k2, k3, N = job
    data = our_guidance_data(parameter, k1, k2, k3, N)
    miss_distance = float(np.hypot(data["x"][-1], data["y"][-1]))
    e_theta = float((data["e_theta"][-1] + np.pi) % (2 * np.pi) - np.pi)
    if miss_distance < 12 and e_theta < 0.017:
        return float(data["total_a"])
    return 1e8


def _validate_cases():
    for index, parameter in enumerate(parameter_list):
        try:
            values = np.asarray(parameter, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"case {index} must contain four finite numeric values") from exc
        if values.shape != (4,) or not np.isfinite(values).all():
            raise ValueError(f"case {index} must contain four finite numeric values")


def _positive_int(value):
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


@use_named_args(search_space)
def objective(**params):
    """Return the summed objective across configured simulation cases."""
    global _evaluation_count
    jobs = [(p, params["k1"], params["k2"], params["k3"], params["N"])
            for p in parameter_list]
    if not jobs:
        raise ValueError("simulation.parameter_list is empty")

    if _worker_pool is None:
        with Pool(min(len(jobs), os.cpu_count() or 1),
                  initializer=_silence_worker_stdout) as pool:
            values = list(tqdm(pool.imap(_evaluate_case, jobs), total=len(jobs)))
    else:
        values = list(tqdm(_worker_pool.imap(_evaluate_case, jobs), total=len(jobs)))
    _evaluation_count += 1
    score = sum(values)
    print(f"objective evaluation {_evaluation_count}: {score:.6g}")
    return score


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=_positive_int, default=100,
                        help="number of Bayesian optimization evaluations")
    parser.add_argument("--seed", type=int, default=42,
                        help="optimizer random seed")
    parser.add_argument("--workers", type=_positive_int,
                        default=min(len(parameter_list), os.cpu_count() or 1),
                        help="maximum simulation worker processes")
    parser.add_argument("--output", help="optional JSON output path for the best result")
    args = parser.parse_args(argv)
    if args.calls < 10:
        parser.error("--calls must be at least 10 for the optimizer's default initial design")
    if not parameter_list:
        parser.error("simulation.parameter_list is empty")
    try:
        _validate_cases()
    except ValueError as exc:
        parser.error(str(exc))

    global _worker_pool
    _worker_pool = Pool(min(args.workers, len(parameter_list)),
                        initializer=_silence_worker_stdout)
    try:
        result = gp_minimize(objective, search_space, n_calls=args.calls,
                             random_state=args.seed)
    finally:
        _worker_pool.close()
        _worker_pool.join()
        _worker_pool = None

    report = {"parameters": dict(zip([d.name for d in search_space], result.x)),
              "objective": float(result.fun), "calls": args.calls, "seed": args.seed}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        output_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(report, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
    return result


if __name__ == "__main__":
    main()
