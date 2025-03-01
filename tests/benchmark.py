"""Benchmarking DTW."""

import argparse

import torch
from torch.utils.benchmark import Compare, Timer
from tqdm import tqdm


def benchmark_dtw(dims: list[int], num_threads: int, min_run_time: float) -> Compare:
    """Benchmark DTW on input of size (n, n) and (n, n, n, n)."""
    results = []
    for n in (pbar := tqdm(dims, leave=False)):
        pbar.set_postfix(n=n)
        x = torch.randn((n, n))
        d = torch.randn((n, n, n, n))
        sx = torch.ones(n, dtype=torch.long) * n
        t_cpu = Timer(
            stmt="dtw(x)",
            setup="from fastabx.dtw import dtw",
            globals={"x": x},
            num_threads=num_threads,
            label="2D DTW",
            sub_label=str(n),
            description="cpu",
        )
        t_gpu = Timer(
            stmt="dtw(x)",
            setup="from fastabx.dtw import dtw",
            globals={"x": x.cuda()},
            num_threads=num_threads,
            label="2D DTW",
            sub_label=str(n),
            description="gpu",
        )
        t_gpu_compiled = Timer(
            stmt="u(x)",
            setup="from fastabx.dtw import dtw; import torch; u = torch.compile(dtw, dynamic=True)",
            globals={"x": x.cuda()},
            num_threads=num_threads,
            label="2D DTW",
            sub_label=str(n),
            description="gpu - compiled",
        )
        m_cpu = t_cpu.blocked_autorange(min_run_time=min_run_time)
        m_gpu = t_gpu.blocked_autorange(min_run_time=min_run_time)
        m_gpu_compiled = t_gpu_compiled.blocked_autorange(min_run_time=min_run_time)
        results += [m_cpu, m_gpu, m_gpu_compiled]

        t_cpu_batch = Timer(
            stmt="dtw_batch(d, sx, sy, symmetric=False)",
            setup="from fastabx.dtw import dtw_batch",
            globals={"d": d, "sx": sx, "sy": sx},
            num_threads=num_threads,
            label="Batched DTW",
            sub_label=str(n),
            description="cpu",
        )
        t_gpu_batch = Timer(
            stmt="dtw_batch(d, sx, sy, symmetric=False)",
            setup="from fastabx.dtw import dtw_batch",
            globals={"d": d.cuda(), "sx": sx.cuda(), "sy": sx.cuda()},
            num_threads=num_threads,
            label="Batched DTW",
            sub_label=str(n),
            description="gpu",
        )
        t_gpu_batch_compiled = Timer(
            stmt="v(d, sx, sy, symmetric=False)",
            setup="from fastabx.dtw import dtw_batch; import torch; v = torch.compile(dtw_batch, dynamic=True)",
            globals={"d": d.cuda(), "sx": sx.cuda(), "sy": sx.cuda()},
            num_threads=num_threads,
            label="Batched DTW",
            sub_label=str(n),
            description="gpu - compiled",
        )
        m_cpu_batch = t_cpu_batch.blocked_autorange(min_run_time=min_run_time)
        m_gpu_batch = t_gpu_batch.blocked_autorange(min_run_time=min_run_time)
        m_gpu_batch_compiled = t_gpu_batch_compiled.blocked_autorange(min_run_time=min_run_time)
        results += [m_cpu_batch, m_gpu_batch, m_gpu_batch_compiled]
    return Compare(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare execution time of CPU and CUDA DTW")
    parser.add_argument("--nthreads", default=torch.get_num_threads(), type=int)
    parser.add_argument("--min-runtime", default=0.2, type=float)
    parser.add_argument("--sizes", type=int, nargs="+")
    args = parser.parse_args()

    sizes = [1, *list(range(8, 129, 8))] if args.sizes is None else args.sizes
    compare = benchmark_dtw(sizes, args.nthreads, args.min_runtime)
    compare.colorize(rowwise=True)
    compare.print()
