"""
Part D.2 -- empirical benchmarking and load testing.

Measures response-latency percentiles (P50/P90/P99), request throughput,
and server-process RAM/CPU, under both single-client (sequential) and
concurrent multi-client traffic patterns, against any target base URL --
run once against a local instance and once against the live cloud
deployment (see README/Report for both result sets).
"""
import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

try:
    import psutil
except ImportError:
    psutil = None

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

ENDPOINTS = {
    "sentiment": dict(path="/api/v1/sentiment", payload={"text": "This benchmark run is going quite well so far."}),
    "summarize": dict(path="/api/v1/summarize", payload={
        "text": "Load testing is the process of putting demand on a software system and measuring its "
                "response. It is performed to determine a system's behavior under both normal and "
                "anticipated peak load conditions, and to identify the breaking point of a system.",
        "max_length": 40, "min_length": 10}),
    "generate": dict(path="/api/v1/generate", payload={
        "prompt": "The results of this benchmark show that", "max_new_tokens": 20,
        "decoding_strategy": "top_p"}),
}


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


async def single_request(client: httpx.AsyncClient, base_url: str, endpoint: str):
    spec = ENDPOINTS[endpoint]
    t0 = time.perf_counter()
    try:
        resp = await client.post(base_url + spec["path"], json=spec["payload"], timeout=60.0)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return dict(ok=resp.status_code == 200, status=resp.status_code, latency_ms=elapsed_ms)
    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return dict(ok=False, status=None, latency_ms=elapsed_ms, error=str(e))


async def run_pattern(base_url: str, endpoint: str, n_requests: int, concurrency: int, monitor_pid=None):
    """concurrency=1 -> single-client sequential; concurrency>1 -> that many
    requests in flight at once, repeated until n_requests total complete."""
    results = []
    cpu_samples, mem_samples = [], []
    proc = psutil.Process(monitor_pid) if (psutil and monitor_pid) else None

    async def sampler(stop_event):
        while not stop_event.is_set():
            if proc:
                try:
                    cpu_samples.append(proc.cpu_percent(interval=None))
                    mem_samples.append(proc.memory_info().rss / (1024 * 1024))
                except psutil.NoSuchProcess:
                    pass
            await asyncio.sleep(0.2)

    stop_event = asyncio.Event()
    sampler_task = asyncio.create_task(sampler(stop_event)) if proc else None

    t0 = time.perf_counter()
    async with httpx.AsyncClient() as client:
        if concurrency == 1:
            for _ in range(n_requests):
                results.append(await single_request(client, base_url, endpoint))
        else:
            remaining = n_requests
            while remaining > 0:
                batch = min(concurrency, remaining)
                batch_results = await asyncio.gather(
                    *[single_request(client, base_url, endpoint) for _ in range(batch)])
                results.extend(batch_results)
                remaining -= batch
    total_time_s = time.perf_counter() - t0

    if sampler_task:
        stop_event.set()
        await sampler_task

    latencies = [r["latency_ms"] for r in results if r["ok"]]
    n_errors = sum(1 for r in results if not r["ok"])
    summary = dict(
        endpoint=endpoint, concurrency=concurrency, n_requests=n_requests,
        n_success=len(latencies), n_errors=n_errors,
        total_time_s=round(total_time_s, 3),
        throughput_rps=round(len(latencies) / total_time_s, 3) if total_time_s > 0 else None,
        p50_ms=round(percentile(latencies, 50), 2) if latencies else None,
        p90_ms=round(percentile(latencies, 90), 2) if latencies else None,
        p99_ms=round(percentile(latencies, 99), 2) if latencies else None,
        min_ms=round(min(latencies), 2) if latencies else None,
        max_ms=round(max(latencies), 2) if latencies else None,
        avg_cpu_percent=round(statistics.mean(cpu_samples), 1) if cpu_samples else None,
        avg_rss_mb=round(statistics.mean(mem_samples), 1) if mem_samples else None,
        peak_rss_mb=round(max(mem_samples), 1) if mem_samples else None,
    )
    return summary


async def main_async(args):
    base_url = args.base_url.rstrip("/")
    all_results = []
    for endpoint in ENDPOINTS:
        for concurrency, n in [(1, args.n_sequential), (args.concurrency, args.n_concurrent)]:
            print(f"Running: endpoint={endpoint} concurrency={concurrency} n={n} against {base_url}")
            summary = await run_pattern(base_url, endpoint, n, concurrency, monitor_pid=args.pid)
            print(f"  -> P50={summary['p50_ms']}ms P90={summary['p90_ms']}ms P99={summary['p99_ms']}ms "
                  f"throughput={summary['throughput_rps']}req/s errors={summary['n_errors']}")
            all_results.append(summary)

    out_path = RESULTS_DIR / f"{args.label}.json"
    with open(out_path, "w") as f:
        json.dump(dict(base_url=base_url, label=args.label, results=all_results), f, indent=2)
    print(f"\nWrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Load test the Task 28 LLM microservice.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--label", default="local", help="output file label, e.g. 'local' or 'cloud'")
    parser.add_argument("--n-sequential", type=int, default=15, help="requests for the single-client pattern")
    parser.add_argument("--n-concurrent", type=int, default=30, help="requests for the concurrent pattern")
    parser.add_argument("--concurrency", type=int, default=6, help="concurrent in-flight requests")
    parser.add_argument("--pid", type=int, default=None, help="server process PID to sample CPU/RAM from (local only)")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
