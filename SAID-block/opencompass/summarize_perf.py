#!/usr/bin/env python3
"""
Summarize [Perf] / [Perf-SAID] timing from opencompass log files.

Usage:
    python summarize_perf.py <outputs_dir>

Example:
    python summarize_perf.py outputs/default/20260323_092117
"""

import sys
import os
import re
from collections import defaultdict

def parse_logs(output_dir):
    logs_dir = os.path.join(output_dir, 'logs', 'infer')
    if not os.path.isdir(logs_dir):
        print(f"Error: {logs_dir} not found")
        sys.exit(1)

    # model_name -> dataset_task -> list of (time, tokens)
    results = defaultdict(lambda: defaultdict(lambda: {'time': 0.0, 'tokens': 0, 'samples': 0}))

    for model_name in os.listdir(logs_dir):
        model_dir = os.path.join(logs_dir, model_name)
        if not os.path.isdir(model_dir):
            continue
        for log_file in sorted(os.listdir(model_dir)):
            if not log_file.endswith('.out'):
                continue
            task_name = log_file.replace('.out', '')
            filepath = os.path.join(model_dir, log_file)

            with open(filepath, 'r', errors='ignore') as f:
                for line in f:
                    # Match [Perf] or [Perf-SAID] lines
                    m = re.search(
                        r'\[Perf(?:-SAID)?\]\s+'
                        r'batch_size=(\d+),\s*gen_length=(\d+),\s*'
                        r'total_tokens=(\d+),\s*time=([\d.]+)s',
                        line
                    )
                    if m:
                        batch_size = int(m.group(1))
                        tokens = int(m.group(3))
                        time_s = float(m.group(4))
                        results[model_name][task_name]['time'] += time_s
                        results[model_name][task_name]['tokens'] += tokens
                        results[model_name][task_name]['samples'] += batch_size

    return results


def print_summary(results):
    for model_name in sorted(results.keys()):
        tasks = results[model_name]
        print("=" * 80)
        print(f"Model: {model_name}")
        print("=" * 80)
        print(f"{'Task':<40} {'Samples':>8} {'Tokens':>10} {'Time(s)':>10} {'Tok/s':>10}")
        print("-" * 80)

        total_time = 0.0
        total_tokens = 0
        total_samples = 0

        for task_name in sorted(tasks.keys()):
            d = tasks[task_name]
            tok_s = d['tokens'] / d['time'] if d['time'] > 0 else 0
            print(f"{task_name:<40} {d['samples']:>8} {d['tokens']:>10} {d['time']:>10.2f} {tok_s:>10.2f}")
            total_time += d['time']
            total_tokens += d['tokens']
            total_samples += d['samples']

        print("-" * 80)
        avg_tok_s = total_tokens / total_time if total_time > 0 else 0
        avg_latency = total_time / total_samples if total_samples > 0 else 0
        print(f"{'TOTAL':<40} {total_samples:>8} {total_tokens:>10} {total_time:>10.2f} {avg_tok_s:>10.2f}")
        print(f"Average latency/sample: {avg_latency:.3f}s")
        print()


def main():
    if len(sys.argv) < 2:
        # Auto-find latest output dir
        base = 'outputs/default'
        if os.path.isdir(base):
            dirs = sorted(os.listdir(base))
            if dirs:
                output_dir = os.path.join(base, dirs[-1])
            else:
                print(f"Usage: python {sys.argv[0]} <outputs_dir>")
                sys.exit(1)
        else:
            print(f"Usage: python {sys.argv[0]} <outputs_dir>")
            sys.exit(1)
    else:
        output_dir = sys.argv[1]

    print(f"Scanning: {output_dir}")
    print()
    results = parse_logs(output_dir)

    if not results:
        print("No [Perf] / [Perf-SAID] entries found in logs.")
        sys.exit(0)

    print_summary(results)

    # ── Cross-model comparison ──
    model_names = sorted(results.keys())
    if len(model_names) >= 2:
        print("=" * 80)
        print("Speed Comparison")
        print("=" * 80)
        for mn in model_names:
            tasks = results[mn]
            t = sum(d['time'] for d in tasks.values())
            tok = sum(d['tokens'] for d in tasks.values())
            sam = sum(d['samples'] for d in tasks.values())
            tok_s = tok / t if t > 0 else 0
            print(f"  {mn:<40} total_time={t:.2f}s  avg_tok/s={tok_s:.2f}")

        # Speedup
        base_model = model_names[0]
        base_time = sum(d['time'] for d in results[base_model].values())
        for mn in model_names[1:]:
            mn_time = sum(d['time'] for d in results[mn].values())
            if mn_time > 0:
                speedup = base_time / mn_time
                print(f"\n  Speedup: {mn} vs {base_model} = {speedup:.2f}x")


if __name__ == '__main__':
    main()
