"""Collect timing.json files from all runs and print a summary table.

Usage:
    python -m sep.collect_timing \
        --gen-base "$SEP_SCRATCH" \
        --transfer-base "$SEP_SCRATCH/transfer_results"
"""
import argparse
import glob
import json
import os
from datetime import datetime


def mins(t1, t2):
    fmt = "%Y-%m-%d %H:%M:%S"
    return round((datetime.strptime(t2, fmt) - datetime.strptime(t1, fmt)).total_seconds() / 60, 1)


def secs(t1, t2):
    fmt = "%Y-%m-%d %H:%M:%S"
    return round((datetime.strptime(t2, fmt) - datetime.strptime(t1, fmt)).total_seconds())


def find_timing_json(run_dir):
    """Find timing.json recursively under a wandb run dir."""
    matches = glob.glob(os.path.join(run_dir, '**/timing.json'), recursive=True)
    return matches[0] if matches else None


def collect_gen_timings(base):
    """Collect t1/t2a/t2b for each dataset/model from generate_answers runs."""
    results = {}
    for dataset in ['nq', 'squad']:
        for run_dir in sorted(glob.glob(os.path.join(base, f'{dataset}_all_*'))):
            for model_dir in sorted(glob.glob(os.path.join(run_dir, '*'))):
                model = os.path.basename(model_dir)
                if model == 'logs':
                    continue
                tf = find_timing_json(model_dir)
                if not tf:
                    continue
                try:
                    with open(tf) as f:
                        t = json.load(f)
                except Exception:
                    continue
                key = (dataset, model)
                if key not in results:
                    results[key] = t
    return results


def collect_transfer_timings(base):
    """Collect t3/t4/t5 for each transfer pair from transfer runs."""
    results = {}
    for dataset in ['nq', 'squad']:
        for pair_dir in sorted(glob.glob(os.path.join(base, dataset, '*'))):
            pair = os.path.basename(pair_dir)
            tf = os.path.join(pair_dir, 'timing.json')
            if not os.path.exists(tf):
                continue
            with open(tf) as f:
                t = json.load(f)
            results[(dataset, pair)] = t
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gen-base', default=os.getenv('SEP_SCRATCH', ''))
    p.add_argument('--transfer-base',
                   default=os.path.join(os.getenv('SEP_SCRATCH', ''), 'transfer_results'))
    p.add_argument('--out', default='timing_summary.json',
                   help='Output JSON path for the full summary table')
    args = p.parse_args()

    gen = collect_gen_timings(args.gen_base)
    transfer = collect_transfer_timings(args.transfer_base)

    summary = []

    # ---------- generation timing (t1, t2a, t2b) ----------
    print('\n' + '=' * 75)
    print('  Generation timing (t1 + t2a + t2b)')
    print('=' * 75)
    print(f"  {'Dataset':<8} {'Model':<16} {'t1 11x(min)':>12} {'t2a SE(min)':>12} {'t2b probe(s)':>13}")
    print('  ' + '-' * 65)
    POOL = 1500 / 2000
    for (dataset, model), t in sorted(gen.items()):
        t1 = t.get('t1_inference_start'), t.get('t1_inference_end')
        t2a = t.get('t2a_clustering_se_start'), t.get('t2a_clustering_se_end')
        t2b = t.get('t2b_probe_start'), t.get('t2b_probe_end')
        t1_min = mins(*t1) * POOL if all(t1) else None
        t2a_min = mins(*t2a) if all(t2a) else None
        t2b_sec = secs(*t2b) if all(t2b) else None
        t1_s = f'{t1_min:.1f}' if t1_min is not None else 'N/A'
        t2a_s = f'{t2a_min:.1f}' if t2a_min is not None else 'N/A'
        t2b_s = f'{t2b_sec}s' if t2b_sec is not None else 'N/A'
        print(f"  {dataset:<8} {model:<16} {t1_s:>12} {t2a_s:>12} {t2b_s:>13}")
        summary.append({'dataset': dataset, 'model': model,
                        't1_11x_1500_min': t1_min,
                        't2a_clustering_se_min': t2a_min,
                        't2b_probe_sec': t2b_sec})

    # ---------- transfer timing (t3, t4, t5) ----------
    print('\n' + '=' * 85)
    print('  Transfer timing (t3 src probe | t4 tgt probe | t5 ridge n=1500)')
    print('=' * 85)
    print(f"  {'Dataset':<8} {'Pair':<35} {'t3 src(s)':>10} {'t4 tgt(s)':>10} {'t5 ridge(s)':>12}")
    print('  ' + '-' * 78)
    for (dataset, pair), t in sorted(transfer.items()):
        t3 = t.get('t3_src_probe_start'), t.get('t3_src_probe_end')
        t4 = t.get('t4_tgt_probe_start'), t.get('t4_tgt_probe_end')
        # use n=1500 ridge if available, else largest n
        t5_keys = sorted([k for k in t if k.startswith('t5_ridge_n') and k.endswith('_start')])
        last_n = t5_keys[-1].replace('_start', '') if t5_keys else None
        t5 = (t.get(f'{last_n}_start'), t.get(f'{last_n}_end')) if last_n else (None, None)
        t3_s = f'{secs(*t3)}s' if all(t3) else 'N/A'
        t4_s = f'{secs(*t4)}s' if all(t4) else 'N/A'
        t5_s = f'{secs(*t5)}s' if all(t5) else 'N/A'
        print(f"  {dataset:<8} {pair:<35} {t3_s:>10} {t4_s:>10} {t5_s:>12}")
        summary.append({'dataset': dataset, 'pair': pair,
                        't3_src_probe_sec': secs(*t3) if all(t3) else None,
                        't4_tgt_probe_sec': secs(*t4) if all(t4) else None,
                        't5_ridge_1500_sec': secs(*t5) if all(t5) else None})

    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nFull summary saved to {args.out}')


if __name__ == '__main__':
    main()
