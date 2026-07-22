"""Merge data-parallel generation shards and run the compute stage once.

Each shard produced by `sep-generate --num_shards N --shard_index i` writes its
own `{train,validation}_generations.pkl` and (for validation) an
`uncertainty_measures.pkl` holding precomputed p_false. This driver concatenates
those partial files in shard order, drops the merged pkls into a fresh offline
wandb run directory, and invokes `compute_uncertainty_measures` with
`assign_new_wandb_id=False` so it reads the merged files locally (no wandb API).
"""
import glob
import logging
import os
import pickle

import wandb

from sep.compute_uncertainty_measures import main as main_compute
from sep.uncertainty.utils import utils


def _find_pkl(shard_dir, filename):
    matches = glob.glob(f'{shard_dir}/**/{filename}', recursive=True)
    if not matches:
        return None
    # Newest wins if a shard dir was reused across runs.
    return max(matches, key=os.path.getmtime)


def _load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def merge_generations(shard_dirs, filename):
    """Concatenate per-shard generation dicts in shard order."""
    merged = {}
    total = 0
    for shard_dir in shard_dirs:
        path = _find_pkl(shard_dir, filename)
        if path is None:
            logging.info('No %s in %s; skipping.', filename, shard_dir)
            continue
        part = _load(path)
        overlap = set(part) & set(merged)
        if overlap:
            raise ValueError(
                f'Duplicate example ids across shards for {filename}: '
                f'{list(overlap)[:5]}...')
        merged.update(part)
        total += len(part)
        logging.info('Merged %d examples from %s (%s).', len(part), shard_dir, filename)
    logging.info('Total merged %s examples: %d', filename, total)
    return merged


def merge_p_false(shard_dirs):
    """Concatenate precomputed p_false / p_false_fixed in the same shard order.

    Alignment holds because each shard appends p_trues in the same loop order in
    which it inserts into its validation_generations dict, and we merge both in
    identical shard order.
    """
    p_false, p_false_fixed = [], []
    have_any = False
    for shard_dir in shard_dirs:
        path = _find_pkl(shard_dir, 'uncertainty_measures.pkl')
        if path is None:
            continue
        um = _load(path).get('uncertainty_measures', {})
        if 'p_false' in um:
            have_any = True
            p_false.extend(um['p_false'])
            p_false_fixed.extend(um.get('p_false_fixed', []))
    if not have_any:
        return {}
    return {'p_false': p_false, 'p_false_fixed': p_false_fixed}


def cli():
    parser = utils.get_parser(stages=['compute'])
    parser.add_argument(
        '--shards_parent', type=str, required=True,
        help='Parent directory containing shard subdirectories (shard_*).')
    parser.add_argument(
        '--shard_glob', type=str, default='shard_*',
        help='Glob (relative to --shards_parent) matching each shard directory.')
    args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError(f'Unknown args: {unknown}')

    # Force local, offline-friendly behaviour for the merged compute pass.
    args.assign_new_wandb_id = False
    # p_ik needs training-set embeddings, which the data-parallel flow does not
    # generate (shards run --no-get_training_set_generations).
    if args.compute_p_ik or args.compute_p_ik_answerable:
        logging.info('Disabling p_ik: no training generations in shard flow.')
        args.compute_p_ik = False
        args.compute_p_ik_answerable = False

    shard_dirs = sorted(glob.glob(os.path.join(args.shards_parent, args.shard_glob)))
    if not shard_dirs:
        raise ValueError(
            f'No shard dirs matched {args.shard_glob} under {args.shards_parent}.')
    logging.info('Found %d shard dirs: %s', len(shard_dirs), shard_dirs)

    user = os.environ['USER']
    scratch_dir = os.getenv('SCRATCH_DIR') or args.out_dir
    wandb.init(
        entity=args.entity,
        project='semantic_uncertainty' if not args.debug else 'semantic_uncertainty_debug',
        dir=f'{scratch_dir}/{user}/uncertainty',
        config=args,
        notes=f'merged shards from {args.shards_parent}',
    )
    logging.info('Merged run id: %s dir: %s', wandb.run.id, wandb.run.dir)

    val = merge_generations(shard_dirs, 'validation_generations.pkl')
    with open(f'{wandb.run.dir}/validation_generations.pkl', 'wb') as f:
        pickle.dump(val, f)

    train = merge_generations(shard_dirs, 'train_generations.pkl')
    if train:
        with open(f'{wandb.run.dir}/train_generations.pkl', 'wb') as f:
            pickle.dump(train, f)

    measures = merge_p_false(shard_dirs)
    result_dict = {'uncertainty_measures': measures} if measures else {}
    with open(f'{wandb.run.dir}/uncertainty_measures.pkl', 'wb') as f:
        pickle.dump(result_dict, f)

    # eval/train run ids only matter when assign_new_wandb_id=True (API path);
    # set them to the active run for logging clarity.
    args.eval_wandb_runid = wandb.run.id
    args.train_wandb_runid = wandb.run.id

    logging.info(50 * '#')
    logging.info('STARTING merged `compute_uncertainty_measures`!')
    main_compute(args)
    logging.info('FINISHED merged `compute_uncertainty_measures`!')


if __name__ == '__main__':
    cli()
