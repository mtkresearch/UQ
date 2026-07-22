"""Lightweight omegaconf-backed config loading.

Kept free of heavy imports (torch/transformers/wandb) so the standalone
research scripts under `sep/transfer` and `sep/moe` can layer YAML configs onto
their own argparse parsers without dragging in the model stack.

Two entry points:
- `load_config(paths)` merges base.yaml with overlay YAMLs into a plain dict;
  used by `utils.get_parser` to source argparse defaults.
- `apply_yaml_config(parser)` adds a repeatable `--config` flag to an existing
  parser and returns parsed args with YAML values applied underneath the CLI.

Precedence everywhere: CLI flag > YAML overlay(s) > base.yaml > code default.
"""
import os
import sys
import argparse

from omegaconf import OmegaConf


def _repo_config_dir():
    """Locate the configs/ dir at the repo root (override with SEP_CONFIG_DIR)."""
    here = os.path.abspath(__file__)          # .../src/sep/uncertainty/utils/config.py
    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(here)))))
    return os.getenv('SEP_CONFIG_DIR', os.path.join(repo, 'configs'))


def _resolve_config_path(path):
    """Resolve a config path to an existing file.

    `--config` takes a real YAML path (e.g. `configs/model/qwen3-8b.yaml`),
    absolute or relative to the current working directory.
    """
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(
        f'Config not found: {path!r}. Pass a YAML path such as '
        f'configs/model/qwen3-8b.yaml (relative to the repo root or absolute).')


def load_config(config_paths=None):
    """Merge base.yaml with optional overlay YAMLs into a plain dict.

    Later paths win. Each entry in `config_paths` is a YAML file path. Returns
    {} if base.yaml is absent so hardcoded argparse defaults still apply.
    """
    cfg_dir = _repo_config_dir()
    base = os.path.join(cfg_dir, 'base.yaml')
    merged = OmegaConf.load(base) if os.path.exists(base) else OmegaConf.create({})
    for path in (config_paths or []):
        merged = OmegaConf.merge(merged, OmegaConf.load(_resolve_config_path(path)))
    return OmegaConf.to_container(merged, resolve=True)


def sniff_config_paths(argv=None):
    """Pull --config values out of argv before the main parser is built."""
    if argv is None:
        argv = sys.argv[1:]
    mini = argparse.ArgumentParser(add_help=False)
    mini.add_argument('--config', action='append', default=[])
    known, _ = mini.parse_known_args(argv)
    return known.config


def add_config_arg(parser):
    """Register the repeatable `--config` flag on an existing parser."""
    parser.add_argument(
        '--config', action='append', default=[], metavar='PATH',
        help='YAML config path(s) layered on top of base.yaml (later wins), '
             'e.g. configs/model/qwen3-8b.yaml. Repeatable.')
    return parser


def apply_yaml_config(parser, argv=None):
    """Layer YAML config values under an existing argparse parser's CLI.

    Adds `--config`, loads the merged YAML, and pushes any keys that match the
    parser's known dest names into the parser defaults (so the CLI still wins
    and a YAML value can satisfy an otherwise-required argument). Returns the
    parsed `Namespace`. Unknown args raise, matching the scripts' prior
    `parse_args()` behaviour.
    """
    if argv is None:
        argv = sys.argv[1:]
    add_config_arg(parser)
    cfg = load_config(sniff_config_paths(argv))

    known_dests = {a.dest for a in parser._actions}
    required_actions = [a for a in parser._actions if getattr(a, 'required', False)]
    overrides = {k: v for k, v in cfg.items() if k in known_dests}
    if overrides:
        parser.set_defaults(**overrides)
        # A YAML-supplied value should satisfy a required arg.
        for action in required_actions:
            if action.dest in overrides:
                action.required = False
    return parser.parse_args(argv)
