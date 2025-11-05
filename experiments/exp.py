#!/usr/bin/env python3
"""
Unified CLI for the Speed Dating experiments (flat layout)

Goals:
- Single entry point for all tasks
- Zero behavioral change: delegates to existing scripts
- Clear, consistent flags and sensible defaults

Usage examples (run from test/):
  python experiments/exp.py run-all --dataset "Speed Dating Data.csv"
  python experiments/exp.py simulate --pairs results/personas.json --num-rounds 5
  python experiments/exp.py eval-llm --stage 1 --method both
  python experiments/exp.py consolidate
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent  # .../test
EXP = Path(__file__).parent          # .../test/experiments
PY = sys.executable                  # current python


def run(cmd: list[str]) -> int:
    print("=" * 70)
    print("Running:", " ".join(cmd))
    print("=" * 70)
    return subprocess.call(cmd)


# ---------- Subcommand implementations (thin wrappers) ----------

def cmd_preprocess(args: argparse.Namespace) -> int:
    return run([PY, str(EXP / 'data_preprocessing.py'),
                '--data', args.dataset])


def cmd_generate_personas(args: argparse.Namespace) -> int:
    cmd = [PY, str(EXP / 'persona_generator.py')]
    if args.pairs:
        cmd += ['--pairs', args.pairs]
    if args.encoded_time2:
        cmd += ['--encoded-time2', args.encoded_time2]
    return run(cmd)


def cmd_encode_time2(args: argparse.Namespace) -> int:
    cmd = [PY, str(EXP / 'encode_time2_reflections_with_changes.py'),
           '--personas', args.personas,
           '--output', args.output,
           '--max-concurrent', str(args.max_concurrent)]
    return run(cmd)


def cmd_simulate(args: argparse.Namespace) -> int:
    cmd = [PY, str(EXP / 'speed_dating_simulator.py'),
           '--pairs', args.pairs,
           '--output-dir', args.output_dir,
           '--num-rounds', str(args.num_rounds)]
    if args.sample_size:
        cmd += ['--sample-size', str(args.sample_size)]
    return run(cmd)


def cmd_make_icl(args: argparse.Namespace) -> int:
    cmd = [PY, str(EXP / 'create_icl_examples.py'),
           '--conversations', args.conversations,
           '--personas', args.personas,
           '--output', args.output,
           '--num-examples', str(args.num_examples)]
    return run(cmd)


def cmd_eval_llm(args: argparse.Namespace) -> int:
    cmd = [PY, str(EXP / 'llm_score_evaluator.py'),
           '--conversations', args.conversations,
           '--output-dir', args.output_dir,
           '--stage', str(args.stage),
           '--method', args.method,
           '--max-pair-workers', str(args.max_pair_workers)]
    if args.icl_examples:
        cmd += ['--icl-examples', args.icl_examples]
    if args.participant_model:
        cmd += ['--participant-model', args.participant_model]
    if args.observer_model:
        cmd += ['--observer-model', args.observer_model]
    if args.threshold is not None:
        cmd += ['--threshold', str(args.threshold)]
    if args.report_curves:
        cmd += ['--report-curves']
    return run(cmd)


def cmd_ensemble(args: argparse.Namespace) -> int:
    if args.stage == 0:
        # run both stages if files exist
        rc1 = run([PY, str(EXP / 'ensemble_model.py'),
                   '--llm-results', str(ROOT / 'results' / 'llm_score_evaluation_stage1.json'),
                   '--output', str(ROOT / 'results' / 'ensemble_evaluation_stage1.json')])
        rc2 = run([PY, str(EXP / 'ensemble_model.py'),
                   '--llm-results', str(ROOT / 'results' / 'llm_score_evaluation_stage2.json'),
                   '--output', str(ROOT / 'results' / 'ensemble_evaluation_stage2.json')])
        return rc1 or rc2
    else:
        src = ROOT / 'results' / f'llm_score_evaluation_stage{args.stage}.json'
        out = ROOT / 'results' / f'ensemble_evaluation_stage{args.stage}.json'
        return run([PY, str(EXP / 'ensemble_model.py'),
                    '--llm-results', str(src),
                    '--output', str(out)])


def cmd_baselines(args: argparse.Namespace) -> int:
    return run([PY, str(EXP / 'baseline_models_v2.py'),
                '--personas', args.personas,
                '--output-dir', args.output_dir])


def cmd_consolidate(_: argparse.Namespace) -> int:
    return run([PY, str(EXP / 'generate_consolidated_report.py')])


def cmd_run_all(args: argparse.Namespace) -> int:
    # 1) Preprocess → 2) Personas → 3) Encode Time2 → 4) Simulate → 5) ICL → 6) LLM S1/S2
    # 7) Ensembles both → 8) Baselines → 9) Consolidated
    rc = 0
    results = ROOT / 'results'
    results.mkdir(exist_ok=True)

    if args.dataset:
        rc |= cmd_preprocess(argparse.Namespace(dataset=args.dataset))
    rc |= cmd_generate_personas(argparse.Namespace(pairs=None, encoded_time2=None))
    rc |= cmd_encode_time2(argparse.Namespace(personas=str(results / 'personas.json'),
                                              output=str(results / 'personas.json'),
                                              max_concurrent=args.max_concurrent))
    rc |= cmd_simulate(argparse.Namespace(pairs=str(results / 'personas.json'),
                                          output_dir=str(results),
                                          num_rounds=args.num_rounds,
                                          sample_size=args.sample_size))
    rc |= cmd_make_icl(argparse.Namespace(conversations=str(results / 'conversations.json'),
                                          personas=str(results / 'personas.json'),
                                          output=str(results / 'icl_examples.json'),
                                          num_examples=5))
    rc |= cmd_eval_llm(argparse.Namespace(conversations=str(results / 'conversations.json'),
                                          output_dir=str(results),
                                          stage=1, method='both', max_pair_workers=args.max_pair_workers,
                                          icl_examples=str(results / 'icl_examples.json'),
                                          participant_model=None, observer_model=None,
                                          threshold=None, report_curves=False))
    rc |= cmd_eval_llm(argparse.Namespace(conversations=str(results / 'conversations.json'),
                                          output_dir=str(results),
                                          stage=2, method='both', max_pair_workers=args.max_pair_workers,
                                          icl_examples=str(results / 'icl_examples.json'),
                                          participant_model=None, observer_model=None,
                                          threshold=None, report_curves=False))
    rc |= cmd_ensemble(argparse.Namespace(stage=0))
    rc |= cmd_baselines(argparse.Namespace(personas=str(results / 'personas.json'),
                                           output_dir=str(results)))
    rc |= cmd_consolidate(argparse.Namespace())
    return rc


# ---------- Parser ----------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified CLI for Speed Dating experiments")
    sub = p.add_subparsers(dest='cmd', required=True)

    sp = sub.add_parser('preprocess-data', help='Prepare dataset (CSV → processed pairs)')
    sp.add_argument('--dataset', type=str, default='Speed Dating Data.csv')
    sp.set_defaults(func=cmd_preprocess)

    sp = sub.add_parser('generate-personas', help='Generate personas from processed pairs')
    sp.add_argument('--pairs', type=str, help='Path to processed_pairs.json (optional)')
    sp.add_argument('--encoded-time2', type=str, help='Path to encoded Time2 (optional)')
    sp.set_defaults(func=cmd_generate_personas)

    sp = sub.add_parser('encode-time2', help='Encode Time 2 reflections with changes')
    sp.add_argument('--personas', type=str, default=str(ROOT / 'results' / 'personas.json'))
    sp.add_argument('--output', type=str, default=str(ROOT / 'results' / 'personas.json'))
    sp.add_argument('--max-concurrent', type=int, default=10)
    sp.set_defaults(func=cmd_encode_time2)

    sp = sub.add_parser('simulate', help='Simulate speed dating conversations')
    sp.add_argument('--pairs', type=str, default=str(ROOT / 'results' / 'personas.json'))
    sp.add_argument('--output-dir', type=str, default=str(ROOT / 'results'))
    sp.add_argument('--num-rounds', type=int, default=5)
    sp.add_argument('--sample-size', type=int, help='Optional subset of pairs')
    sp.set_defaults(func=cmd_simulate)

    sp = sub.add_parser('make-icl', help='Create in-context learning examples')
    sp.add_argument('--conversations', type=str, default=str(ROOT / 'results' / 'conversations.json'))
    sp.add_argument('--personas', type=str, default=str(ROOT / 'results' / 'personas.json'))
    sp.add_argument('--output', type=str, default=str(ROOT / 'results' / 'icl_examples.json'))
    sp.add_argument('--num-examples', type=int, default=5)
    sp.set_defaults(func=cmd_make_icl)

    sp = sub.add_parser('eval-llm', help='Evaluate LLM methods (participant/observer)')
    sp.add_argument('--conversations', type=str, default=str(ROOT / 'results' / 'conversations.json'))
    sp.add_argument('--output-dir', type=str, default=str(ROOT / 'results'))
    sp.add_argument('--stage', type=int, choices=[1, 2], default=1)
    sp.add_argument('--method', type=str, choices=['participant', 'observer', 'both'], default='both')
    sp.add_argument('--max-pair-workers', type=int, default=10)
    sp.add_argument('--icl-examples', type=str, default=str(ROOT / 'results' / 'icl_examples.json'))
    sp.add_argument('--participant-model', type=str)
    sp.add_argument('--observer-model', type=str)
    sp.add_argument('--threshold', type=float)
    sp.add_argument('--report-curves', action='store_true')
    sp.set_defaults(func=cmd_eval_llm)

    sp = sub.add_parser('ensemble', help='Train/evaluate ensemble models')
    sp.add_argument('--stage', type=int, choices=[0, 1, 2], default=0, help='0=both')
    sp.set_defaults(func=cmd_ensemble)

    sp = sub.add_parser('baselines', help='Train/evaluate baseline ML models')
    sp.add_argument('--personas', type=str, default=str(ROOT / 'results' / 'personas.json'))
    sp.add_argument('--output-dir', type=str, default=str(ROOT / 'results'))
    sp.set_defaults(func=cmd_baselines)

    sp = sub.add_parser('consolidate', help='Generate consolidated report')
    sp.set_defaults(func=cmd_consolidate)

    sp = sub.add_parser('run-all', help='Run the full pipeline end-to-end')
    sp.add_argument('--dataset', type=str, help='Path to Speed Dating Data.csv (optional)')
    sp.add_argument('--max-concurrent', type=int, default=10)
    sp.add_argument('--num-rounds', type=int, default=5)
    sp.add_argument('--sample-size', type=int)
    sp.add_argument('--max-pair-workers', type=int, default=10)
    sp.set_defaults(func=cmd_run_all)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
