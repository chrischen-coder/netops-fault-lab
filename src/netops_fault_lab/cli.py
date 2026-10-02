"""CLI commands avoid contacting network devices or loading model services."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from . import __version__
from .inference import diagnose, fit_noise
from .report import render_html, render_text
from .schema import Case, NoiseModel, read_dataset, read_json


def _write(text: str, output: str | None) -> None:
    if output:
        with Path(output).open('x', encoding='utf-8') as stream:
            stream.write(text + '\n')
    else:
        print(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Diagnose link faults from known path probes.')
    parser.add_argument('--version', action='version', version=__version__)
    commands = parser.add_subparsers(dest='command', required=True)
    diagnosis = commands.add_parser('diagnose', help='Rank faults and recommend a supplied probe')
    diagnosis.add_argument('case')
    diagnosis.add_argument('--model')
    diagnosis.add_argument('--threshold', type=float, default=0.9)
    diagnosis.add_argument('--format', choices=['text', 'json', 'html'], default='text')
    diagnosis.add_argument('--output')
    fit = commands.add_parser('fit', help='Estimate observation noise from labeled JSONL')
    fit.add_argument('dataset')
    fit.add_argument('--output', required=True)
    benchmark = commands.add_parser('benchmark', help='Generate held-out topologies and compare methods; requires [bench]')
    benchmark.add_argument('--output', required=True)
    benchmark.add_argument('--seeds', default='7,17,29')
    benchmark.add_argument('--train-topologies', type=int, default=24)
    benchmark.add_argument('--test-topologies', type=int, default=24)
    benchmark.add_argument('--cases-per-topology', type=int, default=12)
    args = parser.parse_args(argv)
    try:
        if args.command == 'diagnose':
            raw = read_json(args.case)
            case = Case.from_dict(raw)
            model = NoiseModel.from_dict(read_json(args.model)) if args.model else NoiseModel()
            report = diagnose(case, model, decision_threshold=args.threshold)
            report['input_sha256'] = hashlib.sha256(Path(args.case).read_bytes()).hexdigest()
            rendered = (render_html(case, report) if args.format == 'html' else
                        json.dumps(report, indent=2, ensure_ascii=False) if args.format == 'json' else render_text(report))
            _write(rendered, args.output)
        elif args.command == 'fit':
            model = fit_noise(read_dataset(args.dataset))
            model.provenance['training_file_sha256'] = hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest()
            _write(json.dumps(model.to_dict(), indent=2), args.output)
        else:
            from .benchmark import run_benchmark
            seeds = [int(s) for s in args.seeds.split(',')]
            result = run_benchmark(Path(args.output), seeds=seeds, train_topologies=args.train_topologies,
                                   test_topologies=args.test_topologies, cases_per_topology=args.cases_per_topology)
            print(json.dumps(result['summary'], indent=2))
        return 0
    except (ValueError, OSError, UnicodeError, ImportError) as error:
        print(f'error: {error}', file=sys.stderr)
        return 2
