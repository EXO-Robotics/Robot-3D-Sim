from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .qualification import ChassisQualifier, write_qualification
from .recorder import inspect_replay
from .registry import ArtifactRegistry
from .runner import BenchmarkRunner, RunResult


def _print_result(result: RunResult) -> None:
    print("EXO BENCH")
    print(f"Robot        {result.robot}")
    print(f"Controller   {result.controller}")
    print(f"Course       {result.course}")
    print(f"Seed         {result.seed}")
    print(f"Result       {'PASS' if result.passed else 'FAIL'} ({result.reason})")
    print(f"Distance     {result.progress_meters:.2f} m")
    print(f"Time         {result.time_seconds:.2f} s")
    print(f"Falls        {result.falls}")
    print(f"Energy       {result.energy_joules:.2f} J")
    print(f"Peak torque  {result.peak_torque_nm:.2f} Nm")
    if result.stop_average_speed_mps is not None:
        print(f"Stop speed   {result.stop_average_speed_mps:.3f} m/s average")
    print(f"Digest       {result.determinism_digest}")
    if result.replay_path:
        print(f"Replay       {result.replay_path}")


def _run(args: argparse.Namespace) -> int:
    runner = BenchmarkRunner.from_aliases(args.robot, args.controller, args.course, args.reset_profile)
    output = Path(args.record).resolve() if args.record else None
    result = runner.run(seed=args.seed, record_path=output)
    if args.verify_determinism:
        repeated = runner.run(seed=args.seed)
        if repeated.determinism_digest != result.determinism_digest:
            print("Determinism verification failed", file=sys.stderr)
            return 2
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        _print_result(result)
    return 0 if result.passed else 1


def _replay(args: argparse.Namespace) -> int:
    summary = inspect_replay(
        Path(args.path).resolve(),
        require_archive_receipt=args.require_archive_receipt,
    )
    if args.json:
        print(
            json.dumps(
                {
                    "manifest": summary.manifest,
                    "metrics": summary.metrics,
                    "verified_files": summary.verified_files,
                    "samples": summary.samples,
                    "archive_sha256": summary.archive_sha256,
                    "archive_receipt_verified": summary.archive_receipt_verified,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print("EXO BENCH REPLAY")
        print(f"File         {summary.path}")
        print(f"Robot        {summary.manifest['robot']['id']}")
        print(f"Controller   {summary.manifest['controller']['id']}")
        print(f"Course       {summary.manifest['course']['id']}")
        print(f"Result       {'PASS' if summary.metrics['passed'] else 'FAIL'}")
        print(f"Verified     {summary.verified_files} content files")
        print(f"Archive SHA  {summary.archive_sha256}")
        print(f"Sidecar      {'VERIFIED' if summary.archive_receipt_verified else 'NOT PRESENT'}")
        print(f"Digest       {summary.manifest['determinism_digest']}")
    return 0


def _qualify(args: argparse.Namespace) -> int:
    evidence_dir = Path(args.evidence_dir).resolve() if args.evidence_dir else None
    report = ChassisQualifier(args.robot, args.controller).qualify(evidence_dir=evidence_dir)
    if args.output:
        write_qualification(report, Path(args.output).resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 1


def _verify(args: argparse.Namespace) -> int:
    artifacts = ArtifactRegistry().resolve(args.robot, args.controller, args.course, args.reset_profile)
    hashes = ArtifactRegistry.verify_hashes(artifacts)
    print(json.dumps(hashes, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="exo-bench")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Run a deterministic benchmark episode")
    run.add_argument("--course", default="basic-walk-stop")
    run.add_argument("--robot", default="h1")
    run.add_argument("--controller", default="baseline")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--reset-profile", default="basic")
    run.add_argument("--record")
    run.add_argument("--verify-determinism", action="store_true")
    run.add_argument("--json", action="store_true")
    run.set_defaults(function=_run)

    replay = commands.add_parser("replay", help="Validate and inspect an .exorun trajectory")
    replay.add_argument("path")
    replay.add_argument("--require-archive-receipt", action="store_true")
    replay.add_argument("--json", action="store_true")
    replay.set_defaults(function=_replay)

    qualify = commands.add_parser("qualify", help="Run CQ-001 through CQ-007")
    qualify.add_argument("--robot", default="h1")
    qualify.add_argument("--controller", default="baseline")
    qualify.add_argument("--output")
    qualify.add_argument("--evidence-dir")
    qualify.set_defaults(function=_qualify)

    verify = commands.add_parser("verify-artifacts", help="Verify pinned artifact hashes")
    verify.add_argument("--robot", default="h1")
    verify.add_argument("--controller", default="baseline")
    verify.add_argument("--course", default="BASIC-001")
    verify.add_argument("--reset-profile", default="basic")
    verify.set_defaults(function=_verify)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.function(args))


if __name__ == "__main__":
    main()
