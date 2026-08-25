from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .agent_qualification import AgentQualifier, write_agent_qualification
from .agent_runner import AgentBenchmarkRunner, AgentRunResult
from .live import LocalLiveServer
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


def _print_agent_result(result: AgentRunResult) -> None:
    print("EXO BENCH AGENT")
    print(f"Robot        {result.robot}")
    print(f"Controller   {result.controller}")
    print(f"Agent        {result.agent}")
    print(f"Course       {result.course}")
    print(f"Scenario     seed {result.seed}")
    print(f"Result       {'PASS' if result.passed else 'FAIL'} ({result.reason})")
    print(f"Sim time     {result.simulated_completion_time_s:.2f} s")
    print(f"Path         {result.path_length_m:.2f} m")
    print(f"Position err {result.final_position_error_m:.3f} m")
    print(f"Heading err  {result.final_heading_error_rad:.3f} rad")
    print(f"Final speed  {result.final_speed_mps:.3f} m/s")
    if result.final_window_average_speed_mps is not None:
        print(f"Stop window  {result.final_window_average_speed_mps:.3f} m/s average")
    print(f"Decisions    {result.agent_decision_count}")
    print(f"Agent latency {result.agent_wall_latency_ms['mean']:.3f} ms mean (recorded, not scored)")
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


def _agent_run(args: argparse.Namespace) -> int:
    runner = AgentBenchmarkRunner.from_aliases(
        args.robot,
        args.controller,
        args.course,
        args.reset_profile,
        agent_id=args.agent,
    )
    output = Path(args.record).resolve() if args.record else None
    if args.live_port:
        with LocalLiveServer(args.live_port) as live:
            print(f"EXO live visualization: {live.url}", file=sys.stderr, flush=True)
            result = runner.run(seed=args.seed, record_path=output, live_sink=live.bridge)
            if args.live_hold_seconds > 0:
                time.sleep(args.live_hold_seconds)
    else:
        result = runner.run(seed=args.seed, record_path=output)
    if args.verify_determinism:
        repeated = runner.run(seed=args.seed)
        if repeated.determinism_digest != result.determinism_digest:
            print("Agent determinism verification failed", file=sys.stderr)
            return 2
    if args.json:
        print(json.dumps(result.to_dict(), allow_nan=False, indent=2, sort_keys=True))
    else:
        _print_agent_result(result)
    return 0 if result.passed else 1


def _agent_qualify(args: argparse.Namespace) -> int:
    evidence_dir = Path(args.evidence_dir).resolve() if args.evidence_dir else None
    report = AgentQualifier().qualify(evidence_dir=evidence_dir)
    if args.output:
        write_agent_qualification(report, Path(args.output).resolve())
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 1


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

    agent = commands.add_parser("agent", help="Run or qualify the versioned EXO Agent division")
    agent_commands = agent.add_subparsers(dest="agent_command", required=True)

    agent_run = agent_commands.add_parser("run", help="Run AGENT-001 through the public Agent contract")
    agent_run.add_argument("--course", default="agent-navigation")
    agent_run.add_argument("--robot", default="h1")
    agent_run.add_argument("--controller", default="baseline")
    agent_run.add_argument("--agent", default="exo.agent.scripted-baseline.v0.1")
    agent_run.add_argument("--seed", type=int, default=0)
    agent_run.add_argument("--reset-profile", default="basic")
    agent_run.add_argument("--record")
    agent_run.add_argument("--verify-determinism", action="store_true")
    agent_run.add_argument("--live-port", type=int)
    agent_run.add_argument("--live-hold-seconds", type=float, default=10.0)
    agent_run.add_argument("--json", action="store_true")
    agent_run.set_defaults(function=_agent_run)

    agent_qualify = agent_commands.add_parser("qualify", help="Run EXO Agent v0.1 qualification")
    agent_qualify.add_argument("--output")
    agent_qualify.add_argument("--evidence-dir")
    agent_qualify.set_defaults(function=_agent_qualify)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.function(args))


if __name__ == "__main__":
    main()
