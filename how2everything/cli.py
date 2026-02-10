from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="h2e", description="how2everything CLI.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # New, namespaced interface:
    #
    #   h2e mine run --config ...
    #   h2e mine validate --config ...
    #   h2e mine status --out ...
    #
    # We keep legacy top-level aliases (run/validate/status) for compatibility.
    mine = sub.add_parser("mine", help="mine: rewrite documents into procedures.")
    mine_sub = mine.add_subparsers(dest="mine_cmd", required=True)

    mine_run = mine_sub.add_parser("run", help="Run the mine pipeline.")
    mine_run.add_argument("--config", required=True, help="Path to a mine YAML config.")

    mine_validate = mine_sub.add_parser("validate", help="Validate mine config and environment.")
    mine_validate.add_argument("--config", required=True, help="Path to a mine YAML config.")

    mine_status = mine_sub.add_parser("status", help="Show pipeline progress for an out_root.")
    mine_status.add_argument("--out", required=True, help="Output root directory.")

    bench = sub.add_parser("bench", help="bench: generate + judge procedures.")
    bench_sub = bench.add_subparsers(dest="bench_cmd", required=True)

    bench_validate = bench_sub.add_parser("validate", help="Validate bench config and environment.")
    bench_validate.add_argument("--config", required=True, help="Path to a bench YAML config.")

    bench_gen = bench_sub.add_parser("gen", help="Run bench generation only.")
    bench_gen.add_argument("--config", required=True, help="Path to a bench YAML config.")

    bench_judge = bench_sub.add_parser("judge", help="Run bench judging only.")
    bench_judge.add_argument("--config", required=True, help="Path to a bench YAML config.")

    bench_run = bench_sub.add_parser("run", help="Run bench end-to-end (gen -> judge -> aggregate).")
    bench_run.add_argument("--config", required=True, help="Path to a bench YAML config.")

    bench_leader = bench_sub.add_parser(
        "leaderboard",
        help="Aggregate results across many generation runs for one judge.",
    )
    bench_leader.add_argument(
        "--generations-root",
        required=True,
        help="Directory containing many generation run folders (each with generations.jsonl).",
    )
    bench_leader.add_argument(
        "--judge",
        default=None,
        help="Optional filter for a single judge dir (exact, suffix _<hash>, or substring).",
    )
    bench_leader.add_argument(
        "--output",
        "-o",
        default=None,
        help="Optional output CSV path (default: stdout).",
    )
    bench_leader.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print a terminal table (writes CSV only if --output is set).",
    )

    data = sub.add_parser("data", help="Create training data artifacts. (TODO)")
    data.add_subparsers(dest="data_cmd", required=False)

    # Legacy aliases (pre-namespace).
    run = sub.add_parser("run", help="[Deprecated] Alias for `h2e mine run`.")
    run.add_argument("--config", required=True, help="Path to a mine YAML config.")

    validate = sub.add_parser("validate", help="[Deprecated] Alias for `h2e mine validate`.")
    validate.add_argument("--config", required=True, help="Path to a mine YAML config.")

    status = sub.add_parser("status", help="[Deprecated] Alias for `h2e mine status`.")
    status.add_argument("--out", required=True, help="Output root directory.")

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Defer imports so `h2e --help` works even if optional deps missing.
    if args.cmd == "bench":
        if args.bench_cmd == "validate":
            from how2everything.bench.runner import validate_config_and_env

            validate_config_and_env(args.config)
            return 0
        if args.bench_cmd == "gen":
            from how2everything.bench.config import load_suite_config
            from how2everything.bench.pipeline import gen_suite

            suite = load_suite_config(args.config)
            gen_suite(suite)
            return 0
        if args.bench_cmd == "judge":
            from how2everything.bench.aggregate import write_aggregates
            from how2everything.bench.config import load_suite_config, suite_to_judge_config
            from how2everything.bench.judge import run_judge
            from how2everything.bench.pipeline import make_client

            suite = load_suite_config(args.config)
            cfg = suite_to_judge_config(suite)
            judge_path = run_judge(
                cfg,
                make_client(cfg, "judge"),
                generations_path=cfg.generations_path(),
                out_path=cfg.judgments_path(),
            )
            summary = write_aggregates(out_root=judge_path.parent, judgments_path=judge_path)
            how2score = float(summary.get("how2score_percent", 0.0)) if isinstance(summary, dict) else 0.0
            n_examples = int(summary.get("n_examples", 0)) if isinstance(summary, dict) else 0
            print(f"\n  how2score: {how2score:.1f}%  (n={n_examples})", flush=True)
            return 0
        if args.bench_cmd == "run":
            from how2everything.bench.runner import run_from_config

            run_from_config(args.config)
            return 0
        if args.bench_cmd == "leaderboard":
            from pathlib import Path

            from how2everything.bench.leaderboard import print_leaderboard_table, write_leaderboard_csv

            out_csv = Path(args.output) if args.output else None
            if args.pretty:
                print_leaderboard_table(
                    generations_root=Path(args.generations_root),
                    judge=args.judge,
                )
                if out_csv is not None:
                    write_leaderboard_csv(
                        generations_root=Path(args.generations_root),
                        judge=args.judge,
                        out_csv=out_csv,
                    )
            else:
                write_leaderboard_csv(
                    generations_root=Path(args.generations_root),
                    judge=args.judge,
                    out_csv=out_csv,
                )
            return 0
        parser.error(f"Unhandled bench cmd: {args.bench_cmd}")
        return 2

    if args.cmd == "data":
        print("Training-data subcommands are not implemented yet. Use `h2e mine ...` for now.", file=sys.stderr)
        return 2

    if args.cmd == "mine":
        if args.mine_cmd == "validate":
            from how2everything.mine.runner import validate_config_and_env

            validate_config_and_env(args.config)
            return 0

        if args.mine_cmd == "run":
            from how2everything.mine.runner import run_from_config

            run_from_config(args.config)
            return 0

        if args.mine_cmd == "status":
            from how2everything.mine.status import print_status

            print_status(args.out)
            return 0

        parser.error(f"Unhandled mine cmd: {args.mine_cmd}")
        return 2

    # Legacy aliases.
    if args.cmd == "validate":
        from how2everything.mine.runner import validate_config_and_env

        print("Note: `h2e validate` is deprecated; use `h2e mine validate`.", file=sys.stderr)
        validate_config_and_env(args.config)
        return 0

    if args.cmd == "run":
        from how2everything.mine.runner import run_from_config

        print("Note: `h2e run` is deprecated; use `h2e mine run`.", file=sys.stderr)
        run_from_config(args.config)
        return 0

    if args.cmd == "status":
        from how2everything.mine.status import print_status

        print("Note: `h2e status` is deprecated; use `h2e mine status`.", file=sys.stderr)
        print_status(args.out)
        return 0

    parser.error(f"Unhandled cmd: {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
