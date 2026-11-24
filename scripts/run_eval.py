"""Command-line evaluation.

    python scripts/run_eval.py data/eval/sample_eval.json --k 5 [--no-llm] [--judge]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402
from docintel.evaluation import Evaluator, load_dataset  # noqa: E402
from docintel.pipeline import Pipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--no-llm", action="store_true", help="use extractive answers")
    ap.add_argument("--judge", action="store_true", help="add LLM-as-judge scores")
    ap.add_argument("--out", type=Path, help="write full JSON results here")
    args = ap.parse_args()

    pipe = Pipeline(Config)
    result = Evaluator(pipe).run(load_dataset(args.dataset), k=args.k, use_llm=not args.no_llm, llm_judge=args.judge)
    s = result["summary"]
    k = s["k"]
    cols = [f"hit@{k}", f"recall@{k}", f"precision@{k}", "mrr", f"ndcg@{k}", "latency_ms"]
    print(f"\n{s['n_questions']} questions, k={k}\n")
    print(f"{'config':<15}" + "".join(f"{c:>14}" for c in cols))
    for name, m in s["retrieval"].items():
        print(f"{name:<15}" + "".join(f"{(m.get(c) or 0):>14.3f}" for c in cols))
    if "answer" in s:
        print(f"\nAnswer quality ({s['answer_config']}):")
        for key, v in s["answer"].items():
            print(f"  {key:<20} {'—' if v is None else f'{v:.3f}'}")
    if args.out:
        args.out.write_text(json.dumps(result, indent=2))
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
