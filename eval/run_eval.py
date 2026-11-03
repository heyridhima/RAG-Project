import json
import sys
from pathlib import Path
import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # eval/ -> simple-rag/
QA_PAIRS_PATH = PROJECT_ROOT / "eval" / "qa_pairs.json"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retriever import retrieve
from reranker import rerank


def load_eval_set(path: str) -> dict:
    """Load qa_pairs.json and return the full dict (incl. _meta)."""
    with open(path, "r") as f:
        return json.load(f)


def compute_metrics_for_question(
    retrieved: list[dict],
    ground_truth: dict,
    k: int,
) -> dict:
    """Pure function: takes retrieved chunks + ground truth, returns
    {precision_at_k, recall_at_k, reciprocal_rank}. Does not call the retriever
    (so it is unit-testable on hand-crafted inputs).
    """
    if k == 0:
        raise ValueError("k must be > 0")

    relevant_pages_set = {
        (p["source"], p["page"]) for p in ground_truth["relevant_pages"]
    }

    # precision@k: fraction of top-k chunks whose (source, page) is in GT
    relevant_count = sum(
        1 for c in retrieved[:k] if (c["source"], c["page"]) in relevant_pages_set
    )
    precision_at_k = relevant_count / k

    # recall@k: how many UNIQUE relevant pages we covered in top-k
    unique_relevant_pages = {
        (c["source"], c["page"])
        for c in retrieved[:k]
        if (c["source"], c["page"]) in relevant_pages_set
    }
    recall_at_k = (
        len(unique_relevant_pages) / len(relevant_pages_set)
        if relevant_pages_set
        else 0.0
    )

    # reciprocal rank: 1 / rank of the first relevant chunk (1-indexed); 0 if none
    rank_of_first_relevant = None
    for i, chunk in enumerate(retrieved[:k]):
        if (chunk["source"], chunk["page"]) in relevant_pages_set:
            rank_of_first_relevant = i + 1
            break
    reciprocal_rank = (
        1 / rank_of_first_relevant if rank_of_first_relevant is not None else 0.0
    )

    return {
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "reciprocal_rank": reciprocal_rank,
    }


def evaluate_question(
    question: dict,
    k: int = 5,
    use_reranker: bool = False,
    retrieve_n: int = 20,
) -> dict:
    """Full evaluation of a single question.

    - in_corpus=True  -> retrieve(k, max_distance=2.0) so nothing is filtered;
                         compute precision@k / recall@k / RR. When
                         use_reranker=True, retrieve `retrieve_n` candidates
                         first and rerank them with a cross-encoder down to k.
    - in_corpus=False -> retrieve(k, max_distance=0.7), the production
                         threshold; success = retriever returned an empty
                         list (refusal). Reranker is intentionally NOT used
                         here — we are testing the retriever-level threshold.

    Always returns question metadata so aggregation can group by type.
    """
    base = {
        "id": question["id"],
        "question": question["question"],
        "type": question["type"],
        "in_corpus": question["in_corpus"],
    }

    if question["in_corpus"]:
        if use_reranker:
            initial = retrieve(
                question["question"], k=retrieve_n, max_distance=2.0
            )
            answer = rerank(question["question"], initial, top_k=k)
        else:
            answer = retrieve(question["question"], k=k, max_distance=2.0)
        metrics = compute_metrics_for_question(answer, question["ground_truth"], k)
        return {**base, "metrics": metrics, "raw_retrieved": answer}

    # trap question: verify the production threshold filters everything out.
    # The reranker is bypassed here on purpose — refusal accuracy measures
    # the retriever-level distance threshold, not the reranking layer.
    answer = retrieve(question["question"], k=k, max_distance=0.7)
    refusal_correct = (answer == [])
    return {**base, "refusal_correct": refusal_correct, "raw_retrieved": answer}


def aggregate_results(per_question: list[dict]) -> dict:
    """Compute overall, per-type and refusal aggregates from per-question results."""
    in_corpus = [r for r in per_question if r["in_corpus"]]
    traps = [r for r in per_question if not r["in_corpus"]]

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    # overall: averages over all in-corpus questions
    overall = {
        "n": len(in_corpus),
        "precision_at_k": mean([r["metrics"]["precision_at_k"] for r in in_corpus]),
        "recall_at_k": mean([r["metrics"]["recall_at_k"] for r in in_corpus]),
        "mrr": mean([r["metrics"]["reciprocal_rank"] for r in in_corpus]),
    }

    # per-type breakdown: definitional, factual, comparative, multi_hop, procedural
    by_type = {}
    for t in sorted({r["type"] for r in in_corpus}):
        bucket = [r for r in in_corpus if r["type"] == t]
        by_type[t] = {
            "n": len(bucket),
            "precision_at_k": mean([r["metrics"]["precision_at_k"] for r in bucket]),
            "recall_at_k": mean([r["metrics"]["recall_at_k"] for r in bucket]),
            "mrr": mean([r["metrics"]["reciprocal_rank"] for r in bucket]),
        }

    # refusal: fraction of trap questions correctly returning an empty list
    refusal = {
        "n": len(traps),
        "correct": sum(1 for r in traps if r["refusal_correct"]),
        "accuracy": mean([1.0 if r["refusal_correct"] else 0.0 for r in traps]),
    }

    return {"overall": overall, "by_type": by_type, "refusal": refusal}


def print_results(per_question: list[dict], aggregate: dict, k: int) -> None:
    """Pretty-print results using rich tables."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # 1) Per-question metrics for in-corpus questions
    per_q_table = Table(title=f"Per-question metrics (k={k})")
    per_q_table.add_column("ID")
    per_q_table.add_column("Type")
    per_q_table.add_column(f"P@{k}", justify="right")
    per_q_table.add_column(f"R@{k}", justify="right")
    per_q_table.add_column("RR", justify="right")
    per_q_table.add_column("Question", overflow="fold", max_width=60)

    for r in per_question:
        if not r["in_corpus"]:
            continue
        m = r["metrics"]
        per_q_table.add_row(
            r["id"],
            r["type"],
            f"{m['precision_at_k']:.3f}",
            f"{m['recall_at_k']:.3f}",
            f"{m['reciprocal_rank']:.3f}",
            r["question"],
        )
    console.print(per_q_table)

    # 2) Trap questions: did the system correctly refuse?
    trap_table = Table(title=f"Trap questions (n={aggregate['refusal']['n']})")
    trap_table.add_column("ID")
    trap_table.add_column("Question", overflow="fold", max_width=60)
    trap_table.add_column("Refused?", justify="center")

    for r in per_question:
        if r["in_corpus"]:
            continue
        mark = "[green]yes[/green]" if r["refusal_correct"] else "[red]no[/red]"
        trap_table.add_row(r["id"], r["question"], mark)
    console.print(trap_table)

    # 3) Overall aggregates
    overall = aggregate["overall"]
    overall_table = Table(title=f"Overall (n={overall['n']} in-corpus questions)")
    overall_table.add_column("Metric")
    overall_table.add_column("Value", justify="right")
    overall_table.add_row(f"Precision@{k}", f"{overall['precision_at_k']:.3f}")
    overall_table.add_row(f"Recall@{k}", f"{overall['recall_at_k']:.3f}")
    overall_table.add_row("MRR", f"{overall['mrr']:.3f}")
    console.print(overall_table)

    # 4) Per-type breakdown
    by_type_table = Table(title="By question type")
    by_type_table.add_column("Type")
    by_type_table.add_column("n", justify="right")
    by_type_table.add_column(f"P@{k}", justify="right")
    by_type_table.add_column(f"R@{k}", justify="right")
    by_type_table.add_column("MRR", justify="right")
    for t, vals in aggregate["by_type"].items():
        by_type_table.add_row(
            t,
            str(vals["n"]),
            f"{vals['precision_at_k']:.3f}",
            f"{vals['recall_at_k']:.3f}",
            f"{vals['mrr']:.3f}",
        )
    console.print(by_type_table)

    # 5) Refusal accuracy headline
    refusal = aggregate["refusal"]
    console.print(
        f"\n[bold]Refusal accuracy:[/bold] {refusal['accuracy']:.3f} "
        f"({refusal['correct']}/{refusal['n']} traps correctly refused)"
    )


def main(
    run_name: str = "baseline_fixed_500_100",
    k: int = 5,
    use_reranker: bool = False,
    retrieve_n: int = 20,
) -> None:
    eval_set = load_eval_set(str(QA_PAIRS_PATH))
    per_question = [
        evaluate_question(q, k=k, use_reranker=use_reranker, retrieve_n=retrieve_n)
        for q in eval_set["questions"]
    ]
    aggregate = aggregate_results(per_question)
    print_results(per_question, aggregate, k=k)

    # Save a timestamped copy for full history (gitignored)
    results_dir = PROJECT_ROOT / "eval" / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"{run_name}_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "config": {
                    "k": k,
                    "use_reranker": use_reranker,
                    "retrieve_n": retrieve_n if use_reranker else None,
                },
                "per_question": per_question,
                "aggregate": aggregate,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nResults saved to: {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RAG retrieval evaluation.")
    parser.add_argument(
        "--name",
        default="baseline_fixed_500_100",
        help="Run name used as filename prefix in eval/results/ (e.g. recursive_500_100).",
    )
    parser.add_argument("--k", type=int, default=5, help="Top-K for retrieval metrics.")
    parser.add_argument(
        "--reranker",
        action="store_true",
        help="Enable cross-encoder reranking (retrieve --retrieve-n, then rerank to top-K).",
    )
    parser.add_argument(
        "--retrieve-n",
        type=int,
        default=20,
        help="How many candidates to retrieve before reranking. Ignored without --reranker.",
    )
    args = parser.parse_args()

    main(
        run_name=args.name,
        k=args.k,
        use_reranker=args.reranker,
        retrieve_n=args.retrieve_n,
    )
