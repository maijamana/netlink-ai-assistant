import json
from pathlib import Path
from statistics import mean

from src.retrieval import Retriever


BASE_DIR = Path(__file__).resolve().parent.parent
GOLDEN_PATH = BASE_DIR / "data" / "golden_dataset.json"
RESULTS_PATH = (
    BASE_DIR
    / "results"
    / "retrieval"
    / "evaluation.json"
)


def load_golden_dataset():
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def reciprocal_rank(retrieved_ids, expected_sources):
    """
    Return the reciprocal rank of the first relevant document.

    Example:
    expected = ["KB-04"]
    retrieved = ["KB-11", "KB-09", "KB-04"]

    The first relevant document is at rank 3, so RR = 1 / 3.
    """
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in expected_sources:
            return 1 / rank

    return 0.0


def source_recall_at_k(retrieved_ids, expected_sources, k):
    retrieved_top_k = set(retrieved_ids[:k])
    expected = set(expected_sources)

    if not expected:
        return 0.0

    found = expected.intersection(retrieved_top_k)

    return len(found) / len(expected)


def evaluate_retrieval(retriever, dataset, k=3):
    results = []

    for item in dataset:
        # Out-of-KB questions are evaluated separately later.
        if not item["answerable"]:
            continue

        retrieved = retriever.search(
            item["question"],
            k=k,
        )

        retrieved_ids = [result["id"] for result in retrieved]
        expected_sources = item["expected_sources"]

        hit_at_1 = int(
            any(
                source in retrieved_ids[:1]
                for source in expected_sources
            )
        )

        hit_at_3 = int(
            any(
                source in retrieved_ids[:3]
                for source in expected_sources
            )
        )

        rr = reciprocal_rank(
            retrieved_ids,
            expected_sources,
        )

        source_recall_3 = source_recall_at_k(
            retrieved_ids,
            expected_sources,
            k=3,
        )

        result = {
            "id": item["id"],
            "question": item["question"],
            "category": item["category"],
            "expected_sources": expected_sources,
            "retrieved_ids": retrieved_ids,
            "hit_at_1": hit_at_1,
            "hit_at_3": hit_at_3,
            "reciprocal_rank": rr,
            "source_recall_3": source_recall_3,
        }

        results.append(result)

    return results


def compute_summary(results):
    return {
        "evaluated_examples": len(results),
        "hit_at_1": mean(
            result["hit_at_1"]
            for result in results
        ),
        "hit_at_3": mean(
            result["hit_at_3"]
            for result in results
        ),
        "mrr": mean(
            result["reciprocal_rank"]
            for result in results
        ),
        "source_recall_at_3": mean(
            result["source_recall_3"]
            for result in results
        ),
    }


def print_summary(results, summary):
    print("\n=== RETRIEVAL SUMMARY ===")
    print(
        f"Evaluated examples: "
        f"{summary['evaluated_examples']}"
    )
    print(f"Hit@1: {summary['hit_at_1']:.3f}")
    print(f"Hit@3: {summary['hit_at_3']:.3f}")
    print(f"MRR:   {summary['mrr']:.3f}")
    print(
        f"Source Recall@3: "
        f"{summary['source_recall_at_3']:.3f}"
    )

    print("\n=== FAILED HIT@1 CASES ===")

    failed = [
        result
        for result in results
        if result["hit_at_1"] == 0
    ]

    if not failed:
        print("None")
        return

    for result in failed:
        print(f"\n{result['id']}: {result['question']}")
        print(
            "Expected:",
            result["expected_sources"],
        )
        print(
            "Retrieved:",
            result["retrieved_ids"],
        )


def save_results(results, summary):
    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "summary": summary,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def main():
    dataset = load_golden_dataset()

    print("Loading retriever...")
    retriever = Retriever()

    print("Running retrieval evaluation...")

    results = evaluate_retrieval(
        retriever,
        dataset,
        k=3,
    )
    summary = compute_summary(results)

    save_results(results, summary)
    print_summary(results, summary)

    print(
        f"\nDetailed results saved to: "
        f"{RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()
