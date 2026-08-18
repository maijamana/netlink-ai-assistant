import argparse
import json
from pathlib import Path
from statistics import mean

BASE_DIR = Path(__file__).resolve().parent.parent
GOLDEN_PATH = BASE_DIR / "data" / "golden_dataset.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate generation and source attribution."
    )
    parser.add_argument(
        "--version",
        choices=("v1", "v2"),
        required=True,
        help="Generation prompt version to evaluate.",
    )
    return parser.parse_args()


def load_golden_dataset():
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def source_recall(
    generated_sources,
    expected_sources,
):
    generated = set(generated_sources)
    expected = set(expected_sources)

    if not expected:
        return 1.0 if not generated else 0.0

    return len(generated.intersection(expected)) / len(expected)


def source_precision(
    generated_sources,
    expected_sources,
):
    generated = set(generated_sources)
    expected = set(expected_sources)

    if not generated:
        return 1.0 if not expected else 0.0

    return len(generated.intersection(expected)) / len(generated)


def evaluate(version):
    from src.rag import RAGPipeline

    dataset = load_golden_dataset()

    print("Loading RAG pipeline...")
    rag = RAGPipeline(
        retrieval_k=3,
        prompt_version=version,
    )

    results = []

    for index, item in enumerate(dataset, start=1):
        print(
            f"Evaluating {index}/{len(dataset)}: "
            f"{item['id']}"
        )

        generated = rag.answer(item["question"])

        answerability_correct = (
            generated.answerable
            == item["answerable"]
        )

        result = {
            "id": item["id"],
            "question": item["question"],
            "category": item["category"],
            "expected_answerable": item["answerable"],
            "generated_answerable": generated.answerable,
            "answerability_correct": answerability_correct,
            "expected_answer": item["expected_answer"],
            "generated_answer": generated.answer,
            "expected_sources": item["expected_sources"],
            "generated_sources": generated.sources,
            "source_recall": source_recall(
                generated.sources,
                item["expected_sources"],
            ),
            "source_precision": source_precision(
                generated.sources,
                item["expected_sources"],
            ),
        }

        results.append(result)

    return results


def print_summary(results):
    answerability_accuracy = mean(
        r["answerability_correct"]
        for r in results
    )

    avg_source_recall = mean(
        r["source_recall"]
        for r in results
    )

    avg_source_precision = mean(
        r["source_precision"]
        for r in results
    )

    print("\n=== GENERATION / ATTRIBUTION SUMMARY ===")
    print(
        f"Answerability Accuracy: "
        f"{answerability_accuracy:.3f}"
    )
    print(
        f"Citation Recall:        "
        f"{avg_source_recall:.3f}"
    )
    print(
        f"Citation Precision:     "
        f"{avg_source_precision:.3f}"
    )

    print("\n=== FAILED ANSWERABILITY CASES ===")

    failed = [
        r for r in results
        if not r["answerability_correct"]
    ]

    if not failed:
        print("None")
    else:
        for result in failed:
            print(
                f"\n{result['id']}: "
                f"{result['question']}"
            )
            print(
                "Expected:",
                result["expected_answerable"],
            )
            print(
                "Generated:",
                result["generated_answerable"],
            )


def save_results(results, results_path):
    results_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
        )


def main():
    args = parse_args()
    results_path = (
        BASE_DIR
        / "results"
        / "generation"
        / args.version
        / "evaluation.json"
    )
    results = evaluate(args.version)

    save_results(results, results_path)
    print_summary(results)

    print(
        f"\nDetailed results saved to: "
        f"{results_path}"
    )


if __name__ == "__main__":
    main()
