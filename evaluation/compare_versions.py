import csv
import json
from pathlib import Path
from statistics import mean


BASE_DIR = Path(__file__).resolve().parent.parent
RETRIEVAL_RESULTS_PATH = (
    BASE_DIR
    / "results"
    / "retrieval"
    / "evaluation.json"
)
CSV_RESULTS_PATH = (
    BASE_DIR
    / "results"
    / "comparison"
    / "evaluation_results.csv"
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_generation_metrics(
    version,
    retrieval_metrics,
):
    generation_results_path = (
        BASE_DIR
        / "results"
        / "generation"
        / version
        / "evaluation.json"
    )
    judged_results_path = (
        BASE_DIR
        / "results"
        / "generation"
        / version
        / "judged.json"
    )

    generation_results = load_json(
        generation_results_path
    )
    judged_results = load_json(
        judged_results_path
    )

    answerability_accuracy = mean(
        r["answerability_correct"]
        for r in generation_results
    )

    source_recall = mean(
        r["source_recall"]
        for r in generation_results
    )

    source_precision = mean(
        r["source_precision"]
        for r in generation_results
    )

    correctness = mean(
        r["correctness"]
        for r in judged_results
    ) / 2

    faithfulness = mean(
        r["faithfulness"]
        for r in judged_results
    ) / 2

    return {
        "version": version,
        "hit_at_1": retrieval_metrics["hit_at_1"],
        "hit_at_3": retrieval_metrics["hit_at_3"],
        "mrr": retrieval_metrics["mrr"],
        "retrieval_source_recall_at_3": (
            retrieval_metrics["source_recall_at_3"]
        ),
        "answerability_accuracy": answerability_accuracy,
        "citation_recall": source_recall,
        "citation_precision": source_precision,
        "correctness": correctness,
        "faithfulness": faithfulness,
    }


def print_table(results):
    headers = [
        "Version",
        "Hit@1",
        "Hit@3",
        "MRR",
        "Retrieval Src R@3",
        "Answerability",
        "Citation Recall",
        "Citation Precision",
        "Correctness",
        "Faithfulness",
    ]

    print("\n=== FINAL EVALUATION TABLE ===\n")

    print(
        f"{headers[0]:<10}"
        f"{headers[1]:>10}"
        f"{headers[2]:>10}"
        f"{headers[3]:>10}"
        f"{headers[4]:>19}"
        f"{headers[5]:>16}"
        f"{headers[6]:>17}"
        f"{headers[7]:>19}"
        f"{headers[8]:>14}"
        f"{headers[9]:>16}"
    )

    print("-" * 150)

    for r in results:
        print(
            f"{r['version']:<10}"
            f"{r['hit_at_1']:>10.3f}"
            f"{r['hit_at_3']:>10.3f}"
            f"{r['mrr']:>10.3f}"
            f"{r['retrieval_source_recall_at_3']:>19.3f}"
            f"{r['answerability_accuracy']:>16.3f}"
            f"{r['citation_recall']:>17.3f}"
            f"{r['citation_precision']:>19.3f}"
            f"{r['correctness']:>14.3f}"
            f"{r['faithfulness']:>16.3f}"
        )


def print_differences(v1, v2):
    print("\n=== CHANGE v1 -> v2 ===\n")

    metrics = [
        "answerability_accuracy",
        "citation_recall",
        "citation_precision",
        "correctness",
        "faithfulness",
    ]

    for metric in metrics:
        delta = v2[metric] - v1[metric]

        print(
            f"{metric:<25}"
            f"{v1[metric]:.3f} -> "
            f"{v2[metric]:.3f} "
            f"(Δ {delta:+.3f})"
        )


def save_csv(results):
    fieldnames = [
        "version",
        "hit_at_1",
        "hit_at_3",
        "mrr",
        "retrieval_source_recall_at_3",
        "answerability_accuracy",
        "citation_recall",
        "citation_precision",
        "correctness",
        "faithfulness",
    ]

    CSV_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        CSV_RESULTS_PATH,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(results)


def main():
    retrieval_metrics = load_json(
        RETRIEVAL_RESULTS_PATH
    )["summary"]
    v1 = compute_generation_metrics(
        "v1",
        retrieval_metrics,
    )
    v2 = compute_generation_metrics(
        "v2",
        retrieval_metrics,
    )

    results = [v1, v2]

    save_csv(results)
    print_table(results)
    print_differences(v1, v2)

    print(
        f"\nCSV results saved to: "
        f"{CSV_RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()
