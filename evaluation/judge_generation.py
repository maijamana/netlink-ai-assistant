import argparse
import json
import os
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent.parent
KB_PATH = BASE_DIR / "data" / "knowledge_base.json"
load_dotenv(BASE_DIR / ".env")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Judge saved generation results."
    )
    parser.add_argument(
        "--version",
        choices=("v1", "v2"),
        required=True,
        help="Saved generation version to judge.",
    )
    return parser.parse_args()


class JudgeAssessment(BaseModel):
    correctness: int = Field(
        ge=0,
        le=2,
        description=(
            "Correctness score: 0=incorrect, "
            "1=partially correct, 2=fully correct."
        ),
    )
    faithfulness: int = Field(
        ge=0,
        le=2,
        description=(
            "Faithfulness score: 0=major unsupported claims, "
            "1=minor unsupported claim, 2=fully supported."
        ),
    )
    judge_reason: str = Field(
        description=(
            "Brief explanation of both scores."
        ),
    )


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_context(source_ids, documents_by_id):
    chunks = []

    for source_id in source_ids:
        document = documents_by_id.get(source_id)

        if document is None:
            continue

        chunks.append(
            f"""
SOURCE: {document["id"]}
TITLE: {document["title"]}
CONTENT:
{document["text"]}
""".strip()
        )

    if not chunks:
        return "NO KB SOURCES WERE CITED."

    return "\n\n---\n\n".join(chunks)


def judge_example(
    client,
    model,
    item,
    context,
):
    system_prompt = """
You are evaluating a grounded customer-support answer.

Evaluate two criteria independently.

CORRECTNESS:
Compare the generated answer with the expected answer and determine whether it
correctly answers the user's question.

Score:
0 = incorrect or contradicts the expected answer
1 = partially correct or missing important information
2 = fully correct

FAITHFULNESS:
Determine whether every factual claim in the generated answer is supported by
the provided cited KB context.

Score:
0 = contains major unsupported claims
1 = contains a minor unsupported claim or overstatement
2 = all factual claims are fully supported

The context contains only KB sources cited by the generated answer. Do not use
outside knowledge. Do not transfer information from the expected answer into
the faithfulness assessment.

A generic abstention stating that the available knowledge base information is
insufficient and suggesting contacting an operator does not itself count as an
unsupported factual claim when no KB sources were cited.

Return integer scores and one concise reason explaining both.
""".strip()

    user_prompt = f"""
QUESTION:
{item["question"]}

EXPECTED ANSWER:
{item["expected_answer"]}

CITED KB CONTEXT:
{context}

GENERATED ANSWER:
{item["generated_answer"]}
""".strip()

    response = client.responses.parse(
        model=model,
        input=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        text_format=JudgeAssessment,
    )

    return response.output_parsed


def main():
    args = parse_args()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Add it to the .env file."
        )

    generation_results_path = (
        BASE_DIR
        / "results"
        / "generation"
        / args.version
        / "evaluation.json"
    )
    judged_results_path = (
        BASE_DIR
        / "results"
        / "generation"
        / args.version
        / "judged.json"
    )

    generation_results = load_json(
        generation_results_path
    )
    knowledge_base = load_json(KB_PATH)
    documents_by_id = {
        document["id"]: document
        for document in knowledge_base
    }

    client = OpenAI(api_key=api_key)
    model = "gpt-4.1-mini"
    judged_results = []

    for index, item in enumerate(
        generation_results,
        start=1,
    ):
        print(
            f"Judging {index}/{len(generation_results)}: "
            f"{item['id']}"
        )

        context = build_context(
            item["generated_sources"],
            documents_by_id,
        )

        assessment = judge_example(
            client=client,
            model=model,
            item=item,
            context=context,
        )

        judged_results.append(
            {
                **item,
                "correctness": assessment.correctness,
                "faithfulness": assessment.faithfulness,
                "judge_reason": assessment.judge_reason,
            }
        )

    judged_results_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        judged_results_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            judged_results,
            f,
            ensure_ascii=False,
            indent=2,
        )

    avg_correctness = mean(
        r["correctness"]
        for r in judged_results
    )

    avg_faithfulness = mean(
        r["faithfulness"]
        for r in judged_results
    )

    normalized_correctness = avg_correctness / 2
    normalized_faithfulness = avg_faithfulness / 2

    print("\n=== LLM JUDGE SUMMARY ===")

    print(
        f"Correctness: "
        f"{avg_correctness:.3f}/2 "
        f"({normalized_correctness:.3f})"
    )

    print(
        f"Faithfulness: "
        f"{avg_faithfulness:.3f}/2 "
        f"({normalized_faithfulness:.3f})"
    )

    print("\n=== NON-PERFECT CASES ===")

    non_perfect = [
        r
        for r in judged_results
        if r["correctness"] < 2
        or r["faithfulness"] < 2
    ]

    if not non_perfect:
        print("None")

    for result in non_perfect:
        print(
            f"\n{result['id']}: "
            f"{result['question']}"
        )
        print(
            f"Correctness: "
            f"{result['correctness']}/2"
        )
        print(
            f"Faithfulness: "
            f"{result['faithfulness']}/2"
        )
        print(
            f"Reason: "
            f"{result['judge_reason']}"
        )

    print(
        f"\nDetailed judged results saved to: "
        f"{judged_results_path}"
    )


if __name__ == "__main__":
    main()
