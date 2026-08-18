import os
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class GeneratedAnswer(BaseModel):
    answerable: bool = Field(
        description=(
            "Whether the retrieved context contains enough information "
            "to answer the user's question."
        )
    )
    answer: str = Field(
        description=(
            "Answer based only on the provided knowledge base context."
        )
    )
    sources: List[str] = Field(
        description=(
            "IDs of the KB articles actually used in the answer."
        )
    )


class Generator:
    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        prompt_version: str = "v2",
    ):
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Add it to the .env file."
            )

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.prompt_version = prompt_version

    def generate(
        self,
        question: str,
        retrieved_documents: List[Dict],
    ) -> GeneratedAnswer:
        context = self._build_context(retrieved_documents)

        system_prompt = self._get_system_prompt()

        user_prompt = f"""
USER QUESTION:
{question}

KNOWLEDGE BASE CONTEXT:
{context}
""".strip()

        response = self.client.responses.parse(
            model=self.model,
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
            text_format=GeneratedAnswer,
        )

        result = response.output_parsed

        if result is None:
            return self._abstain()

        return self._validate_sources(
            result,
            retrieved_documents,
        )

    def _get_system_prompt(self) -> str:
        if self.prompt_version == "v1":
            return """
You are a customer-support assistant for the NetLink internet provider.

Your task is to answer the user's question using ONLY information explicitly
supported by the provided knowledge base context.

The retrieved context may contain irrelevant documents. Do not assume that
every retrieved document is relevant.

Rules:

1. Use only facts explicitly stated in the provided context.
2. Do not add diagnoses, causes, policies, recommendations, or next steps
   unless they are explicitly supported by the context.
3. Do not combine unrelated retrieved documents merely because they were
   provided to you.
4. First determine whether the context contains sufficient evidence to answer
   the user's actual question.

5. If the question cannot be answered from the context:
   - set answerable=false;
   - clearly state that the knowledge base does not contain the required information;
   - suggest contacting an operator;
   - return sources=[].

6. If the question can be answered:
   - set answerable=true;
   - answer concisely in Ukrainian;
   - include only information necessary to answer the question;
   - do not suggest contacting an operator unless the relevant KB article
     explicitly requires escalation.

7. SOURCES:
   - include only source IDs that directly support factual claims in the answer;
   - use the smallest sufficient set of sources;
   - do not include a source merely because it was retrieved;
   - never invent source IDs.

8. When multiple retrieved documents conflict in relevance, prefer the document
   that directly addresses the specific details in the user's question.

9. If only part of the question is supported, answer only the supported part
   and explicitly state what cannot be determined from the knowledge base.

Do not use outside knowledge.
""".strip()

        if self.prompt_version == "v2":
            return """
You are a customer-support assistant for the NetLink internet provider.

Answer the user's question using ONLY information explicitly supported by the
provided knowledge base context.

Important: retrieved documents are candidates and may be irrelevant.
The presence of a related document does NOT mean that the question is answerable.

ANSWERABILITY RULE:

Set answerable=true ONLY when the provided context explicitly contains the
information required to answer the user's actual question.

A document being about the same general topic is not sufficient.

For example:
- if the question asks whether a specific service or feature is included in
  a tariff, a tariff article that does not mention that service is NOT enough
  evidence to answer;
- absence of a fact from a related article must not be interpreted as
  evidence that the fact is false.

If the required fact is not explicitly supported:
- set answerable=false;
- return sources=[];
- do not infer the answer from missing information.

GROUNDING RULES:

1. Use only facts explicitly stated in the context.
2. Do not use outside knowledge.
3. Do not invent diagnoses, policies, services, prices, causes, or next steps.
4. Retrieved documents may be irrelevant; ignore them when they do not
   directly support the answer.
5. Do not combine unrelated documents only because they were retrieved.
6. If only part of a question is supported, answer only the supported part
   and clearly state what cannot be determined.

WHEN ANSWERABLE:

- answer concisely in Ukrainian;
- include only the information needed to answer the question;
- cite only source IDs that directly support factual claims;
- use the smallest sufficient set of sources;
- never invent source IDs.

WHEN NOT ANSWERABLE:

- set answerable=false;
- return sources=[];
""".strip()

        raise ValueError(
            f"Unknown prompt version: {self.prompt_version}"
        )

    def _build_context(
        self,
        retrieved_documents: List[Dict],
    ) -> str:
        chunks = []

        for doc in retrieved_documents:
            chunks.append(
                f"""
SOURCE: {doc["id"]}
TITLE: {doc["title"]}
CONTENT:
{doc["text"]}
""".strip()
            )

        return "\n\n---\n\n".join(chunks)

    def _validate_sources(
        self,
        answer: GeneratedAnswer,
        retrieved_documents: List[Dict],
    ) -> GeneratedAnswer:
        allowed_sources = {
            doc["id"]
            for doc in retrieved_documents
        }

        answer.sources = [
            source
            for source in answer.sources
            if source in allowed_sources
        ]

        if not answer.answerable or not answer.sources:
            return self._abstain()

        return answer

    @staticmethod
    def _abstain() -> GeneratedAnswer:
        return GeneratedAnswer(
            answerable=False,
            answer=(
                "У базі знань немає достатньої інформації, "
                "щоб відповісти на це запитання. "
                "Будь ласка, зверніться до оператора."
            ),
            sources=[],
        )
