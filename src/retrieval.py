import json
import os
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import faiss
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
KB_PATH = BASE_DIR / "data" / "knowledge_base.json"


class Retriever:
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
    ):
        self.model = SentenceTransformer(model_name)

        with open(KB_PATH, "r", encoding="utf-8") as f:
            self.documents = json.load(f)

        if not isinstance(self.documents, list) or not self.documents:
            raise ValueError(
                "Knowledge base must be a non-empty JSON array."
            )

        required_fields = {"id", "title", "text"}
        for position, document in enumerate(self.documents):
            missing = required_fields.difference(document)
            if missing:
                raise ValueError(
                    "Knowledge base document "
                    f"{position} is missing fields: "
                    f"{sorted(missing)}"
                )

        self.embedding_texts = [
            f"passage: {doc['title']}\n\n{doc['text']}"
            for doc in self.documents
        ]

        self.document_embeddings = self.model.encode(
            self.embedding_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        dimension = self.document_embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.document_embeddings)

    def search(self, query: str, k: int = 3):
        if k <= 0:
            return []

        k = min(k, len(self.documents))

        query_embedding = self.model.encode(
            [f"query: {query}"],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(query_embedding, k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            doc = self.documents[idx]

            results.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "text": doc["text"],
                    "score": float(score),
                }
            )

        return results


if __name__ == "__main__":
    retriever = Retriever()

    query = "У мене червоний LOS на роутері"

    results = retriever.search(query, k=3)

    for result in results:
        print(
            result["id"],
            round(result["score"], 4),
            result["title"],
        )
