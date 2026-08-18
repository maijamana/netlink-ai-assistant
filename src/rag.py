from src.generation import GeneratedAnswer, Generator
from src.retrieval import Retriever


class RAGPipeline:
    def __init__(
        self,
        retrieval_k: int = 3,
        prompt_version: str = "v2",
    ):
        self.retriever = Retriever()
        self.generator = Generator(
            prompt_version=prompt_version,
        )
        self.retrieval_k = retrieval_k

    def answer(
        self,
        question: str,
    ) -> GeneratedAnswer:
        retrieved_documents = self.retriever.search(
            question,
            k=self.retrieval_k,
        )

        result = self.generator.generate(
            question=question,
            retrieved_documents=retrieved_documents,
        )

        return result
