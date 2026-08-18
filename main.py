from src.rag import RAGPipeline


def main():
    rag = RAGPipeline(
        retrieval_k=3,
        prompt_version="v2",
    )

    questions = [
        "Я заплатив, але інтернету немає, лампочки всі зелені",
        "Скільки коштує Гіга 1000?",
        "Чи можна підключити кабельне телебачення?",
    ]

    for question in questions:
        print("\n" + "=" * 80)
        print("QUESTION:")
        print(question)

        result = rag.answer(question)

        print("\nANSWERABLE:")
        print(result.answerable)

        print("\nANSWER:")
        print(result.answer)

        print("\nSOURCES:")
        print(result.sources)


if __name__ == "__main__":
    main()
