from src.retrieval import Retriever


QUERIES = [
    "У мене червоний LOS на роутері",
    "Скільки коштує Гіга 1000?",
    "Я заплатив, але інтернет ще не з'явився",
    "Хочу поставити інтернет на паузу на два місяці",
    "Чи можна підключити кабельне телебачення?",
]


def main():
    retriever = Retriever()

    for query in QUERIES:
        print(f"\nQUERY: {query}")

        for rank, result in enumerate(
            retriever.search(query, k=3),
            start=1,
        ):
            print(
                rank,
                result["id"],
                round(result["score"], 4),
                result["title"],
            )


if __name__ == "__main__":
    main()
