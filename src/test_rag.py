from rag_pipeline import run_rag


def main():
    query = "Why are there so many critical alarms from component 103?"
    answer = run_rag(query)

    print("\n🧠 ANSWER:\n")
    print(answer)


if __name__ == "__main__":
    main()
