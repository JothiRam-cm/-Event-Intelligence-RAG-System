from retriever import get_relevant_events


def main():
    query = "Why are there so many critical alarms from component 103?"

    docs = get_relevant_events(query)

    print(f"\n🔍 Retrieved {len(docs)} documents\n")

    for i, doc in enumerate(docs, 1):
        print(f"--- Document {i} ---")
        print(doc.page_content[:600])
        print("Metadata:", doc.metadata)
        print()


if __name__ == "__main__":
    main()
