import json
from pathlib import Path
from collections import Counter

DATA_PATH = Path("data/raw/hotpotqa/hotpot_dev_distractor_v1.json")


def main():
    print("=" * 70)
    print("ClearRAG - HotpotQA Dataset Inspection")
    print("=" * 70)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"\nTotal examples: {len(data):,}")

    # Basic distributions
    question_types = Counter(item.get("type") for item in data)
    difficulty = Counter(item.get("level") for item in data)

    print("\nQuestion Type Distribution:")
    for key, value in question_types.items():
        print(f"  {key}: {value:,}")

    print("\nDifficulty Distribution:")
    for key, value in difficulty.items():
        print(f"  {key}: {value:,}")

    # Context statistics
    context_counts = []
    supporting_fact_counts = []
    supporting_document_counts = []

    for item in data:
        context = item.get("context", [])
        supporting_facts = item.get("supporting_facts", [])

        context_counts.append(len(context))
        supporting_fact_counts.append(len(supporting_facts))

        supporting_docs = set(
            fact[0] for fact in supporting_facts
        )
        supporting_document_counts.append(len(supporting_docs))

    print("\nContext Statistics:")
    print(f"  Average documents/question: "
          f"{sum(context_counts) / len(context_counts):.2f}")
    print(f"  Minimum documents/question: {min(context_counts)}")
    print(f"  Maximum documents/question: {max(context_counts)}")

    print("\nSupporting Fact Statistics:")
    print(f"  Average supporting facts/question: "
          f"{sum(supporting_fact_counts) / len(supporting_fact_counts):.2f}")
    print(f"  Minimum supporting facts/question: "
          f"{min(supporting_fact_counts)}")
    print(f"  Maximum supporting facts/question: "
          f"{max(supporting_fact_counts)}")

    print("\nSupporting Document Statistics:")
    print(f"  Average supporting documents/question: "
          f"{sum(supporting_document_counts) / len(supporting_document_counts):.2f}")
    print(f"  Minimum supporting documents/question: "
          f"{min(supporting_document_counts)}")
    print(f"  Maximum supporting documents/question: "
          f"{max(supporting_document_counts)}")

    # Show one complete example
    example = data[0]

    print("\n" + "=" * 70)
    print("FIRST EXAMPLE")
    print("=" * 70)

    print(f"\nID: {example.get('_id')}")
    print(f"Type: {example.get('type')}")
    print(f"Level: {example.get('level')}")
    print(f"Question: {example.get('question')}")
    print(f"Answer: {example.get('answer')}")

    print("\nSupporting Facts:")
    for fact in example.get("supporting_facts", []):
        print(f"  Document: {fact[0]}, Sentence Index: {fact[1]}")

    print("\nContext:")
    for title, sentences in example.get("context", []):
        is_supporting = any(
            fact[0] == title
            for fact in example.get("supporting_facts", [])
        )

        marker = "[SUPPORT]" if is_supporting else "[DISTRACTOR]"

        print(f"\n{marker} {title}")
        for i, sentence in enumerate(sentences):
            print(f"  [{i}] {sentence}")

    print("\n" + "=" * 70)
    print("Inspection complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()