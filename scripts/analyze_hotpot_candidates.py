import json
from pathlib import Path
from collections import Counter, defaultdict
from statistics import mean, median

DATA_PATH = Path("data/raw/hotpotqa/hotpot_dev_distractor_v1.json")


def analyze_question(item):
    context = item["context"]
    supporting_facts = item["supporting_facts"]

    # Map document title -> sentence indices identified as supporting facts
    support_map = defaultdict(list)

    for title, sentence_idx in supporting_facts:
        support_map[title].append(sentence_idx)

    # Context document statistics
    total_documents = len(context)
    supporting_documents = len(support_map)
    distractor_documents = total_documents - supporting_documents

    # Supporting fact statistics
    total_supporting_facts = len(supporting_facts)

    # Count sentences in each supporting document
    supporting_doc_sentence_counts = []

    for title, sentence_indices in support_map.items():
        document = next(
            (doc for doc in context if doc[0] == title),
            None
        )

        if document:
            sentences = document[1]
            supporting_doc_sentence_counts.append(
                len(sentence_indices)
            )

    # Useful for partial-evidence experiments:
    # Can we remove one supporting fact while leaving another
    # supporting fact in the same document?
    has_multi_fact_support_document = any(
        count >= 2
        for count in supporting_doc_sentence_counts
    )

    # Can we remove one entire supporting document while another
    # supporting document remains?
    has_two_supporting_documents = supporting_documents >= 2

    # Supporting facts distributed across both documents
    documents_with_multiple_supporting_facts = sum(
        count >= 2
        for count in supporting_doc_sentence_counts
    )

    return {
        "type": item.get("type"),
        "level": item.get("level"),
        "context_documents": total_documents,
        "supporting_documents": supporting_documents,
        "distractor_documents": distractor_documents,
        "supporting_facts": total_supporting_facts,
        "supporting_doc_sentence_counts": supporting_doc_sentence_counts,
        "has_multi_fact_support_document": has_multi_fact_support_document,
        "has_two_supporting_documents": has_two_supporting_documents,
        "documents_with_multiple_supporting_facts":
            documents_with_multiple_supporting_facts,
    }


def print_distribution(title, values):
    counter = Counter(values)

    print(f"\n{title}")
    print("-" * 50)

    for key in sorted(counter):
        percentage = counter[key] / len(values) * 100
        print(
            f"{str(key):>10}: "
            f"{counter[key]:>5} "
            f"({percentage:>6.2f}%)"
        )


def main():
    print("=" * 75)
    print("ClearRAG - HotpotQA Candidate Analysis")
    print("=" * 75)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"\nTotal questions: {len(data):,}")

    analyses = [
        analyze_question(item)
        for item in data
    ]

    # ---------------------------------------------------------
    # BASIC DISTRIBUTIONS
    # ---------------------------------------------------------

    print_distribution(
        "Question Type",
        [x["type"] for x in analyses]
    )

    print_distribution(
        "Difficulty",
        [x["level"] for x in analyses]
    )

    print_distribution(
        "Context Documents",
        [x["context_documents"] for x in analyses]
    )

    print_distribution(
        "Supporting Documents",
        [x["supporting_documents"] for x in analyses]
    )

    print_distribution(
        "Supporting Facts",
        [x["supporting_facts"] for x in analyses]
    )

    print_distribution(
        "Distractor Documents",
        [x["distractor_documents"] for x in analyses]
    )

    # ---------------------------------------------------------
    # NUMERICAL SUMMARY
    # ---------------------------------------------------------

    def stats(values):
        return {
            "min": min(values),
            "max": max(values),
            "mean": mean(values),
            "median": median(values),
        }

    context_stats = stats(
        [x["context_documents"] for x in analyses]
    )

    supporting_fact_stats = stats(
        [x["supporting_facts"] for x in analyses]
    )

    distractor_stats = stats(
        [x["distractor_documents"] for x in analyses]
    )

    print("\n" + "=" * 75)
    print("NUMERICAL SUMMARY")
    print("=" * 75)

    print("\nContext documents:")
    print(f"  Mean   : {context_stats['mean']:.2f}")
    print(f"  Median : {context_stats['median']:.2f}")
    print(f"  Min    : {context_stats['min']}")
    print(f"  Max    : {context_stats['max']}")

    print("\nSupporting facts:")
    print(f"  Mean   : {supporting_fact_stats['mean']:.2f}")
    print(f"  Median : {supporting_fact_stats['median']:.2f}")
    print(f"  Min    : {supporting_fact_stats['min']}")
    print(f"  Max    : {supporting_fact_stats['max']}")

    print("\nDistractor documents:")
    print(f"  Mean   : {distractor_stats['mean']:.2f}")
    print(f"  Median : {distractor_stats['median']:.2f}")
    print(f"  Min    : {distractor_stats['min']}")
    print(f"  Max    : {distractor_stats['max']}")

    # ---------------------------------------------------------
    # PARTIAL EVIDENCE CANDIDATES
    # ---------------------------------------------------------

    partial_candidates = [
        (idx, item, analysis)
        for idx, (item, analysis)
        in enumerate(zip(data, analyses))
        if analysis["has_multi_fact_support_document"]
        or analysis["has_two_supporting_documents"]
    ]

    multi_fact_candidates = [
        (idx, item, analysis)
        for idx, (item, analysis)
        in enumerate(zip(data, analyses))
        if analysis["has_multi_fact_support_document"]
    ]

    two_doc_candidates = [
        (idx, item, analysis)
        for idx, (item, analysis)
        in enumerate(zip(data, analyses))
        if analysis["has_two_supporting_documents"]
    ]

    print("\n" + "=" * 75)
    print("PARTIAL-EVIDENCE CANDIDATES")
    print("=" * 75)

    print(
        f"\nQuestions with >=2 supporting documents: "
        f"{len(two_doc_candidates):,}"
    )

    print(
        f"Questions with a supporting document containing "
        f">=2 supporting facts: {len(multi_fact_candidates):,}"
    )

    print(
        f"Questions suitable for partial-evidence experiments "
        f"(union): {len(partial_candidates):,}"
    )

    # ---------------------------------------------------------
    # MULTI-SUPPORTING-FACT STRUCTURE
    # ---------------------------------------------------------

    multi_fact_distribution = Counter()

    for analysis in analyses:
        multi_fact_distribution[
            analysis["documents_with_multiple_supporting_facts"]
        ] += 1

    print("\nDocuments containing multiple supporting facts:")
    for count in sorted(multi_fact_distribution):
        print(
            f"  {count} document(s): "
            f"{multi_fact_distribution[count]:,} questions"
        )

    # ---------------------------------------------------------
    # BRIDGE / COMPARISON CANDIDATES
    # ---------------------------------------------------------

    print("\n" + "=" * 75)
    print("CANDIDATES BY QUESTION TYPE")
    print("=" * 75)

    for question_type in ["bridge", "comparison"]:
        subset = [
            x for x in analyses
            if x["type"] == question_type
        ]

        partial = [
            x for x in subset
            if x["has_multi_fact_support_document"]
            or x["has_two_supporting_documents"]
        ]

        print(f"\n{question_type.upper()}")
        print(f"  Total: {len(subset):,}")
        print(f"  Partial candidates: {len(partial):,}")

    # ---------------------------------------------------------
    # DISTRACTOR ANALYSIS
    # ---------------------------------------------------------

    distractor_heavy = [
        x for x in analyses
        if x["distractor_documents"] >= 8
    ]

    print("\n" + "=" * 75)
    print("DISTRACTOR ANALYSIS")
    print("=" * 75)

    print(
        f"\nQuestions with >=8 distractor documents: "
        f"{len(distractor_heavy):,}"
    )

    print(
        f"Percentage of dataset: "
        f"{len(distractor_heavy) / len(data) * 100:.2f}%"
    )

    # ---------------------------------------------------------
    # UNSUPPORTED EXPERIMENT FEASIBILITY
    # ---------------------------------------------------------

    print("\n" + "=" * 75)
    print("UNSUPPORTED CONDITION")
    print("=" * 75)

    print(
        "\nUnsupported questions can be derived by removing "
        "all gold supporting evidence from the available context."
    )

    print(
        "Because every HotpotQA question has known supporting "
        "facts, we can construct this condition deterministically."
    )

    # ---------------------------------------------------------
    # CONFLICT EXPERIMENT
    # ---------------------------------------------------------

    print("\n" + "=" * 75)
    print("CONFLICT CONDITION")
    print("=" * 75)

    print(
        "\nHotpotQA does not natively provide contradiction labels "
        "for these examples."
    )

    print(
        "Conflict cases will therefore require a separate, "
        "controlled perturbation procedure."
    )

    # ---------------------------------------------------------
    # SHOW EXAMPLE CANDIDATES
    # ---------------------------------------------------------

    print("\n" + "=" * 75)
    print("EXAMPLE PARTIAL-EVIDENCE CANDIDATES")
    print("=" * 75)

    for idx, item, analysis in partial_candidates[:10]:
        print("\n" + "-" * 75)

        print(f"Dataset index : {idx}")
        print(f"ID            : {item.get('_id')}")
        print(f"Type          : {item.get('type')}")
        print(f"Question      : {item.get('question')}")
        print(f"Answer        : {item.get('answer')}")

        print(
            f"Supporting docs : "
            f"{analysis['supporting_documents']}"
        )

        print(
            f"Supporting facts: "
            f"{analysis['supporting_facts']}"
        )

        print(
            "Supporting facts per document: "
            f"{analysis['supporting_doc_sentence_counts']}"
        )

    print("\n" + "=" * 75)
    print("ANALYSIS COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()