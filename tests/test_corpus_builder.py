"""Unit tests for CorpusBuilder."""

import json
from pathlib import Path
import pytest
from src.ingestion.corpus_builder import CorpusBuilder


@pytest.fixture
def sample_hotpot_raw_data():
    return [
        {
            "_id": "5a8b57f25542995d1e6f1371",
            "question": "Which film came out first, Doctor Strange or Sinister?",
            "answer": "Sinister",
            "supporting_facts": [
                ["Doctor Strange (film)", 0],
                ["Sinister (film)", 0],
            ],
            "context": [
                [
                    "Doctor Strange (film)",
                    [
                        "Doctor Strange is a 2016 American superhero film.",
                        "It was directed by Scott Derrickson.",
                    ],
                ],
                [
                    "Sinister (film)",
                    [
                        "Sinister is a 2012 supernatural horror film.",
                        "The film stars Ethan Hawke.",
                    ],
                ],
                [
                    "Distractor Article",
                    ["This is a completely unrelated distractor article."],
                ],
            ],
        }
    ]


def test_corpus_builder_parsing(sample_hotpot_raw_data, tmp_path):
    builder = CorpusBuilder(deduplicate_by_title_and_idx=True)
    chunks = builder.build_from_hotpotqa(sample_hotpot_raw_data)

    assert len(chunks) == 5  # 2 + 2 + 1 sentences
    titles = [c.document_title for c in chunks]
    assert "Doctor Strange (film)" in titles
    assert "Sinister (film)" in titles
    assert "Distractor Article" in titles

    # Check supporting facts
    doc_strange_0 = next(
        c for c in chunks
        if c.document_title == "Doctor Strange (film)" and c.sentence_indices == [0]
    )
    assert doc_strange_0.is_supporting_fact is True

    doc_strange_1 = next(
        c for c in chunks
        if c.document_title == "Doctor Strange (film)" and c.sentence_indices == [1]
    )
    assert doc_strange_1.is_supporting_fact is False

    # Test save and load
    save_path = tmp_path / "corpus.json"
    builder.save_corpus(chunks, save_path)
    assert save_path.exists()

    loaded = builder.load_corpus(save_path)
    assert len(loaded) == len(chunks)
    assert loaded[0].text == chunks[0].text
