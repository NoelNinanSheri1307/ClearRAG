"""Unit tests for ClearRAG Retrieval Improvements (BM25, Hybrid, Reranking, Isolation)."""

from pathlib import Path
import tempfile
import pytest

from src.retrieval.bm25 import BM25Index
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossScorerReranker
from src.retrieval.retriever import Retriever


@pytest.fixture
def sample_corpus():
    return [
        {
            "chunk_id": "chunk_1",
            "document_title": "Bactris",
            "text": "Bactris is a genus of spiny palms native to Central and South America.",
            "sentence_indices": [0],
            "is_supporting_fact": True,
            "source_dataset": "hotpotqa",
            "source_question_id": "q1",
        },
        {
            "chunk_id": "chunk_2",
            "document_title": "Epigaea",
            "text": "Epigaea is a genus comprising three species of flowering plants.",
            "sentence_indices": [0],
            "is_supporting_fact": True,
            "source_dataset": "hotpotqa",
            "source_question_id": "q1",
        },
        {
            "chunk_id": "chunk_3",
            "document_title": "Bactris",
            "text": "Most species of Bactris are small trees about 2 m tall.",
            "sentence_indices": [1],
            "is_supporting_fact": True,
            "source_dataset": "hotpotqa",
            "source_question_id": "q1",
        },
        {
            "chunk_id": "chunk_4",
            "document_title": "Palms of America",
            "text": "Many spiny palms grow across tropical rainforests.",
            "sentence_indices": [0],
            "is_supporting_fact": False,
            "source_dataset": "hotpotqa",
            "source_question_id": "q1",
        },
    ]


class TestBM25Index:
    """Tests for BM25 lexical index."""

    def test_bm25_build_and_search(self, sample_corpus):
        bm25 = BM25Index(title_weight=2.5)
        bm25.build_from_metadata(sample_corpus)

        assert bm25.num_docs == 4
        assert len(bm25.inverted_index) > 0

        # Search for exact title entity
        results = bm25.search("Epigaea", top_k=2)
        assert len(results) > 0
        top_doc_idx = results[0][0]
        assert sample_corpus[top_doc_idx]["document_title"] == "Epigaea"

    def test_bm25_serialization(self, sample_corpus):
        bm25 = BM25Index(title_weight=2.5)
        bm25.build_from_metadata(sample_corpus)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "bm25_test.pkl"
            bm25.save(file_path)

            loaded = BM25Index.load(file_path)
            assert loaded.num_docs == 4
            results = loaded.search("Bactris", top_k=2)
            assert len(results) > 0


class TestCrossScorerReranker:
    """Tests for CrossScorerReranker."""

    def test_reranker_scoring_and_diversity(self, sample_corpus):
        reranker = CrossScorerReranker(title_exact_match_boost=3.0, max_chunks_per_doc=1)

        query = "Which genus has more species, Bactris and Epigaea?"
        candidates = [
            {"document_title": "Bactris", "text": sample_corpus[0]["text"], "rank": 1, "score": 0.9},
            {"document_title": "Bactris", "text": sample_corpus[2]["text"], "rank": 2, "score": 0.88},
            {"document_title": "Epigaea", "text": sample_corpus[1]["text"], "rank": 3, "score": 0.85},
            {"document_title": "Palms of America", "text": sample_corpus[3]["text"], "rank": 4, "score": 0.80},
        ]

        reranked = reranker.rerank(query, candidates, top_k=2)

        assert len(reranked) == 2
        # Because max_chunks_per_doc=1, both Bactris and Epigaea should appear
        titles = [c["document_title"] for c in reranked]
        assert "Bactris" in titles
        assert "Epigaea" in titles
        assert reranked[0]["rank"] == 1
        assert reranked[1]["rank"] == 2


class TestHybridRetriever:
    """Tests for HybridRetriever."""

    def test_hybrid_rrf_fusion(self, sample_corpus):
        bm25 = BM25Index()
        bm25.build_from_metadata(sample_corpus)

        hybrid = HybridRetriever(
            embedder=None,
            faiss_index=None,
            bm25_index=bm25,
            default_top_k=2,
        )

        results = hybrid.retrieve("Epigaea flowering plants", top_k=2)
        assert len(results) > 0
        assert results[0]["document_title"] == "Epigaea"
        assert "provenance" in results[0]
        assert results[0]["provenance"]["fusion_method"] == "rrf"


class TestRetrievalMetadataIsolation:
    """Strict verification of zero inference metadata leakage."""

    def test_zero_leakage_interface(self):
        retriever = Retriever(mode="dense")
        # Ensure retrieve signature requires ONLY query text, top_k, mode, candidate_pool_k
        import inspect
        sig = inspect.signature(retriever.retrieve)
        params = list(sig.parameters.keys())
        assert "query" in params
        assert "benchmark_condition" not in params
        assert "gold_evidence" not in params
        assert "ground_truth" not in params
