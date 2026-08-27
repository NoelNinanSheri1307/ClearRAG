"""Unit tests for ClearRAG AttributionEngine."""

import pytest
from src.generation.attribution import AttributionEngine, AnswerClaimAttribution


class TestAttributionEngine:
    """Tests for AttributionEngine sentence segmentation and citation alignment."""

    def test_sentence_segmentation(self):
        engine = AttributionEngine()
        text = "Thomas Carr was born in 1904. He was an American film director. He died in 1997."
        sentences = engine.segment_sentences(text)
        assert len(sentences) == 3
        assert "1904" in sentences[0]
        assert "director" in sentences[1]

    def test_explicit_citation_extraction(self):
        engine = AttributionEngine()
        text = "Thomas Carr was born in 1904 [1]. He directed The Long Riders [2, 3]."
        cits1 = engine.extract_explicit_citations(text)
        assert cits1 == [1, 2, 3]

    def test_attribute_answer_with_explicit_citations(self):
        engine = AttributionEngine()
        answer = "Bactris is a genus of spiny palms [1]. Epigaea has 3 species [2]."
        evidence = [
            {"chunk_id": "c1", "document_title": "Bactris", "text": "Bactris is a genus of spiny palms native to the Americas."},
            {"chunk_id": "c2", "document_title": "Epigaea", "text": "Epigaea comprises three species of flowering plants."},
        ]
        attributions = engine.attribute_answer(answer, evidence)
        assert len(attributions) == 2
        assert attributions[0].is_supported is True
        assert 1 in attributions[0].cited_chunk_indices
        assert attributions[1].is_supported is True
        assert 2 in attributions[1].cited_chunk_indices

    def test_attribute_answer_lexical_fallback(self):
        engine = AttributionEngine(min_token_overlap_ratio=0.25)
        answer = "Walter Hill directed the western film."
        evidence = [
            {"chunk_id": "c10", "document_title": "Walter Hill", "text": "Walter Hill directed many famous western films."},
        ]
        attributions = engine.attribute_answer(answer, evidence)
        assert len(attributions) == 1
        assert attributions[0].is_supported is True
        assert "c10" in attributions[0].supporting_chunk_ids
