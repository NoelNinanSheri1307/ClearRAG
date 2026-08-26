"""Unit tests for PromptBuilder."""

import pytest
from src.generation.prompt_builder import PromptBuilder


def test_prompt_builder_format_context_empty():
    builder = PromptBuilder()
    context = builder.format_context([])
    assert context == "No relevant context found."


def test_prompt_builder_format_context_populated():
    builder = PromptBuilder()
    chunks = [
        {"document_title": "Bactris", "text": "Bactris is a genus of spiny palms."},
        {"document_title": "Epigaea", "text": "Epigaea is a genus of flowering plants."},
    ]
    context = builder.format_context(chunks)
    assert "[1] Document: Bactris\nBactris is a genus of spiny palms." in context
    assert "[2] Document: Epigaea\nEpigaea is a genus of flowering plants." in context


def test_prompt_builder_build_messages():
    builder = PromptBuilder(system_prompt="Custom System Prompt")
    chunks = [
        {"document_title": "Paris", "text": "Paris is the capital of France."},
    ]
    messages = builder.build_messages("What is the capital of France?", chunks)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "Custom System Prompt"
    assert messages[1]["role"] == "user"
    assert "Context:" in messages[1]["content"]
    assert "Paris is the capital of France." in messages[1]["content"]
    assert "Question: What is the capital of France?" in messages[1]["content"]
