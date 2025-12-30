import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document

# Patch heavy dependencies using monkeypatch to avoid side effects during import
@pytest.fixture
def me(monkeypatch):
    mock_openai = MagicMock()
    mock_rag_cls = MagicMock()
    
    # Use monkeypatch for cleaner dependency isolation
    monkeypatch.setattr("app.load_dotenv", lambda **kwargs: None)
    monkeypatch.setattr("app.OpenAI", lambda **kwargs: mock_openai)
    monkeypatch.setattr("app.RAGManager", lambda **kwargs: mock_rag_cls)
    
    from app import Me
    instance = Me()
    # Store mocks on instance for easier test access if needed, 
    # though existing tests use them through attributes already set in __init__
    yield instance

def test_format_context(me):
    """Test that context chunks are formatted correctly with citations."""
    chunks = [
        Document(page_content="I am a Python expert.", metadata={"source": "cv.pdf"}),
        Document(page_content="I built a RAG system.", metadata={"source": "projects.md"})
    ]
    formatted = me.format_context(chunks)
    
    assert "Here is relevant context from Senouci's portfolio documents:" in formatted
    assert "[Source: cv.pdf]" in formatted
    assert "I am a Python expert." in formatted
    assert "[Source: projects.md]" in formatted
    assert "I built a RAG system." in formatted

def test_system_prompt_with_context(me):
    """Test system prompt includes context when provided."""
    context = "Senouci knows Python."
    prompt = me.system_prompt(context)
    
    assert "Senouci knows Python." in prompt
    assert "Mohamed Abdelkrim SENOUCI" in prompt
    assert "professional and engaging" in prompt

def test_system_prompt_without_context(me):
    """Test system prompt fallback when context is empty."""
    prompt = me.system_prompt("")
    
    assert "Mohamed Abdelkrim SENOUCI" in prompt
    # Should still have the persona but no context block
    assert "Senouci knows Python" not in prompt

def test_chat_integration(me):
    """Test that chat calls query_documents and formatting."""
    me.rag.query_documents = MagicMock(return_value=[
        Document(page_content="Expert", metadata={"source": "test.txt"})
    ])
    mock_response = MagicMock()
    mock_response.choices[0].message.tool_calls = None
    mock_response.choices[0].message.content = "Hello there!"
    me.deepseek.chat.completions.create = MagicMock(return_value=mock_response)

    response = me.chat("Tell me about yourself", [])
    
    me.rag.query_documents.assert_called_with("Tell me about yourself")
    assert response == "Hello there!"

def test_chat_history_passing(me):
    """Test that history is passed correctly to the messages list."""
    history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    me.rag.query_documents = MagicMock(return_value=[])
    mock_response = MagicMock()
    mock_response.choices[0].message.tool_calls = None
    mock_response.choices[0].message.content = "Response"
    me.deepseek.chat.completions.create = MagicMock(return_value=mock_response)

    me.chat("Another question", history)
    
    # Extract the messages sent to OpenAI
    args, kwargs = me.deepseek.chat.completions.create.call_args
    sent_messages = kwargs['messages']
    
    assert sent_messages[1] == {"role": "user", "content": "Hi"}
    assert sent_messages[2] == {"role": "assistant", "content": "Hello"}
    assert sent_messages[3] == {"role": "user", "content": "Another question"}

def test_chat_error_handling(me):
    """Test that chat handles RAG failures gracefully."""
    me.rag.query_documents.side_effect = Exception("Retrieval failed")
    
    # Even if RAG fails, it should still try to answer with persona or return error
    # We want a friendly message if everything fails, 
    # but here we check if it falls back to empty context.
    
    mock_response = MagicMock()
    mock_response.choices[0].message.tool_calls = None
    mock_response.choices[0].message.content = "Fallback answer"
    me.deepseek.chat.completions.create = MagicMock(return_value=mock_response)

    response = me.chat("Help", [])
    assert response == "Fallback answer"

def test_chat_reorders_chunks(me):
    """Test that chunks are re-ordered by document sequence before formatting."""
    # Mock RAG to return chunks in "wrong" order (Similarity search might do this)
    me.rag.query_documents = MagicMock(return_value=[
        Document(page_content="Part 2", metadata={"source": "doc.txt", "chunk_index": 1}),
        Document(page_content="Part 1", metadata={"source": "doc.txt", "chunk_index": 0})
    ])
    
    # Mock format_context to see the order it receives
    original_format = me.format_context
    me.format_context = MagicMock(side_effect=original_format)
    
    # Mock DeepSeek
    mock_response = MagicMock()
    mock_response.choices[0].message.tool_calls = None
    mock_response.choices[0].message.content = "Done"
    me.deepseek.chat.completions.create = MagicMock(return_value=mock_response)

    me.chat("Tell me a story", [])
    
    # Verify the order passed to format_context
    args, _ = me.format_context.call_args
    passed_chunks = args[0]
    assert passed_chunks[0].page_content == "Part 1"
    assert passed_chunks[1].page_content == "Part 2"
