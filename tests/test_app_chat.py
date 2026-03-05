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
    monkeypatch.setattr("app.get_llm", lambda: (mock_openai, "deepseek-chat"))
    monkeypatch.setattr("app.RAGManager", lambda **kwargs: mock_rag_cls)
    
    from app import Me
    instance = Me()
    yield instance

def test_format_context(me):
    """Test that context chunks are formatted correctly with citations."""
    chunks = [
        Document(page_content="I am a Python expert.", metadata={"source": "cv.pdf"}),
        Document(page_content="I built a RAG system.", metadata={"source": "projects.md"})
    ]
    formatted = me.format_context(chunks)
    
    assert "Here is relevant context from Mohamed Abdelkrim SENOUCI's portfolio documents:" in formatted
    assert "[Source: cv.pdf]" in formatted
    assert "I am a Python expert." in formatted
    assert "[Source: projects.md]" in formatted
    assert "I built a RAG system." in formatted

def test_format_context_empty(me):
    """Test that empty context returns empty string."""
    assert me.format_context([]) == ""

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
    # Verify safety instructions for no context
    assert "No specific portfolio context found" in prompt

def test_me_chat_unknown_tool_name(me):
    """Test hallucination guard: handle_tool_call returns error for unknown tool."""
    class MockFunction:
        def __init__(self, name, arguments):
            self.name = name
            self.arguments = arguments
    class MockToolCall:
        def __init__(self, id, function):
            self.id = id
            self.function = function

    tool_calls = [MockToolCall("call_123", MockFunction("hallucinated_tool", '{"arg": 1}'))]
    results = me.handle_tool_call(tool_calls)
    
    assert "Tool 'hallucinated_tool' not found or restricted" in results[0]['content']

def test_chat_integration(me):
    """Test that chat calls query_documents and formatting."""
    me.rag.query_documents = MagicMock(return_value=[
        Document(page_content="Expert", metadata={"source": "test.txt"})
    ])
    
    # Mock streaming response
    mock_chunk = MagicMock()
    mock_chunk.choices[0].delta.content = "Hello there!"
    mock_chunk.choices[0].delta.tool_calls = None
    me.llm_client.chat.completions.create = MagicMock(return_value=[mock_chunk])

    # chat() is a generator
    responses = list(me.chat("Tell me about yourself", []))
    
    me.rag.query_documents.assert_called_with("Tell me about yourself")
    assert responses[-1] == "Hello there!"
    
    # Verify context integrity: Check that "Expert" made it into the system prompt
    args, kwargs = me.llm_client.chat.completions.create.call_args
    assert "Expert" in kwargs['messages'][0]['content']

def test_chat_history_passing(me):
    """Test that history is passed correctly to the messages list."""
    history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    me.rag.query_documents = MagicMock(return_value=[])
    
    mock_chunk = MagicMock()
    mock_chunk.choices[0].delta.content = "Response"
    mock_chunk.choices[0].delta.tool_calls = None
    me.llm_client.chat.completions.create = MagicMock(return_value=[mock_chunk])

    list(me.chat("Another question", history))
    
    # Extract the messages sent to the LLM
    args, kwargs = me.llm_client.chat.completions.create.call_args
    sent_messages = kwargs['messages']
    
    assert sent_messages[1] == {"role": "user", "content": "Hi"}
    assert sent_messages[2] == {"role": "assistant", "content": "Hello"}
    assert sent_messages[3] == {"role": "user", "content": "Another question"}

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
    
    # Mock DeepSeek streaming
    mock_chunk = MagicMock()
    mock_chunk.choices[0].delta.content = "Done"
    mock_chunk.choices[0].delta.tool_calls = None
    me.llm_client.chat.completions.create = MagicMock(return_value=[mock_chunk])

    list(me.chat("Tell me a story", []))
    
    # Verify the order passed to format_context
    args, _ = me.format_context.call_args
    passed_chunks = args[0]
    assert passed_chunks[0].page_content == "Part 1"
    assert passed_chunks[1].page_content == "Part 2"

def test_chat_tool_calling_loop(me, monkeypatch):
    """Test that the tool calling loop reassembles streamed fragments and dispatches correctly."""
    me.rag.query_documents = MagicMock(return_value=[])

    # First LLM call: streams tool call fragments across 2 chunks
    def mock_stream_tool():
        chunk1 = MagicMock()
        chunk1.choices[0].delta.content = None
        tc1 = MagicMock(); tc1.index = 0; tc1.id = "call_1"; tc1.function.name = "record_user_details"; tc1.function.arguments = '{"email":'
        chunk1.choices[0].delta.tool_calls = [tc1]
        yield chunk1

        chunk2 = MagicMock()
        chunk2.choices[0].delta.content = None
        tc2 = MagicMock(); tc2.index = 0; tc2.id = None; tc2.function.name = ""; tc2.function.arguments = ' "test@example.com"}'
        chunk2.choices[0].delta.tool_calls = [tc2]
        yield chunk2

    # Second LLM call: final text response after tool result
    def mock_stream_final():
        chunk = MagicMock()
        chunk.choices[0].delta.content = "Recorded!"
        chunk.choices[0].delta.tool_calls = None
        yield chunk

    me.llm_client.chat.completions.create = MagicMock(side_effect=[mock_stream_tool(), mock_stream_final()])

    mock_tool = MagicMock(return_value={"recorded": "ok"})
    monkeypatch.setitem(__import__("app").TOOL_REGISTRY, "record_user_details", mock_tool)

    responses = list(me.chat("contact me", []))

    assert responses[-1] == "Recorded!"
    assert mock_tool.called
    assert me.llm_client.chat.completions.create.call_count == 2

def test_me_chat_max_turns_limit(me):
    """Test that the chat loop terminates after max_turns to prevent infinite loops."""
    me.rag.query_documents = MagicMock(return_value=[])
    
    # Mock LLM to always return a tool call, never finishing
    def mock_infinite_tool():
        chunk = MagicMock()
        chunk.choices[0].delta.content = None
        tc = MagicMock()
        tc.index = 0; tc.id = "call_inf"; tc.function.name = "record_unknown_question"; tc.function.arguments = '{"question": "why?"}'
        chunk.choices[0].delta.tool_calls = [tc]
        yield chunk

    me.llm_client.chat.completions.create = MagicMock(side_effect=lambda **kwargs: mock_infinite_tool())

    # Should run 10 turns and then exit the loop
    responses = list(me.chat("Infinite loop test", []))
    
    # 10 turns = 10 calls to completions.create
    assert me.llm_client.chat.completions.create.call_count == 10

def test_chat_error_handling(me):
    """Test that chat handles RAG failures gracefully."""
    me.rag.query_documents.side_effect = Exception("Retrieval failed")
    
    mock_chunk = MagicMock()
    mock_chunk.choices[0].delta.content = "Fallback answer"
    mock_chunk.choices[0].delta.tool_calls = None
    me.llm_client.chat.completions.create = MagicMock(return_value=[mock_chunk])

    responses = list(me.chat("Help", []))
    assert responses[-1] == "Fallback answer"

def test_me_chat_empty_response_fallback(me):
    """Test that chat yields a fallback message if LLM yields nothing."""
    me.rag.query_documents = MagicMock(return_value=[])
    
    # Mock LLM to yield a single chunk with no content and no tool calls
    mock_chunk = MagicMock()
    mock_chunk.choices[0].delta.content = None
    mock_chunk.choices[0].delta.tool_calls = None
    me.llm_client.chat.completions.create = MagicMock(return_value=[mock_chunk])

    responses = list(me.chat("Hello", []))
    assert responses[-1] == "I'm sorry, I couldn't generate a response."

def test_me_chat_llm_exception_handling(me):
    """Test that chat catches LLM API exceptions and yields an error message."""
    me.rag.query_documents = MagicMock(return_value=[])
    me.llm_client.chat.completions.create.side_effect = Exception("API Timeout")

    responses = list(me.chat("Hello", []))
    assert "trouble connecting to my brain" in responses[-1]
