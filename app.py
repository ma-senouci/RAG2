import argparse
import sys
import os
import json
import requests
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from rag_logic import RAGManager, logger

load_dotenv(override=True)

def push(text):
    """Send a push notification via Pushover."""
    token = os.getenv("PUSHOVER_TOKEN")
    user = os.getenv("PUSHOVER_USER")
    if token and user:
        try:
            requests.post(
                "https://api.pushover.net/1/messages.json",
                data={
                    "token": token,
                    "user": user,
                    "message": text,
                },
                timeout=5
            )
        except Exception as e:
            logger.warning(f"Failed to send push notification: {e}")

def record_user_details(email, name="Name not provided", notes="not provided"):
    """Tool to record user interest and contact info."""
    push(f"User Interest: {name} ({email}) - Notes: {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    """Tool to record questions the AI couldn't answer."""
    push(f"Unknown Question: {question}")
    return {"recorded": "ok"}

# Tool Registry for security (prevents arbitrary function execution)
TOOL_REGISTRY = {
    "record_user_details": record_user_details,
    "record_unknown_question": record_unknown_question,
}

# Tool definitions for LLM
tools = [
    {
        "type": "function",
        "function": {
            "name": "record_user_details",
            "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string", "description": "The email address of this user"},
                    "name": {"type": "string", "description": "The user's name, if they provided it"},
                    "notes": {"type": "string", "description": "Context about the conversation"}
                },
                "required": ["email"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "record_unknown_question",
            "description": "Always use this tool to record any question that couldn't be answered due to lack of information",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The unanswered question"}
                },
                "required": ["question"],
                "additionalProperties": False
            }
        }
    }
]

class Me:
    """Class representing Senouci's persona with RAG and Tool-calling capabilities."""
    
    def __init__(self):
        self.name = "Mohamed Abdelkrim SENOUCI"
        self.deepseek = OpenAI(
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            api_key=os.getenv("DEEPSEEK_API_KEY")
        )
        self.rag = RAGManager()

    def format_context(self, chunks):
        """Format retrieved chunks into a cited context block."""
        if not chunks:
            return ""
        
        context_parts = ["Here is relevant context from Senouci's portfolio documents:\n"]
        for chunk in chunks:
            source = chunk.metadata.get("source", "Unknown Source")
            # Clearer source presentation
            context_parts.append(f"[Source: {os.path.basename(source)}]")
            context_parts.append(chunk.page_content)
            context_parts.append("") # Spacer
            
        return "\n".join(context_parts)

    def system_prompt(self, context=""):
        """Generate the system prompt with optional context injection."""
        prompt = (
            f"You are acting as {self.name}. You are answering questions on {self.name}'s website, "
            f"particularly questions related to {self.name}'s career, background, skills and experience. "
            f"Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. "
            f"You are given a summary of background and CV which you can use to answer questions. "
            "Be professional and engaging, as if talking to a potential client or future employer. "
            "If you don't know the answer to any question, use your record_unknown_question tool to record it. "
            "If the user is engaging in discussion, try to steer them towards getting in touch; "
            "ask for their email and record it using your record_user_details tool."
        )

        if context:
            prompt += f"\n\n### CONTEXT ###\n{context}\n###############\n"
        else:
            prompt += f"\n\n(No specific portfolio context found for this query. Use your general knowledge of {self.name}'s background if possible, or gracefully record any unknowns.)"

        prompt += f"\n\nPlease chat with the user, ALWAYS staying in character as {self.name}."
        return prompt

    def handle_tool_call(self, tool_calls):
        """Execute tool calls and return results for the LLM."""
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            logger.info(f"Tool called: {tool_name}")
            
            # Safe lookup via registry
            tool_func = TOOL_REGISTRY.get(tool_name)
            result = tool_func(**arguments) if tool_func else {"error": f"Tool '{tool_name}' not found or restricted"}
            
            results.append({
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id
            })
        return results

    def chat(self, message, history):
        """Main chat orchestration loop: RAG -> Prompt -> LLM (with tool loop)."""
        logger.info(f"Processing message: {message[:50]}...")
        
        # 1. Retrieve relevant portfolio chunks via semantic search
        try:
            chunks = self.rag.query_documents(message)
            # Sort by source file then chunk position for coherent context
            chunks.sort(key=lambda x: (x.metadata.get("source", ""), x.metadata.get("chunk_index", 0)))
            context = self.format_context(chunks)
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            context = "Note: Portfolio search is currently unavailable."

        # 2. Augment prompt with retrieved context, persona, and conversation history
        system_content = self.system_prompt(context)
        messages = [{"role": "system", "content": system_content}] + history + [{"role": "user", "content": message}]

        # 3. Generate response via LLM, resolving any tool calls before final reply
        try:
            done = False
            max_turns = 10
            turns = 0
            while not done and turns < max_turns:
                turns += 1
                response = self.deepseek.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    tools=tools
                )
                
                msg = response.choices[0].message
                if msg.tool_calls:
                    tool_results = self.handle_tool_call(msg.tool_calls)
                    messages.append(msg)
                    messages.extend(tool_results)
                else:
                    done = True
            
            return msg.content
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return "I apologize, but I'm having trouble connecting to my brain right now. Please try again in a moment."

def main():
    parser = argparse.ArgumentParser(description="RAG2 Application")
    parser.add_argument("--sync", action="store_true", help="Trigger document discovery and indexing")
    
    args = parser.parse_args()
    
    if args.sync:
        logger.info("Starting RAG Sync Pipeline...")
        try:
            manager = RAGManager()
            docs = manager.discover_documents()
            if not docs:
                logger.warning("No documents found in 'docs/' folder. Index remains unchanged.")
                return
            
            result = manager.index_documents(docs)
            
            logger.info("=" * 40)
            logger.info("SYNC OPERATION SUMMARY")
            logger.info("-" * 40)
            logger.info(f"Documents Processed: {result.get('documents_processed')}")
            logger.info(f"Chunks Created:      {result.get('chunks_created')}")
            logger.info(f"Chunks Indexed:      {result.get('chunks_indexed')}")
            logger.info(f"Verification:        {result.get('verification', 'N/A').upper()}")
            logger.info("=" * 40)
            logger.info("Sync complete. System is ready for retrieval.")
        except Exception as e:
            logger.error(f"Sync failed: {str(e)}")
            sys.exit(1)
    else:
        logger.info("Launching portfolio chat interface...")
        me = Me()
        gr.ChatInterface(me.chat).launch()

if __name__ == "__main__":
    main()
