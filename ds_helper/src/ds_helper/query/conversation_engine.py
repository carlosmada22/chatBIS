#!/usr/bin/env python3
"""
Simplified Conversation Engine for DS Helper

This module provides a simplified conversation engine that maintains memory
across multiple interactions using LangGraph's state management, but focuses
exclusively on RAG functionality without function calling or routing.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict, Annotated
from datetime import datetime
import uuid

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from .query import DSHelperRAGQueryEngine

# Configure logging
logger = logging.getLogger(__name__)

# Try to import Ollama
try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    logger.warning("Langchain Ollama package not available.")
    OLLAMA_AVAILABLE = False


class ConversationState(TypedDict):
    """State for the conversation graph."""
    messages: Annotated[List[BaseMessage], "The conversation messages"]
    user_query: str
    rag_context: List[Dict]
    response: str
    session_id: str
    token_count: int


class DSHelperConversationEngine:
    """Simplified conversation engine for DS Helper with memory and RAG integration."""

    def __init__(
        self, 
        db_path: str = "ds_helper_vectordb",
        collection_name: str = "ds_helper_docs",
        model: str = "gpt-oss:20b", 
        memory_db_path: str = "ds_helper_conversation_memory.db"
    ):
        """
        Initialize the conversation engine.

        Args:
            db_path: Path to the ChromaDB database directory
            collection_name: Name of the ChromaDB collection
            model: Ollama model to use
            memory_db_path: Path to SQLite database for conversation memory
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.model = model
        self.memory_db_path = memory_db_path

        # Initialize RAG engine
        self.rag_engine = DSHelperRAGQueryEngine(
            db_path=db_path,
            collection_name=collection_name,
            model=model
        )

        # Initialize LLM for conversation
        if OLLAMA_AVAILABLE:
            self.llm = ChatOllama(model=model)
        else:
            self.llm = None
            logger.warning("Ollama not available. Conversation features will be limited.")

        # Initialize memory
        self.memory = SqliteSaver.from_conn_string(f"sqlite:///{memory_db_path}")

        # Build the conversation graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph conversation flow."""

        def rag_agent(state: ConversationState) -> ConversationState:
            """RAG agent for documentation queries."""
            try:
                # Get relevant chunks for the current query
                relevant_chunks = self.rag_engine.retrieve_relevant_chunks(
                    state["user_query"], top_k=3
                )
                state["rag_context"] = relevant_chunks
                logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks")

                # Generate RAG response
                if not OLLAMA_AVAILABLE or not self.llm:
                    state["response"] = "Ollama not available. Cannot generate response."
                    return state

                # Build conversation context
                messages = []

                # Add system message
                system_msg = SystemMessage(content="""You are DS Helper, a knowledgeable assistant specializing in openBIS and data store operations.
You provide friendly, clear, and accurate answers based on documentation from both openBIS and the Data Store wiki.

Guidelines:
- Be helpful and conversational
- Provide step-by-step instructions when appropriate
- When information is available from both sources, prioritize Data Store wiki content for data store-specific operations
- If you're not sure about something, say so rather than guessing
- Use examples when they help clarify your answer
- Keep track of the conversation context and refer to previous messages when relevant""")
                messages.append(system_msg)

                # Add conversation history (last 5 exchanges to keep context manageable)
                if state["messages"]:
                    recent_messages = state["messages"][-10:]  # Last 10 messages (5 exchanges)
                    messages.extend(recent_messages)

                # Add context from RAG
                if relevant_chunks:
                    context_parts = []
                    for i, chunk in enumerate(relevant_chunks, 1):
                        source = chunk.get('source', 'unknown')
                        title = chunk.get('title', 'Unknown Document')
                        content = chunk.get('content', '')
                        
                        # Add source indicator
                        source_label = "openBIS Documentation" if source == "openbis" else "Data Store Wiki"
                        
                        context_parts.append(f"[Context {i} - {source_label}]\nTitle: {title}\nContent: {content}\n")

                    context = "\n".join(context_parts)
                    context_msg = SystemMessage(content=f"Relevant documentation context:\n{context}")
                    messages.append(context_msg)

                # Add the current user message
                messages.append(HumanMessage(content=state["user_query"]))

                # Generate response
                response = self.llm.invoke(messages)
                state["response"] = response.content

                # Estimate token count (rough approximation)
                total_text = " ".join([msg.content for msg in messages if hasattr(msg, 'content')])
                state["token_count"] = len(total_text.split()) * 1.3  # Rough token estimation

                logger.info(f"Generated RAG response with estimated {state['token_count']} tokens")
                return state

            except Exception as e:
                logger.error(f"Error in RAG agent: {e}")
                state["response"] = f"I encountered an error: {str(e)}"
                state["rag_context"] = []
                return state

        def update_conversation(state: ConversationState) -> ConversationState:
            """Update the conversation history with the new exchange."""
            try:
                # Add the user message and AI response to the conversation history
                new_messages = []
                if state["messages"]:
                    new_messages.extend(state["messages"])
                
                # Add user message
                new_messages.append(HumanMessage(content=state["user_query"]))
                
                # Add AI response
                new_messages.append(AIMessage(content=state["response"]))
                
                state["messages"] = new_messages
                
                logger.info(f"Updated conversation history. Total messages: {len(new_messages)}")
                return state
                
            except Exception as e:
                logger.error(f"Error updating conversation: {e}")
                return state

        # Build the graph
        workflow = StateGraph(ConversationState)

        # Add nodes
        workflow.add_node("rag_agent", rag_agent)
        workflow.add_node("update_conversation", update_conversation)

        # Set entry point
        workflow.set_entry_point("rag_agent")

        # Add edges
        workflow.add_edge("rag_agent", "update_conversation")
        workflow.add_edge("update_conversation", END)

        # Compile the graph with memory
        return workflow.compile(checkpointer=self.memory)

    def create_session(self) -> str:
        """Create a new conversation session."""
        session_id = str(uuid.uuid4())
        logger.info(f"Created new session: {session_id}")
        return session_id

    def clean_response(self, response: str) -> str:
        """Remove <think></think> tags from the response."""
        # Remove everything between <think> and </think> tags (including the tags)
        cleaned = re.sub(r'<think>.*?</think>\s*', '', response, flags=re.DOTALL)
        return cleaned.strip()

    def chat(self, user_input: str, session_id: Optional[str] = None) -> Tuple[str, str, Dict]:
        """
        Process a user input and return the response.

        Args:
            user_input: The user's message
            session_id: Optional session ID for conversation continuity

        Returns:
            Tuple of (response, session_id, metadata)
        """
        if not session_id:
            session_id = self.create_session()

        # Create config for this conversation thread
        config = RunnableConfig(
            configurable={"thread_id": session_id}
        )

        # Create initial state
        initial_state = ConversationState(
            messages=[],
            user_query=user_input,
            rag_context=[],
            response="",
            session_id=session_id,
            token_count=0
        )

        try:
            # Run the conversation graph
            result = self.graph.invoke(initial_state, config)

            # Extract and clean response
            raw_response = result["response"]
            cleaned_response = self.clean_response(raw_response)

            metadata = {
                "session_id": session_id,
                "token_count": result["token_count"],
                "rag_chunks_used": len(result["rag_context"]),
                "conversation_length": len(result["messages"]),
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"Chat completed for session {session_id}: {metadata}")
            return cleaned_response, session_id, metadata

        except Exception as e:
            logger.error(f"Error in chat processing: {e}")
            return f"I encountered an error: {str(e)}", session_id, {"error": str(e)}

    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """
        Get the conversation history for a session.

        Args:
            session_id: The session ID

        Returns:
            List of conversation messages
        """
        try:
            config = RunnableConfig(configurable={"thread_id": session_id})
            
            # Get the current state
            current_state = self.graph.get_state(config)
            
            if current_state and current_state.values.get("messages"):
                messages = current_state.values["messages"]
                
                # Convert to a more readable format
                history = []
                for msg in messages:
                    if isinstance(msg, HumanMessage):
                        history.append({"role": "user", "content": msg.content})
                    elif isinstance(msg, AIMessage):
                        history.append({"role": "assistant", "content": msg.content})
                
                return history
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []

    def get_database_stats(self) -> Dict:
        """Get statistics about the vector database."""
        return self.rag_engine.get_database_stats()

    def close(self) -> None:
        """Close the conversation engine and its resources."""
        self.rag_engine.close()
        logger.info("Conversation engine closed")
