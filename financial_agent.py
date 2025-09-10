# This project demonstrates a powerful AI agent that combines LlamaIndex for Retrieval-Augmented Generation (RAG),
# LangChain for external tool usage, and LangGraph for orchestrating the entire workflow.
#
# Project Description:
# The "Smart Financial Analyst Agent" is designed to answer a variety of financial questions.
#
# 1.  **LlamaIndex (RAG)**: Handles questions that require specific, domain-specific knowledge found in provided documents.
#     For this example, we'll use a mock financial report.
#
# 2.  **LangChain (Tools)**: Provides the agent with the ability to use external tools. We will use a search tool
#     (TavilySearch) and a math tool to answer questions that require up-to-date or numerical data.
#
# 3.  **LangGraph (Orchestration)**: Acts as the "brain" of the agent. It defines a state machine that decides
#     which path to take for a given user query:
#     - Call a LangChain tool? (e.g., for a search query)
#     - Perform a RAG query using LlamaIndex? (e.g., for a question about a provided document)
#     - Directly generate a response? (e.g., for a general knowledge question)

import os
import getpass
from typing import TypedDict, List, Annotated
import operator
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import tool
from langchain_community.tools.google_search import GoogleSearchAPIWrapper
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolExecutor
from langgraph.graph import StateGraph, END

# --- Environment Setup ---
# Set up your API keys. You can also set these as environment variables.
# os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
# os.environ["TAVILY_API_KEY"] = getpass.getpass("Enter your Tavily API key: ")

# --- Step 1: Initialize the LlamaIndex RAG Pipeline ---
# This part handles document-specific queries.

# Create a mock financial document. In a real-world scenario, you would load files
# from a directory.
mock_document_content = """
Company: InnovateTech Solutions
2024 Annual Report Summary

InnovateTech Solutions had a strong year in 2024, with total revenue reaching $500 million,
representing a 20% increase from the previous year. Net profit was $85 million.
The company's primary focus in 2024 was expanding its cloud services division, which
now accounts for 40% of its total revenue.
"""

if not os.path.exists("financial_documents"):
    os.makedirs("financial_documents")
with open("financial_documents/2024_InnovateTech_Report.txt", "w") as f:
    f.write(mock_document_content)

# Load the documents and create a LlamaIndex Vector Store.
documents = SimpleDirectoryReader("financial_documents").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# Wrap the LlamaIndex query engine as a LangChain-compatible tool.
rag_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="financial_document_reader",
    description="Provides information about financial reports and company-specific documents. Use this tool for questions about a company's performance, revenue, or specific details found in annual reports.",
)

# --- Step 2: Initialize the LangChain Tools ---
# These tools handle external queries.

# Use a tool for web searches. Tavily is a good alternative to Google.
search_tool = TavilySearchResults(max_results=3, name="search_tool")
# Use a simple math tool for calculations.
@tool
def calculator_tool(expression: str) -> str:
    """Performs a mathematical calculation. Use this tool for any math-related questions.
    The input must be a valid mathematical expression (e.g., '2+2', 'sqrt(9)', '500 * 0.2')."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

# Combine all tools into a list.
tools = [rag_tool, search_tool, calculator_tool]
tool_executor = ToolExecutor(tools)

# --- Step 3: Define the LangGraph Agent ---
# This part orchestrates the flow using a state machine.

class AgentState(TypedDict):
    """The state of the agent's workflow."""
    input: str
    intermediate_steps: Annotated[list, operator.add]
    agent_outcome: str

# Define the nodes (actions) in our graph.
def call_tool(state):
    action = state['agent_outcome']
    tool_name = action.tool
    tool_input = action.tool_input
    
    if tool_name == "financial_document_reader":
        print(f"\n--- DEBUG: RAG Tool Invoked ---")
        print(f"Query: {tool_input}\n")
    else:
        print(f"\n--- DEBUG: LangChain Tool Invoked ---")
        print(f"Tool: {tool_name}, Input: {tool_input}\n")

    # Use the tool executor to call the selected tool.
    tool_output = tool_executor.invoke(
        {"tool": tool_name, "tool_input": tool_input}
    )
    
    return {"intermediate_steps": [(action, tool_output)]}

def execute_rag(state):
    action = state['agent_outcome']
    tool_input = action.tool_input
    
    print(f"\n--- DEBUG: RAG Query Executed ---")
    print(f"Query: {tool_input}\n")

    # Directly use the RAG query engine to get the result.
    response = query_engine.query(tool_input)
    
    return {"agent_outcome": str(response)}

# Define the master node for decision making.
def decide_to_act(state):
    """Decides whether to call a tool, perform RAG, or generate a final answer."""
    input_text = state['input']
    
    # Use the LLM to decide on the next action.
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Simple prompt to guide the LLM's decision-making.
    prompt = f"""
    You are a helpful assistant. Based on the user's input, decide what action to take.
    
    - If the user asks a question that requires a web search for recent information, choose 'search_tool'.
    - If the user asks about a mathematical calculation, choose 'calculator_tool'.
    - If the user asks about a company's annual report or specific financial data (like revenue), choose 'financial_document_reader'.
    - Otherwise, respond directly to the user's question without using any tools.
    
    User Input: {input_text}
    
    Respond with only the tool name to use, or 'direct_response' if no tool is needed.
    """
    
    try:
        response = llm.invoke(prompt)
        decision = response.content.strip()
        print(f"--- DEBUG: LLM Decision ---")
        print(f"Input: '{input_text}' -> Decision: '{decision}'")
        return decision
    except Exception as e:
        print(f"--- DEBUG: LLM Decision Failed ---")
        print(f"Error: {e}")
        return "direct_response" # Fallback to direct response if decision fails.

# Build the graph.
workflow = StateGraph(AgentState)

# Define the nodes.
workflow.add_node("decide_to_act", decide_to_act)
workflow.add_node("call_tool", call_tool)
workflow.add_node("execute_rag", execute_rag)

# Define the edges and conditions.
workflow.set_entry_point("decide_to_act")

# Route based on the LLM's decision.
workflow.add_conditional_edges(
    "decide_to_act",
    lambda x: x['agent_outcome'],
    {
        "search_tool": "call_tool",
        "calculator_tool": "call_tool",
        "financial_document_reader": "execute_rag",
        "direct_response": END
    },
)

workflow.add_edge("call_tool", END)
workflow.add_edge("execute_rag", END)

# Compile the graph.
app = workflow.compile()

# --- Run the Agent with Different Queries ---

def run_agent(query):
    print(f"\n--- USER QUERY: {query} ---")
    final_output = app.invoke({"input": query, "intermediate_steps": []})
    print(f"\n--- FINAL ANSWER ---")
    print(final_output)

if __name__ == "__main__":
    # Example 1: A general knowledge question (should get a direct response).
    run_agent("What is the capital of France?")

    # Example 2: A question that requires a web search (should use the search_tool).
    run_agent("What is the current stock price of Apple?")

    # Example 3: A question that requires a calculation (should use the calculator_tool).
    run_agent("What is the result of 1500 multiplied by 5?")

    # Example 4: A question that requires RAG from the provided document.
    run_agent("What was Google' revenue in 2024?")
