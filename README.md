This is a project that demonstrates how LlamaIndex, LangChain, and LangGraph can be integrated to create a "Smart Financial Analyst Agent." 
This agent can answer questions by either using its internal knowledge, retrieving information from specific documents (RAG), or using external tools like a search engine.

Design Overview: 
 The "Smart Financial Analyst Agent" is designed to answer a variety of financial questions.
 1.  **LlamaIndex (RAG)**: Handles questions that require specific, domain-specific knowledge found in provided documents.
     For this example, we'll use a mock financial report.
 2.  **LangChain (Tools)**: Provides the agent with the ability to use external tools. We will use a search tool
     (TavilySearch) and a math tool to answer questions that require up-to-date or numerical data.
 3.  **LangGraph (Orchestration)**: Acts as the "brain" of the agent. It defines a state machine that decides
     which path to take for a given user query:
     - Call a LangChain tool? (e.g., for a search query)
     - Perform a RAG query using LlamaIndex? (e.g., for a question about a provided document)

# To-Do:
*  Integrate More Tools: Add tools for retrieving news articles, accessing a stock price API, or connecting to a company's internal knowledge base. You could even create a tool for a SQL database to query structured data.
*  Implement a Chat Interface: The current example is a command-line script. You could wrap it in a simple web or desktop application to make it interactive.
*  Enhance Decision-Making: Refine the LLM's prompt in the decide_to_act node to include more nuanced decision-making logic. You could also use a more sophisticated model or few-shot examples to improve the routing accuracy.
*  Add Error Handling: Implement more robust error handling and fallback mechanisms within the graph to gracefully handle failed tool calls or empty search results.
