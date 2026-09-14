from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

from langchain.agents import create_agent # The agent creation function
from langchain.tools import tool # The decorator to create a tool from a function
from langchain_core.messages import HumanMessage # The message type for human input
from langchain_google_genai import ChatGoogleGenerativeAI # The brain for the agent, using Google 
from langchain_tavily import TavilySearch # The search tool for the agent

class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="Thr agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )

# from schemas import AgentResponse # The response format for the agent

# @tool # The decorator to create a tool from a function
# def search(query: str) -> str:
#     """
#     Tool: that searches over internal documentation and returns relevant information based on the query.
#     Args: 
#         query: str: The search query provided by the user.
#     Returns:
#         The search results or relevant information based on the query.
#     """
#     print(f"Searching for: {query}")  # Log the search query
#     # Placeholder implementation replace with actual search logic
#     return f"Results for query: {query}"

    
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash") # Initialize the Google AI model

tools = [TavilySearch()] # List of tools available to the agent

agent = create_agent(model=llm,
                     tools=tools,
                     response_format=AgentResponse) # Create the agent with the model and tools)






def main():
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "search for 3 job postings for an ai engineer "
                    "using langchain in cairo on linkedin and list their details",
                }
            ]
        }
    )

    # Get the last message (the agent's final answer, after any tool calls)
    final_message = result["messages"][-1]

    # Handle both plain string content and the structured list format
    if isinstance(final_message.content, str):
        print(final_message.content)
    else:
        text = "".join(
            block["text"] for block in final_message.content if block.get("type") == "text"
        )
        print(text)

if __name__ == "__main__":
    main()