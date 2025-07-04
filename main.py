from dotenv import load_dotenv
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()

GoogleLlm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20")

class state(TypedDict):
    Messages: Annotated[list, add_messages]


GraphBuilder = StateGraph(state)

def ChatBot(State: state):
    return {"Messages": [GoogleLlm.invoke(State["Messages"])]}


GraphBuilder.add_node("ChatBot", ChatBot)
GraphBuilder.add_edge(START, "ChatBot")
GraphBuilder.add_edge( "ChatBot", END)

Graph = GraphBuilder.compile()
UserInput = input("Enter a message: ")
State = Graph.invoke({"Messages" : [{"role" : "user", "content" : UserInput}]})
print(State["Messages"][-1].content)