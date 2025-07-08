from sys import api_version

from dotenv import load_dotenv
from typing import Annotated, Literal

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from prompts_loader import LoadPrompts
Prompts = LoadPrompts(".prompts")

load_dotenv()

#GoogleLlm = AzureChatOpenAI(azure_deployment="gpt4o", api_version="2024-10-21")
GoogleLlm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
class state(TypedDict):
    Messages: Annotated[list, add_messages]
    Next: str | None
    Iterations : int


GraphBuilder = StateGraph(state)


def CallLlmWithChatHistory(Llm, State, Prompt):
    Messages = State.get("Messages", []) + [HumanMessage(content = Prompt)]
    Reply = Llm.invoke(Messages)
    return Reply

def HypothesisAgent(State: state):
    Prompt = Prompts["HypothesisAgent"]

    Reply = CallLlmWithChatHistory(GoogleLlm, State, Prompt)
    #print("\n")
    #print(Reply.content)
    return {"Messages": [AIMessage(content = Reply.content)]}

def ChallengerAgent(State: state):
    Prompt = Prompts["ChallengerAgent"]

    Reply = CallLlmWithChatHistory(GoogleLlm, State, Prompt)
    return {"Messages": [AIMessage(content = Reply.content)]}

def TestChooserAgent(State: state):
    Prompt = Prompts["TestChooserAgent"]
    Reply = CallLlmWithChatHistory(GoogleLlm, State, Prompt)
    return {"Messages": [AIMessage(content = Reply.content)]}

def ChecklistAgent(State: state):
    pass


class action_taken(BaseModel):
    ActionType: Literal["AskQuestion", "RequestTest", "ProvideDiagnosis"] = Field(
        ...,
        description="See if we require to ask a question about the patient, ask the patient to take a test or we are confident enough to provide final diagnosis."
    )

def ActionChooser(State: state):
    ActionLlm = GoogleLlm.with_structured_output(action_taken)
    Prompt = Prompts["ActionChooser"]

    ActionResult = CallLlmWithChatHistory(ActionLlm, State, Prompt)
    return {"Next": ActionResult.ActionType}

class proceed_data(BaseModel):
    ShouldProceed: Literal["Yes", "No"] = Field(
        ...,
        description="Yes means we need to proceed with another round of consultation to get further tests and questions and No means we should relay the tests and questions to the user"
    )

def Proceed(State: state):

    Iterations = State.get("Iterations") + 1
    if(Iterations >= 3):
        return {"Next": "No"}

    ProceedLlm = GoogleLlm.with_structured_output(proceed_data)
    Prompt = Prompts["Proceed"]
    ProceedLlm = CallLlmWithChatHistory(ProceedLlm, State, Prompt)
    return {"Next": ProceedLlm.ShouldProceed, "Iterations": Iterations}

def AskQuestion(State: state):
    Prompt = Prompts["AskQuestion"]
    Reply = CallLlmWithChatHistory(GoogleLlm, State, Prompt)
    print("\nQuestions To ask\n")
    print(Reply.content)
    return {"Messages": [AIMessage(content = Reply.content)]}
def RequestTest(State: state):
    Prompt = Prompts["RequestTest"]
    Reply = CallLlmWithChatHistory(GoogleLlm, State, Prompt)
    print("\nTests Wanted\n")
    print(Reply.content)
    return {"Messages": [AIMessage(content = Reply.content)]}

def ProvideDiagnosis(State: state):
    Prompt = Prompts["ProvideDiagnosis"]

    Reply = CallLlmWithChatHistory(GoogleLlm, State, Prompt)
    print("\nFinal Diagnosis\n")
    print(Reply.content)
    return {"Messages": [AIMessage(content = Reply.content)]}


GraphBuilder.add_node("HypothesisAgent", HypothesisAgent)
GraphBuilder.add_node("ChallengerAgent", ChallengerAgent)
GraphBuilder.add_node("TestChooserAgent", TestChooserAgent)
### GraphBuilder.add_node("ChecklistAgent", ChecklistAgent)

GraphBuilder.add_node("ActionChooser", ActionChooser)
GraphBuilder.add_node("Proceed", Proceed)

GraphBuilder.add_node("AskQuestion", AskQuestion)
GraphBuilder.add_node("RequestTest", RequestTest)
GraphBuilder.add_node("ProvideDiagnosis", ProvideDiagnosis)


GraphBuilder.add_edge(START, "HypothesisAgent")
GraphBuilder.add_edge("HypothesisAgent", "ChallengerAgent")
GraphBuilder.add_edge("ChallengerAgent", "TestChooserAgent")
GraphBuilder.add_edge("TestChooserAgent", "ActionChooser")
GraphBuilder.add_conditional_edges("ActionChooser", lambda State: State.get("Next"), {"AskQuestion" : "AskQuestion",
                                                                                      "RequestTest" : "RequestTest", "ProvideDiagnosis" : "ProvideDiagnosis"} )
GraphBuilder.add_edge("AskQuestion", "Proceed")
GraphBuilder.add_edge("RequestTest", "Proceed")
GraphBuilder.add_edge("ProvideDiagnosis", "Proceed")

GraphBuilder.add_conditional_edges("Proceed", lambda State: State.get("Next"), {"Yes" : "HypothesisAgent", "No" : END})


Graph = GraphBuilder.compile()
def AiAssistant():
    UserInput = ""
    while UserInput == "":
        UserInput = input("Enter a message: ")
    GraphState = {"Messages" : [], "Next" : None, "Iterations" : 0}
    while UserInput != "Exit":
        GraphState = Graph.invoke({"Messages" : GraphState.get("Messages", []) + [HumanMessage(UserInput)], "Next" : None, "Iterations" : 0})
        UserInput=""
        while UserInput == "":
            UserInput = input("Enter a message: ")

if __name__ == "__main__":
    AiAssistant()