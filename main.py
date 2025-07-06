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
    Prompt = """
           You are diagnostic assistant
           Your goal is: Maintains a probability-ranked differential diagnosis with the top three most likely conditions, updating probabilities in a Bayesian manner after each new finding. nothing more
           Dont act as patient
           """

    Reply = CallLlmWithChatHistory(GoogleLlm, State, Prompt)
    #print("\n")
    #print(Reply.content)
    return {"Messages": [AIMessage(content = Reply.content)]}

def ChallengerAgent(State: state):
    Prompt = """
            You are diagnostic assistant
            Act as devil’s advocate by identifying potential anchoring bias, highlighting contradictory evidence, and proposing tests that could falsify the current leading diagnosis.
            Dont act as Patient
            """


    Reply = CallLlmWithChatHistory(GoogleLlm, State, Prompt)
    return {"Messages": [AIMessage(content = Reply.content)]}

def TestChooserAgent(State: state):
    Prompt = """
            You are diagnostic assistant
             your goal is: Selects up to three diagnostic tests per round that maximally discriminate between leading hypotheses nothing more
             Dont act as Patient
            """
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
    Prompt = """
            Your role is to act as a Medical Interaction Agent. Based on the entire previous chat history, you must decide the most appropriate immediate action to take to progress the patient's diagnostic process.

            You have three possible actions:
            
            * "AskQuestion": If more information is needed from the patient (symptoms, history, context, clarification).
            * "RequestTest": If a specific diagnostic test (lab, imaging, physical exam maneuver, etc.) is the most logical next step to gather crucial objective data or differentiate between leading diagnoses.
            * "ProvideDiagnosis": If sufficient information has been gathered to confidently (or with the highest possible confidence given the available data) provide a primary diagnosis.
            
            Consider the following in your decision:
            
            * Sufficiency of Information: Is there enough data to confidently form a diagnosis, or is key information missing?
            * Diagnostic Ambiguity: Are there still multiple plausible diagnoses that need to be narrowed down?
            * Urgency: Does the patient's reported condition suggest a need for immediate objective data (tests) versus further subjective information (questions)?
            * Completeness of History: Has a thorough history been taken?
            
            Based on the full chat history, decide the next logical step and respond with only one of the following exact phrases:
            
            "AskQuestion"
            "RequestTest"
            "ProvideDiagnosis"
            """
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
    Prompt = """
            You are a diagnostic assistant
            Please evaluate the current state of the diagnostic process. Based on the complete chat history, including all questions asked, tests requested, and the information gathered from the patient's responses and test results, determine if we have sufficient information from the agents to proceed by relaying the questions and tests to the user, or if another round of agent consultation for further questions and tests is necessary.
            
            Respond with either "Yes" (to proceed and relay) or "No" (to do another round of agent consultation).
            """
    ProceedLlm = CallLlmWithChatHistory(ProceedLlm, State, Prompt)
    return {"Next": ProceedLlm.ShouldProceed, "Iterations": Iterations}

def AskQuestion(State: state):
    Prompt = """
            You are a diagnostic assistant and given the previous information
            Based on the preceding information with the patient, what is the single most important diagnostic question you would ask next?
            Dont act as Patient
            """
    Reply = CallLlmWithChatHistory(GoogleLlm, State, Prompt)
    print("\nQuestions To ask\n")
    print(Reply.content)
    return {"Messages": [AIMessage(content = Reply.content)]}
def RequestTest(State: state):
    Prompt = """
            You are a diagnostic assistant and given the previous information
            Based on the preceding information with the patient, what is the single most important diagnostic test you would ask next?
            Dont act as Patient
            """
    Reply = CallLlmWithChatHistory(GoogleLlm, State, Prompt)
    print("\nTests Wanted\n")
    print(Reply.content)
    return {"Messages": [AIMessage(content = Reply.content)]}

def ProvideDiagnosis(State: state):
    Prompt = """
            You are a diagnostic assistant and given the previous information
            Your goal is to provide the **final diagnosis** with the highest level of confidence, based on the entirety of the previous chat.
        
            To achieve this, you must:
            
            1.  Synthesize all available information: Review every piece of data, including symptoms, patient history, physical exam findings, and test results discussed previously.
            2.  Evaluate the likelihood of all considered diagnoses: Based on the aggregated evidence, determine which single diagnosis is most strongly supported.
            3.  State the diagnosis clearly: Provide the final diagnosis in a direct and unambiguous manner.
            4.  Justify the diagnosis with key supporting evidence: Briefly explain why this diagnosis is the most confident conclusion, citing the most compelling pieces of evidence from the chat history.
            5.  Acknowledge any remaining uncertainties (briefly): If there are minor inconsistencies or areas where further clarification could be beneficial (but not enough to undermine the primary diagnosis), mention them concisely.
            
            ---
            
            Based on the complete chat history, what is your final diagnosis with the highest confidence level? Please justify your conclusion with the most pertinent supporting evidence.
            """

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
    UserInput = input("Enter your message: ")
    GraphState = {"Messages" : [], "Next" : None, "Iterations" : 0}
    while UserInput != "Exit":
        GraphState = Graph.invoke({"Messages" : GraphState.get("Messages", []) + [HumanMessage(UserInput)], "Next" : None, "Iterations" : 0})
        UserInput = input("Enter a message: ")

if __name__ == "__main__":
    AiAssistant()