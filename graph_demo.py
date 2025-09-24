import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict

# load .env and an input file
load_dotenv()
input_file = "data/input.txt"

# define state
class State(TypedDict):
    text: str
    summary: str

# model
model = ChatOpenAI(model="gpt-4o", temperature=0)

# step 1: read the file
def read(state: State) -> State:
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    return {"text": content, "summary": ""}

# step 2: summarize the file 
def summarize(state: State) -> State:
    text = state["text"]
    prompt = f"Briefly summarize the following document and return the summarization in korean. \n\n {text}"
    response = model.invoke(prompt)

    return {"text": text, "summary":response.content}

# step 3: return the summary
def output(state: State): 
    print(state["summary"])

    return state

# configure LangGraph
workflow = StateGraph(State)
workflow.add_node("read", read)
workflow.add_node("summarize", summarize)
workflow.add_node("output", output)

workflow.set_entry_point("read")
workflow.add_edge("read", "summarize")
workflow.add_edge("summarize", "output")
workflow.add_edge("output", END)

app = workflow.compile()
app.invoke({"text": "", "summary": ""})
