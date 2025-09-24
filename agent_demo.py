import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
 
# 0. load .env
load_dotenv()
input_file = "data/input.txt"
 
# read the file
with open(input_file, "r", encoding="utf-8") as f:
    document = f.read()
 
# define tools
@tool
def summarize_text(text: str) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke(f"Briefly summarize the following document and return the summarization in korean. \n\n {text}")
    return response.content
 
# define agent
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [summarize_text]
agent = create_react_agent(llm, tools)
 
# invoke agent
result = agent.invoke({
    "messages": [("user", f"summarize the document using summarize_text tool:\n\n{document}")]
})
 
# output
print("\n res:")
print(result["messages"][-1].content)
