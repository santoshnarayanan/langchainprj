from dotenv import load_dotenv
import os
load_dotenv()

from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

tools = [TavilySearch()]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# This prompt expects variables: ["input", "agent_scratchpad"]
react_prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
chain = agent_executor

def searchagent():
    result = chain.invoke(input={
        "input": "search for three job posting for ai engineer using langchain in the frankfurt am main area on linkedIn",
                                 })
    print(result)

if __name__ == "__main__":
    searchagent()
