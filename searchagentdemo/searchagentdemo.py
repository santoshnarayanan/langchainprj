from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from promptdemo import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

load_dotenv()

tools = [TavilySearch()]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# This prompt expects variables: ["input", "agent_scratchpad"]
react_prompt = hub.pull("hwchase17/react")

output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
reactpromptwithformatinstructions = PromptTemplate(
    input_variables=["input", "agent_scratchpad","tool_names"],
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
).partial(format_instructions=output_parser.get_format_instructions())

agent = create_react_agent(llm=llm, tools=tools, prompt=reactpromptwithformatinstructions)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

extract_output = RunnableLambda(lambda x: x["output"])
parse_output =  RunnableLambda(lambda x: output_parser.parse(x))

chain = agent_executor | extract_output | parse_output

def searchagent():
    result = chain.invoke(input={
        "input": "search for three job posting for ai engineer using langchain in the frankfurt am main area on linkedIn",
                                 })
    print(result)

if __name__ == "__main__":
    searchagent()
