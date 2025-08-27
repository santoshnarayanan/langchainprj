from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool
from langchain_core.tools import render_text_description
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor

load_dotenv()


@tool
def gettextlength(text: str) -> int:
    """Return the length of the input string in characters."""
    return len(text)


def build_prompt(tools):
    template = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}
"""
    # In LC 0.3, PromptTemplate lives in langchain_core.prompts and wants explicit variables.
    prompt = PromptTemplate(
        template=template,
        input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
    ).partial(
        tools=render_text_description(tools),
        tool_names=", ".join(t.name for t in tools),
    )
    return prompt


if __name__ == "__main__":
    print("ReAct LangChain project")
    tools = [gettextlength]

    prompt = build_prompt(tools)

    # OpenAI (requires OPENAI_API_KEY in your environment)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    result = executor.invoke({"input": "What is the length of 'DOG' in characters?"})
    print("Final:", result["output"])
