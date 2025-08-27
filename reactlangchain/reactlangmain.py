from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()


@tool
def gettextlength(text) -> int:
    """_summary_

    Args:
        text (_type_): _description_

    Returns:
        int: _description_
    """
    return len(text)


if __name__ == '__main__':
    print("ReAct  Langchain project")
    print(gettextlength("Testing length of string"))
