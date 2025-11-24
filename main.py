from dotenv import load_dotenv
import os

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool # Allows us to have langchain run specified functions
from langchain.agents import create_openai_tools_agent, AgentExecutor

from todoist_api_python.api import TodoistAPI

# Load API Keys

load_dotenv()

todoist_api_key = os.getenv("TODOIST_API_KEY")
todoist = TodoistAPI(todoist_api_key)

gemini_api_key = os.getenv("GEMINI_API_KEY")

@tool
def add_task(task, desc=None): # Adding arguments allows ai agent to send values to function
    """Adds a new task to the users task list.
    Use this when the user wants to add or create a task
    """

    todoist.add_task(content=task, description=desc)
    print(task)
    print("Task added")

@tool
def show_tasks():
    """Gets all user tasks from todoist and prints them
    Use this tool when user wants to see their list.
    """
    tasks = []
    results_paginator = todoist.get_tasks()
    for task_list in results_paginator:
        for task in task_list:
            tasks.append(task.content)
    return tasks

def main():

    # Identfiy which functions are agent tools
    tools = [add_task, show_tasks]  # Needed to have agent run functions

    #Initialize LLM

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key = gemini_api_key,
        temperature=0.3 # closer to 0 means model chooses safer answers, higher numbers result in more creative answers
    )

    #Initialze Prompts
    system_prompt = """You are a helpful assistant. 
    You will help the user add tasks,
    You will help the user show existing tasks. If a user asks to see the tasks in any way
    print out the task to the user. Print them in a bullet list format.
    """ # Prompt to llm/agent reads to identify its purpose

    prompt = ChatPromptTemplate([
        ("system", system_prompt),
        MessagesPlaceholder("history"),
        ("user", "{input}"),
        MessagesPlaceholder("agent_scratchpad"), # placeholder for agent

    ])

    # Initialze Agent

    # Chain essentially creates a chatbot
    # chain = prompt | llm | StrOutputParser() # Prompt is sent to llm and result is StrOutputParser
    # response = chain.invoke({"input":user_input}) # Allows us to be able to view an actual response from the llm

    # Agent is able to execute tools provided in tools list, will only answer related to system prompt
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)


    history = []
    while True:
        user_input = input("You: ")
        response = agent_executor.invoke({"input": user_input, "history": history}) # Sent to prompt
        print(response["output"])
        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=response["output"]))

if __name__ == "__main__":
    main()