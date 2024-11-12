from langchain.agents import AgentType, Tool, initialize_agent 
from langchain_openai.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory 
# put your API key in the env variable BING_SUBSCRIPTION_KEY
from langchain.utilities.bing_search import BingSearchAPIWrapper

import os
os.environ["BING_SEARCH_URL"] = "https://api.bing.microsoft.com/v7.0/search"

llm = ChatOpenAI(  
    temperature=0,
    model='gpt-4o'
)

tools = [   
    Tool(
        name="My Bing Web Search",
        func=BingSearchAPIWrapper().run,
        description="Get information from the web",
    )
]

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="output"
)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True
)

pydict = agent.invoke({"input": "Bonjour, je suis Jack"})
print(pydict["output"])
pydict = agent.invoke({"input": "Quel est mon nom?"})
print(pydict["output"])
pydict = agent.invoke({"input": "Qui est le PDG d'IBM en 2015?"})
print(pydict["output"])
pydict = agent.invoke({"input": "Traduit en anglais la phrase suivante: 'Qui est le PDG d'IBM en 2015?'"})
print(pydict["output"]) 


