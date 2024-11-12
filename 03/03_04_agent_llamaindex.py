from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
 
 
 


from llama_index.llms.openai_like import OpenAILike
 
# put your API key in the env variable BING_SUBSCRIPTION_KEY
from llama_index.tools.bing_search import BingSearchToolSpec


import os
api_key = os.environ["BING_SUBSCRIPTION_KEY"]
os.environ["BING_SEARCH_URL"] = "https://api.bing.microsoft.com/v7.0/search"
tool_spec = BingSearchToolSpec(api_key=api_key)
tool_list = tool_spec.to_tool_list()

llm = OpenAILike(
    is_chat_model=True,
    model="gpt-4o" 
)
agent = ReActAgent.from_tools(tool_list, llm=llm, verbose=True)

response = agent.chat("Bonjour, je suis Jack")
print(str(response))

response = agent.chat("Quel est mon nom?")
print(str(response))

response = agent.chat("Qui est le PDG d'IBM en 2024?")
print(str(response))

response = agent.chat("Traduit en anglais la phrase suivante: 'Qui est le PDG d'IBM en 2024?'")
print(str(response))
