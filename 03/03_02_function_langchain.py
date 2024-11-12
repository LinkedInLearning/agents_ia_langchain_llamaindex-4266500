import json
import random
from langchain.tools import Tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType

# La fonction de récupération de la météo
def get_weather_for_city(city: str):
    print(f"Calling local get_weather_for_city for {city}")
    return json.dumps({"city": city, "temperature": random.randint(1, 70)})

# Création d'un outil Langchain pour la fonction
weather_tool = Tool(
    name="Weather Tool",
    func=lambda city: get_weather_for_city(city),
    description="Provides the weather information for a given city. Use this when a user asks for the weather in a specific city."
)

# Modèle de langage de base (par exemple, GPT-4 dans ce cas)
llm = ChatOpenAI(temperature=0)

# Initialisation de l'agent avec l'outil
agent = initialize_agent(
    tools=[weather_tool],  # Liste des outils disponibles
    llm=llm,  # Modèle de langue
    
)

# Prompt utilisateur
user_input = "Quel temps fait-il à Paris ?"

# Appel de l'agent avec la question de l'utilisateur
response = agent.invoke(user_input)

print(response)
