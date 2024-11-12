# pip install langchain-openai
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

application_prompt = """Je souhaite écrire un livre sur le sujet correspondant à la description ci-dessous. Proposez-moi 3 titres accrocheurs pour ce livre. Répondez uniquement avec les titres, un par ligne, sans texte supplémentaire.
    DESCRIPTION:
    {user_input} 
"""
user_input = """Le sujet porte sur la tarte au citron"""

llm = ChatOpenAI(
    temperature=1,
    max_tokens=500,
    model='gpt-4o'
)
prompt = PromptTemplate(  
    input_variables=["user_input"],
    template=application_prompt
)
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"user_input": user_input})

print(result)

 
