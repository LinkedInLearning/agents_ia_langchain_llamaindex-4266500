# pip install llama-index-llms-openai-like
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.llms import ChatMessage

application_prompt = """Je souhaite écrire un livre sur le sujet correspondant à la description ci-dessous. Proposez-moi 3 titres accrocheurs pour ce livre. Répondez uniquement avec les titres, un par ligne, sans texte supplémentaire.
    DESCRIPTION:
"""
user_input = """Le sujet porte sur la tarte au citron"""

llm = OpenAILike(
    is_chat_model=True,     
    temperature=0.7,
    max_tokens=500,
    model="gpt-4o"
)
messages = [
    ChatMessage(role="system", content=application_prompt),
    ChatMessage(role="user", content=user_input),
]
results = llm.chat(messages)

print(results)

 