import os
from openai import OpenAI 


'''
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
print(OPENAI_API_KEY) 
'''

 
llm = OpenAI()

system_prompt = """Je souhaite écrire un livre sur le sujet correspondant à la description ci-dessous. Proposez-moi 3 titres accrocheurs pour ce livre. Répondez uniquement avec les titres, un par ligne, sans texte supplémentaire.
    DESCRIPTION:
"""
user_input = """Le sujet porte sur la tarte au citron"""

response = llm.chat.completions.create(
    model="gpt-4o",
    max_tokens=500,
    temperature=0.5,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
)

print(response.choices[0].message.content)
