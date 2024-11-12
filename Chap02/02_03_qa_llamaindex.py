# pip install llama-index
from llama_index.core import Document, VectorStoreIndex



documents = [
    Document(text="Garry Kasparov est né le 13 avril 1963 à Bakou, en Azerbaïdjan, et est devenu l'un des plus grands champions d'échecs de tous les temps."),
    Document(text="En 1985, à seulement 22 ans, Kasparov est devenu le plus jeune champion du monde d'échecs en battant Anatoli Karpov, un titre qu'il a conservé jusqu'en 2000."),
    Document(text="Kasparov est célèbre pour ses matchs historiques contre des ordinateurs, notamment contre Deep Blue, une intelligence artificielle développée par IBM, qu'il a affrontée en 1996 et 1997."),
    Document(text="En 2005, Kasparov a pris sa retraite des compétitions d'échecs pour se consacrer à la politique et à la lutte pour la démocratie en Russie."),   
    Document(text="Auteur prolifique et orateur, Kasparov continue d'influencer les domaines des échecs, de la politique et de la technologie grâce à ses analyses et ses prises de position."),
    Document(text="Zinédine Zidane est né le 23 juin 1972 à Marseille, en France, et est devenu l’un des plus grands footballeurs de tous les temps, reconnu pour son talent et son élégance sur le terrain."),
    Document(text="En 1998, Zidane a mené l'équipe de France à la victoire en Coupe du Monde, marquant deux buts de la tête en finale contre le Brésil."),
    Document(text="Zidane a également remporté le Ballon d'Or en 1998, récompensant le meilleur joueur de football de l'année, après sa performance exceptionnelle lors de la Coupe du Monde."),
    Document(text="Après une carrière de joueur brillante, Zidane est devenu entraîneur et a conduit le Real Madrid à remporter trois Ligues des champions consécutives entre 2016 et 2018."),
    Document(text="Zidane est souvent salué pour sa vision du jeu, sa technique et son calme, et il reste une figure emblématique du football français et mondial."),
]



index = VectorStoreIndex(documents)
query_engine = index.as_query_engine()
response1 = query_engine.query("Qu'est-ce qu'il est devenu, Zinédine ? ")
print(response1)

response2 = query_engine.query("Qui a fait de la politique ?")
print(response2)

response3 = query_engine.query("Quelles sont les personnes concernées par nos docs ?")
print(response3)

 


 

