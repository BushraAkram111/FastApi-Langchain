from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()

prompt = ChatPromptTemplate([
    ("system","you are tasked with to proivde the only ansewr using this data if you cannot find any answer of the user question only say i do not know the ansewr. {data}"),
    ("user","{user}"),
])


chain = prompt | ChatOpenAI(model="gpt-4.1-nano")

with open("data.txt","r") as df:
    content = df.read()

while True:
    user_query = input("Enter you query: ")

    response = chain.invoke({
        "user":user_query,
        "data":content
    }).content

    print(response)
