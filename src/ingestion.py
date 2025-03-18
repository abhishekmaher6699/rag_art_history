import langchain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import langgraph

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS

import bs4
from bs4 import BeautifulSoup
import requests
from uuid import uuid4
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

url = "https://boisestate.pressbooks.pub/arthistory/"
page = requests.get(url)
soup = BeautifulSoup(page.content, "html.parser")

links = []
for i in soup.find_all(class_="toc__title"):
    links.append(i.find("a").get("href"))
print("--LINKS COLLECTED--")
print(len(links))

loader = WebBaseLoader(links, bs_kwargs = dict(parse_only= bs4.SoupStrainer(
    class_ = ("site-content")
)))
data = loader.load()
print("--DATA LOADED--")
print(len(data))


splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=400, chunk_overlap=0)
docs = splitter.split_documents(data)
print("--DOCUMENTS SPLIT--")
print(len(docs))
print(docs[0])

db = FAISS.from_documents(documents=docs, embedding=embeddings)
print("--VECTOR STORE CREATED--")
db.save_local("data")
print("--DATA SAVED--")