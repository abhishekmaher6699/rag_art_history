import time
from langchain.schema import Document

from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, START, END, MessagesState

from typing_extensions import TypedDict, Annotated, List
from dotenv import load_dotenv
import os

from src.agent_comps.chains import query_construction_prompt, re_write_prompt, rag_prompt, grade_prompt, answer_prompt, initial_routing
from src.agent_comps.output_models import *

class GraphState(TypedDict):
    original_query: str
    constructed_query: str
    messages: Annotated[list, add_messages]
    generation: str
    documents: List[str]
    route_to_retrieve: int
    route_to_wiki: int
    source: str

def get_sources(response):
    docs = response['documents']
    sources = []
    for doc in docs:
        sources.append(doc.metadata['source'])
    return list(set(sources))

class Agent:

    def __init__(self, model_name, api_key):
        print("--INITIALIZING AGENT--")
        os.environ['GOOGLE_API_KEY'] = api_key
        self.model = ChatGoogleGenerativeAI(
            model=model_name,
            )
        self.wikipedia_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
        self.wikipedia_tool = WikipediaQueryRun(api_wrapper = self.wikipedia_wrapper)
        
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        self.new_vector_store = FAISS.load_local(
            "./data", self.embeddings, allow_dangerous_deserialization=True
        )
        self.retriever = self.new_vector_store.as_retriever(search_kwargs={"k": 5})

    def construct_query(self, state):
        
        print("--QUERY CONSTRUCTION--")
        
        construction_chain = query_construction_prompt | self.model | StrOutputParser()

        query = state["original_query"]
        messages = state["messages"]
        res = construction_chain.invoke({"query": query, "history": messages})
        print("Constructed Query: ", res)
        return {"messages": res, "constructed_query": res, "original_query": query, "route_to_retrieve" : 0, "route_to_wiki": 0}
    
    def retrieve(self, state):
        
        print("--RETRIEVAL--")

        query = state["constructed_query"]
        route_to_retrieve = state["route_to_retrieve"] + 1
        print(query)

        docs = self.retriever.invoke(query)

        return {"documents": docs, "constructed_query": query, "route_to_retrieve" : route_to_retrieve, "source": 'retrieval'}

    
    def grade_docs(self, state):
        
        print("--GRADING DOCUMENTS--")

        grader_model = self.model.with_structured_output(GradeDocument)
        retrieval_grader = grade_prompt | grader_model

        query = state["constructed_query"]
        docs = state["documents"]
        # print(docs)

        filtered_docs = []

        if len(docs) > 1:
            for doc in docs:
                # print(doc)
                time.sleep(1)
                score = retrieval_grader.invoke({"document": doc, "question": query})
                if score.grade == "yes":
                    print("--RELEVANT--")
                    filtered_docs.append(doc)
                else:
                    print("--IRRELEVANT--")
                continue
        else:
            time.sleep(1)
            score = retrieval_grader.invoke({"document": docs[0], "question": query})
            if score.grade == "yes":
                print("--RELEVANT--")
                filtered_docs.append(docs[0])
            else:
                print("--IRRELEVANT--")
    
        return {"documents": filtered_docs, "constructed_query": query}

    def generate(self, state):

        print("--GENERATION--")
    
        rag_chain = rag_prompt | self.model | StrOutputParser()

        query = state["constructed_query"]
        docs = state["documents"]
        messages = state["messages"]

        time.sleep(1)
        res = rag_chain.invoke({"input": query, "documents": docs, "context": state["messages"]})
        print("Generated answer: ", res)
        return {"generation": res}
    

    def answer_grade(self, state):

        print("--ANSWER GRADING--")

        answer_grader = self.model.with_structured_output(GradeAnswer)
        answer_grader_chain = answer_prompt | answer_grader

        query = state["constructed_query"]
        generation = state["generation"]

        time.sleep(1)
        score = answer_grader_chain.invoke({"question": query, "generation": generation})
        # print(score)
        if score.binary_score == 'yes':
            print("--USEFUL--")
            # state["messages"] = generation
            return "useful"
        else:
            print("--NOT USEFUL--")
            return "not useful"
        
    def save_messages(self, state):
        print("--SAVING MESSAGES--")
        generation = state["generation"]
        new_msg = AIMessage(content = generation)
        return {"messages": new_msg}
    
    def rewrite_query(self, state):

        print("--REWRITING QUERY--")
        
        question_rewriter = re_write_prompt | self.model | StrOutputParser()
        query = state["constructed_query"]

        res = question_rewriter.invoke({"question": query})
        print("Rewritten query: ", res)
        return {"constructed_query": res}
    
    def wiki_search(self, state):
        print("--WIKIPEDIA SEARCH--")
        query = state["constructed_query"]
        route_to_wiki = state["route_to_wiki"] + 1
        docs = self.wikipedia_tool.invoke({"query": query})
        res = Document(page_content=docs)

        return {"documents": [res], "constructed_query": query, "route_to_wiki": route_to_wiki, "source": 'wiki'}

    def decide_to_generate(self, state):

        docs = state["documents"]
        if docs:

            return "generate"
        else:
            return "rewrite"

    def question_router(self, state):

        route_to_retrieve = state["route_to_retrieve"]
        route_to_wiki = state["route_to_wiki"]
        # print(route_to_retrieve, route_to_wiki)

        if route_to_retrieve < 1:
            print("--ROUTING TO RETRIEVER--")
            return "retrieve"
        elif route_to_wiki < 1:
            print("--ROUTING TO WIKI--")
            return "wiki"
        else:
            return "NA"

    def initial_redirection(self, state):
        print("--INITIAL ROUTING--")

        initial_router = self.model.with_structured_output(QuestionRouter)
        initial_router_chain = initial_routing | initial_router

        messages = state["messages"]
        query = state['original_query']

        if len(messages) < 1:
            messages = [query]
        else:
            messages =messages[-1] + [query]

        out = initial_router_chain.invoke({"messages": messages, 'query': query})
        return out.route_to

    def llm(self, state):

        print("--DIRECT GENERATION--")

        prompt_template = ChatPromptTemplate.from_template(
            "You are a helpful RAG assistant that is a part of a system that answers questions related to Art history. Your job is to answer the casual greetings and common inquiries of users. User question: {question}"
        )
        # messages = state["messages"]
        question = state['original_query']
        prompt = prompt_template.invoke({"question": question})
        res = self.model.invoke(prompt)
        print(res)
        return {"generation": res.content}
    
    def na(self, state):
        return {'generation': "I don't know. The input documents doesn't have information on this", "source": 'none'}

    def irrelevant(self, state):
        return {'generation': "The query is irrelevant to art history."}
    
    def create_agent(self):

        workflow = StateGraph(GraphState)

        workflow.add_node("query_construction", self.construct_query)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_docs", self.grade_docs)
        workflow.add_node("generate", self.generate)
        workflow.add_node("save_message", self.save_messages)
        workflow.add_node("rewrite_query", self.rewrite_query)
        workflow.add_node("wiki_search", self.wiki_search)
        workflow.add_node("llm", self.llm)
        workflow.add_node("na", self.na)
        workflow.add_node("irrelevant", self.irrelevant)

        workflow.add_conditional_edges(
            START,
            self.initial_redirection,
            {
                "RAG": "query_construction",
                "LLM": "llm",
                "Irrelevant": "irrelevant"
            }
        )

        workflow.add_edge("query_construction", "retrieve")
        workflow.add_edge("retrieve", "grade_docs")

        workflow.add_conditional_edges(
            'grade_docs',
            self.decide_to_generate,
            {
                "generate": "generate",
                "rewrite": "rewrite_query"
            }
        )

        workflow.add_conditional_edges(
            'generate',
            self.answer_grade,
            {
                "useful": "save_message",
                "not useful": "rewrite_query"
            }
        )
        workflow.add_conditional_edges(
            'rewrite_query',
            self.question_router,
            {
            'retrieve': 'retrieve',
            'wiki': 'wiki_search',
            'NA': 'na'
            }
        )


        workflow.add_edge("llm", "save_message")
        workflow.add_edge("save_message", END)
        workflow.add_edge("na", "save_message")
        workflow.add_edge("irrelevant", "save_message")
        workflow.add_edge("wiki_search", "grade_docs")

        return workflow