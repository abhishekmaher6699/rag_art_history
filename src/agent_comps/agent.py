from langgraph.graph import StateGraph, START, END, MessagesState
from agent_comps.nodes import *


workflow = StateGraph(GraphState)

workflow.add_node("query_construction", construct_query)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_docs", grade_docs)
workflow.add_node("generate", generate)
workflow.add_node("save_message", save_messages)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("wiki_search", wiki_search)
workflow.add_node("llm", llm)
workflow.add_node("na", na)
workflow.add_node("irrelevant", irrelevant)

workflow.add_conditional_edges(
    START,
    initial_redirection,
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
    decide_to_generate,
    {
        "generate": "generate",
        "rewrite": "rewrite_query"
    }
)

workflow.add_conditional_edges(
    'generate',
    answer_grade,
    {
        "useful": "save_message",
        "not useful": "rewrite_query"
    }
)
workflow.add_conditional_edges(
    'rewrite_query',
    question_router,
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