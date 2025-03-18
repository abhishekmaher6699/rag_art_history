from langchain_core.prompts import ChatPromptTemplate

query_construction_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', """Rewrite the query to improve retrieval. Use the previous context to make the current query more detailed and context-aware if it is related to the same topic.
        Examples:
        - If the conversation is about Van Gogh, and the user asks, 'Tell me his famous paintings,' rewrite it as 'Tell me Van Gogh's famous paintings.'
        - If the context is about the Renaissance, and the user asks, 'What were its main features?' rewrite it as 'What were the main features of the Renaissance?'
        - If the discussion is about Picasso and the user asks, 'Where was he born?' rewrite it as 'Where was Picasso born?'
        - If the context is about Impressionism, and the user asks, 'Who were the key figures?' rewrite it as 'Who were the key figures of Impressionism?'
        - If the conversation is about Leonardo da Vinci, and the user asks, 'What is his most famous work?' rewrite it as 'What is Leonardo da Vinci's most famous work?'
        - If the discussion is about abstract art, and the user asks, 'When did it begin?' rewrite it as 'When did abstract art begin?'
        - If the context is about the Baroque period, and the user asks, 'Name some artists,' rewrite it as 'Name some artists from the Baroque period.'
        - If the discussion is about Frida Kahlo, and the user asks, 'What inspired her work?' rewrite it as 'What inspired Frida Kahlo's work?'

        However, if the current query is unrelated to the context or does not benefit from it, simply rephrase it to correct any grammatical or syntactical errors while keeping the original meaning intact.
        Return only the improved query and nothing else. Be concise, precise, and context-aware."""),
        ('human', "Current Query: {query}, Message History: {history}"),
    ]
)


grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
                        You are a grader assessing relevance of a retrieved document to a user question. \n
                        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
                        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
                        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
        """
),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)



rag_prompt = ChatPromptTemplate.from_template(
    """
    Answer the following question based only on the documents and context below.
    If you dont know the answer, just say that you dont know.
    Documents: {documents}
    Context: {context}
    Question: {input}
    Answer:
    """
)



answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a grader assessing whether an answer addresses / resolves a question \n
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question. Don't be too stringent. """
),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)



re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You a question re-writer that converts an input question to a better version that is optimized \n
                        for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning. Just return the improved query"""
),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)




initial_routing = ChatPromptTemplate.from_messages(
    [
        ("system", """
                    You are a helpful assistant assisting a RAG agent in routing user queries. The RAG agent has data about art history from a textbook.
                    You are provided with a series of messages, and the last message in the sequence is the user's current query.

                    Your task is to determine the appropriate routing for the CURRENT QUERY based on the following rules:
                    1. If the query is about art history or related topics, route it to the RAG agent for retrieving data from the vector database.
                    2. If the query is a casual greeting (e.g., "Hello," "How are you?"), route it to a standard LLM for handling.
                    3. Any query that is not related to art history or a casual greeting is considered irrelevant and should not be processed further.

                    Output your response as one of the following:
                    - "RAG" if the query should be routed to the RAG agent.
                    - "LLM" if the query should be handled by the standard language model.
                    - "Irrelevant" if the query is outside the allowed scope.

                    Only art-related queries or casual greetings are allowed. Ensure strict adherence to these rules.
                    """
         ),
        (
            "human",
            "Messages: {messages}, current query: {query}.",
        ),
    ]
)

