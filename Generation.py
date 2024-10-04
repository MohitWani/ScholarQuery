from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from Retrieval import retrieval
template = """Answer the question of user by using the context:

            {context}

            Question: {question}
            """
llm = ChatGroq(groq_api_key="",
                model_name="mixtral-8x7b-32768"
                )

prompt = ChatPromptTemplate(template)

chain = (
            {"context": retrieval, "question":RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser
        )

