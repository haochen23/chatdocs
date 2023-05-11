import os
import openai
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI

from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

import pickle

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
    """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
)


def main():
    with open("vectorstore.pkl", "rb") as f:
        global vectorstore
        vectorstore = pickle.load(f)

    llm = AzureChatOpenAI(
        deployment_name="gpt35", openai_api_version="2023-03-15-preview"
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        return_source_documents=True,
        verbose=False,
    )
    chat_history = []
    query = "what is Azure OpenAI Service?"
    result = qa({"question": query, "chat_history": chat_history})

    print("Question:", query)
    print("Answer:", result["answer"])

    chat_history = [(query, result["answer"])]
    query = "Which regions does the service support?"
    result = qa({"question": query, "chat_history": chat_history})

    print("Question:", query)
    print("Answer:", result["answer"])


if __name__ == "__main__":
    load_dotenv()
    main()
