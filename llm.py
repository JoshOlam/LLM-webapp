# from langchain import PromptTemplate
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
# from langchain.llms import ctransformers
from langchain.chains import RetrievalQA
import chainlit as cl
from langchain.llms.ctransformers import CTransformers

import time

DB_FAISS_PATH = 'vectorstores/db_faiss/'

custom_prompt_template = """Use the following piece of information to answer the user's question.
If you don't know the answer, please just say that you don't know the answer.Don't make up an answer.

Context:{context}
question:{question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vector store
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables = ['context', 'question'])

    return prompt

def load_llm():
    llm = CTransformers(
        model = 'llama-2-7b-chat.ggmlv3.q8_0.bin',
        model_type = 'llama',
        # max_new_tokens = 512,
        temperature = 0.5
    )
    return llm

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever = db.as_retriever(search_kwargs={'k':2}),
        return_source_documents = True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

def qa_bot():
    embeddings = HuggingFaceEmbeddings(
        model_name = 'sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs = {'device': 'cpu'}
    )
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

def final_result(query):
    qa_result = qa_bot()
    response = qa_result({
        'query': query
    })

    while True:
        user_query = input('\nEnter your query here')
        if user_query == 'exit' or 'quit':
            break
        if user_query.strip() == '':
            continue

        # Get the answer from the model
        # start_time = time.time()
        # llm_response = pass

    return response



#>>> Chainlit webapp <<<#
@cl.on_chat_start
async def start():
    # def main():
    # prompt=PromptTemplate(template=template, input_variables=["question"])
    # llm_chain = LLMChain(prompt=prompt, llm=OpenAI(temperature=0), verbose=True)

    # cl.user_session.set("llm_chain", llm_chain)
    chain = qa_bot()
    msg = cl.Message(content='LLM Starting...')
    await msg.send()
    msg.content = "Hello, welcome to the ComX chat buddy. What can I do for you today?"
    await msg.update()
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=['FINAL', 'ANSWER']
    )
    cb.answer_reached=True
    res = await chain.acall(message, callbacks=[cb])
    answer = res['result']
    sources = res['source_documents']

    if sources:
        answer += f'\nSources: ' + str(str(sources))
    else:
        answer += f'\nNo source found!'

    await cl.Message(content=answer).send()
