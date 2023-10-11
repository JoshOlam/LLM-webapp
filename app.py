# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain.chat_models import ChatOpenAI
# import chainlit as cl

# template = '''Question: {question}
# Answer: Let's think step-by-step.'''

# @cl.AsyncLangchainCallbackHandler()
# def factory():
#     prompt = PromptTemplate(template=template, input_variables=['question'])
#     llm_chain = LLMChain(prompt=prompt, llm=ChatOpenAI(temperature=0, streaming=True))
#     return llm_chain

# import chainlit as cl
# from chainlit.prompt import Prompt, PromptMessage
# from chainlit.playground.providers.openai import ChatOpenAI

# import openai
# import os

# openai.api_key = "YOUR_OPEN_AI_API_KEY"


# @cl.on_message
# async def main(message: str):
#     # Create the prompt object for the Prompt Playground
#     prompt = Prompt(
#         provider=ChatOpenAI.id,
#         messages=[
#             PromptMessage(
#                 role="user",
#                 template=template,
#                 formatted=template.format(input=message)
#             )
#         ],
#         settings=settings,
#         inputs={"input": message},
#     )

#     # Prepare the message for streaming
#     msg = cl.Message(
#         content="",
#         language="sql",
#     )

#     # Call OpenAI
#     async for stream_resp in await openai.ChatCompletion.acreate(
#         messages=[m.to_openai() for m in prompt.messages], stream=True, **settings
#     ):
#         token = stream_resp.choices[0]["delta"].get("content", "")
#         await msg.stream_token(token)

#     # Update the prompt object with the completion
#     prompt.completion = msg.content
#     msg.prompt = prompt

#     # Send and close the message stream
#     await msg.send()

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
# Step 1: Import the necessary modules
import os
 
# Step 2: Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"
import chainlit as cl

template = """Question: {question}

Answer: Let's think step by step."""

@cl.on_chat_start
def main():
    prompt=PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=OpenAI(temperature=0), verbose=True)

    cl.user_session.set("llm_chain", llm_chain)

@cl.on_message
async def main(message: str):

    llm_chain = cl.user_session.get("llm_chain")

    res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])

    await cl.Message(
        content=res["text"]
        # content=f"Bot: {message}",
    ).send()

