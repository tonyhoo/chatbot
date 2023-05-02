import os
from threading import Lock
from typing import Optional, Tuple

import gradio as gr
from langchain import ConversationChain
import typer
from rich import print
from langchain.document_loaders import UnstructuredURLLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from models.openai import Conversation
from models.tools import AutoGluonFAQTools
from langchain.agents import initialize_agent
from langchain.agents import load_tools


app = typer.Typer()

@app.command(name="server")
def server(name: Optional[str] = None, port: Optional[int] = None):
    """Start a gradio server on your local box given the name and port number."""
    prompt = """Imagine you are an expert in AutoML and is targeted to provide users guidance on how to use AutoGluon
        to solve a specific problem. If the question is not relevant to machine learning or AutoGluon, just kindly tell the user that
        you can NOT help and ask the user to ask another question relevant to AutoGluon.
        Your answer should be in English and provide code snippets to help users understand how to use AutoGluon."""

    conv = Conversation(prompt, 5)

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        msg = gr.Textbox("Hi, I am AutoGluon agent, how can I help you?")
        clear = gr.Button("Clear")

        def user(user_message, history):
            return "", history + [[user_message, None]]

        def bot(history):
            bot_message = conv.ask(history[-1][0])[0]
            history[-1][1] = bot_message
            print(bot_message)
            return history

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )
        clear.click(lambda: None, None, chatbot, queue=False)

    demo.launch(share=True, server_name=name, server_port=port)



@app.command(name="local")
def local(question: str = "How to use AutoGluon?"):
    """Test the model locally in CLI"""
    while True:
        question = input("Type in your question:")
        print(agent.run(question))       


def validate_credentials():
    open_ai_key = os.environ.get("OPENAI_API_KEY")
    if open_ai_key is None or open_ai_key == "":
        raise ValueError(
            "Please set the OPENAI_API_KEY environment variable to your OpenAI API key."
        )
    print("Found OpenAI API key.")


if __name__ == '__main__':
    validate_credentials()
    tools = [AutoGluonFAQTools()] + load_tools(["serpapi"])
    agent = initialize_agent(tools, OpenAI(model_name="gpt-3.5-turbo"), agent="zero-shot-react-description", verbose=True)
    app()

