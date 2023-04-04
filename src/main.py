import os
from typing import Optional

import gradio as gr
import typer
from rich import print

from models.openai import Conversation

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
    prompt = """Imagine you are an expert in AutoML and is targeted to provide users guidance on how to use AutoGluon
           to solve a specific problem. If the question is not relevant to machine learning or AutoGluon, just kindly tell the user that
           you can NOT help and ask the user to ask another question relevant to AutoGluon.
           Your answer should be in English and provide code snippets to help users understand how to use AutoGluon."""

    conv = Conversation(prompt, 5)
    while True:
        question = input("Type in your question:")
        print(conv.ask(question)[0])


def validate_credentials():
    open_ai_key = os.environ.get("OPENAI_API_KEY")
    if open_ai_key is None or open_ai_key == "":
        raise ValueError(
            "Please set the OPENAI_API_KEY environment variable to your OpenAI API key."
        )
    print("Found OpenAI API key.")


if __name__ == '__main__':
    validate_credentials()
    app()

