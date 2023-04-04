import os

import gradio as gr
import openai

from models.openai import Conversation

if __name__ == '__main__':
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    print(f"the key is {os.environ.get('OPENAI_API_KEY')}")

    prompt = """Imagine you are an expert in AutoML and is targeted to provide users guidance on how to use AutoGluon
    to solve a specific problem. If the question is not relevant to machine learning or AutoGluon, just kindly tell the user that
    you can NOT help and ask the user to ask another question relevant to AutoGluon.
    Your answer should be in English and provide code snippets to help users understand how to use AutoGluon."""

    conv = Conversation(prompt, 5)

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
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

    demo.launch()
