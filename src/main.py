from typing import Optional

import gradio as gr
import typer
from rich import print

from models.falcon import FalconConversation



app = typer.Typer()

@app.command(name="server")
def server(name: Optional[str] = None, port: Optional[int] = None):
    """Start a gradio server on your local box given the name and port number."""
    prompt = """### Human: """

    conv = FalconConversation(prompt)

    with gr.Blocks() as demo:
        msg = gr.Textbox("How can foreign patients enter the UAE for treatment?")
        clear = gr.Button("Clear")
        chatbot_sft = gr.Chatbot(label="falcon-7b-aws-qapairs-67")
        
        
        chatbot_base = gr.Chatbot(label="falcon-7b")
        chatbot_instruct = gr.Chatbot(label="falcon-7b-instruct")

        def user(user_message, history):
            return "", history + [[user_message, None]]

        def bot_sft(history):
            bot_message = conv.ask(question=history[-1][0], model_name="sft")
            history[-1][1] = bot_message
            print(bot_message)
            return history
        
        def bot_base(history):
            bot_message = conv.ask(question=history[-1][0], model_name="base")
            history[-1][1] = bot_message
            print(bot_message)
            return history
        
        def bot_instruct(history):
            bot_message = conv.ask(question=history[-1][0], model_name="instruct")
            history[-1][1] = bot_message
            print(bot_message)
            return history

        msg.submit(user, [msg, chatbot_sft], [msg, chatbot_sft], queue=False).then(
            bot_sft, chatbot_sft, chatbot_sft
        )
        msg.submit(user, [msg, chatbot_base], [msg, chatbot_base], queue=False).then(
            bot_base, chatbot_base, chatbot_base
        )
        msg.submit(user, [msg, chatbot_instruct], [msg, chatbot_instruct], queue=False).then(
            bot_instruct, chatbot_instruct, chatbot_instruct
        )
        clear.click(lambda: None, None, chatbot_sft, queue=False)
        clear.click(lambda: None, None, chatbot_base, queue=False)
        clear.click(lambda: None, None, chatbot_instruct, queue=False)
        

    demo.launch(share=True, server_name=name, server_port=port)   


if __name__ == '__main__':
    app()

