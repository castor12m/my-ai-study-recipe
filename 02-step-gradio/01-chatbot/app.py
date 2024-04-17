# Ollama 모델 로드 및 테스트
from langchain_community.chat_models import ChatOllama
import gradio as gr


def echo(message, history):
    response = model.invoke(message)
    return response.content

if __name__ == "__main__":
    model = ChatOllama(model="EEVE-Korean-10.8B:latest", temperature=0)
    
    demo = gr.ChatInterface(fn=echo, title="EEVE Bot")
    demo.launch()