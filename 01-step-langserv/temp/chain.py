#from langchain_community.chat_models import ChatVertexAI
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# prompt = ChatPromptTemplate.from_template(
#     "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. {text}"
# )

prompt = ChatPromptTemplate.from_template(
    "{topic} 에 대하여 간략히 설명해 줘."
)

#_model = ChatVertexAI()
llm = ChatOllama(model = "EEVE-Korean-10.8B:latest")

# if you update this, you MUST also update ../pyproject.toml
# with the new `tool.langserve.export_attr`

#chain = llm
chain = prompt | llm | StrOutputParser()

print(chain.invoke({"topic": "deep learning"}))