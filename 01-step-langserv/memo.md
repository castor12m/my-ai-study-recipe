## 1. 야놀자 한국어 모델 EEVE-Korean-Instruct-10.8B-v1.0-GGUF!

### 1.1 Ollama 이용 설치

https://huggingface.co/heegyu/EEVE-Korean-Instruct-10.8B-v1.0-GGUF
https://www.youtube.com/watch?v=VkcaigvTrug&t=1s
https://github.com/teddylee777/langserve_ollama/blob/main/ollama-modelfile/EEVE-Korean-Instruct-10.8B-v1.0/Modelfile-V02

Modelfile
```
FROM ggml-model-Q5_K_M.gguf

TEMPLATE """{{- if .System }}
<s>{{ .System }}</s>
{{- end }}
<s>Human:
{{ .Prompt }}</s>
<s>Assistant:
"""

SYSTEM """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."""

PARAMETER temperature 0
PARAMETER num_predict 3000
PARAMETER num_ctx 4096
PARAMETER stop <s>
PARAMETER stop </s>
```


```bash
    $ ~/gguf/EEVE-Korean-Instruct-10.8B-v1.0-GGUF
    # 이곳에 Modelfile 과 gguf 파일이 있다고 할때,

    $ cd ~/gguf

    $ ollama create EEVE-Korean-10.8B -f EEVE-Korean-Instruct-10.8B-v1.0-GGUF/Modelfile

    $ ollama list
    >>>
    NAME                            ID              SIZE    MODIFIED      
    EEVE-Korean-10.8B:latest        beaeb072d1d6    7.7 GB  7 seconds ago
    llama2:latest                   fe938a131f40    3.8 GB  3 months ago 
    llama2-uncensored:latest        44040b922233    3.8 GB  3 months ago 

    $ ollama run EEVE-Korean-10.8B
```


### 1.2 Langserve 데모까지 달려보자

```bash
    $ pip install langchain_community
    $ pip install langchain-cli
 
    $ langchain app new my-demo --package vertexai-chuck-norris

    $ tree
    >>>
        .
    └── my-demo
        ├── Dockerfile
        ├── README.md
        ├── app
        │   ├── __init__.py
        │   ├── __pycache__
        │   │   ├── __init__.cpython-310.pyc
        │   │   └── server.cpython-310.pyc
        │   └── server.py
        ├── packages
        │   ├── README.md
        │   └── vertexai-chuck-norris
        │       ├── LICENSE
        │       ├── README.md
        │       ├── poetry.lock
        │       ├── pyproject.toml
        │       ├── tests
        │       │   └── __init__.py
        │       └── vertexai_chuck_norris
        │           ├── __init__.py
        │           ├── __pycache__
        │           │   ├── __init__.cpython-310.pyc
        │           │   └── chain.cpython-310.pyc
        │           └── chain.py
        └── pyproject.toml

```

아래 server.py, chain.py 을 수정

my-demo/app/server.py
```python
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from vertexai_chuck_norris.chain import chain as vertexai_chuck_norris_chain

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Edit this to add the chain you want to add
#add_routes(app, NotImplemented)
add_routes(app, vertexai_chuck_norris_chain, path="/vertexai-chuck-norris")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

```

my-demo/packages/vertexai-chuck-norris/vertexai_chuck_norris/chain.py
```python
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

```

langserve 실행

```bash
    $ cd my-demo

    $ langchain serve
    >>>
    INFO:     Will watch for changes in these directories: ['/home/user/workspace/project-ai/ollama-temp/adv-serv/my-demo']
    INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
    INFO:     Started reloader process [63733] using StatReload
    INFO:     Started server process [63737]
    INFO:     Waiting for application startup.

    __          ___      .__   __.   _______      _______. _______ .______     ____    ____  _______
    |  |        /   \     |  \ |  |  /  _____|    /       ||   ____||   _  \    \   \  /   / |   ____|
    |  |       /  ^  \    |   \|  | |  |  __     |   (----`|  |__   |  |_)  |    \   \/   /  |  |__
    |  |      /  /_\  \   |  . `  | |  | |_ |     \   \    |   __|  |      /      \      /   |   __|
    |  `----./  _____  \  |  |\   | |  |__| | .----)   |   |  |____ |  |\  \----.  \    /    |  |____
    |_______/__/     \__\ |__| \__|  \______| |_______/    |_______|| _| `._____|   \__/     |_______|

    LANGSERVE: Playground for chain "/vertexai-chuck-norris/" is live at:
    LANGSERVE:  │
    LANGSERVE:  └──> /vertexai-chuck-norris/playground/
    LANGSERVE:
    LANGSERVE: See all available routes at /docs/

    INFO:     Application startup complete.

```

아래 주소에 접속

http://127.0.0.1:8000/vertexai-chuck-norris/playground/