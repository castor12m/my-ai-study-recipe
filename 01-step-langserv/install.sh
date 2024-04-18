
## 모델 다운로드
#cd model
#cd EEVE-Korean-Instruct-10.8B-v1.0-GGUF
#wget https://huggingface.co/heegyu/EEVE-Korean-Instruct-10.8B-v1.0-GGUF/blob/main/ggml-model-Q5_K_M.gguf

## Ollama 에 모델 올리기
cd ..
ollama create EEVE-Korean-10.8B -f EEVE-Korean-Instruct-10.8B-v1.0-GGUF/Modelfile
ollama list

## venv 설치 및 실행
cd ..
python -m venv venv
. ./venv/bin/activate
pip install -r ./requirements.txt

## 설치 중간에 yes
#langchain app new my-demo --package vertexai-chuck-norris
#cp temp/server.py ./my-demo/app/server.py 
#cp temp/chain.py ./my-demo/packages/vertexai-chuck-norris/vertexai_chuck_norris/chain.py 

#cd my-demo
#langchain serve

#아래 주소에 접속
# http://127.0.0.1:8000/vertexai-chuck-norris/playground/