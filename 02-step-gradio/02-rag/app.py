from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

def DoLoadModel():
    model = ChatOllama(model="EEVE-Korean-10.8B:latest", temperature=0)
    return model

def DoSimpleQuery(model, quary):
    response = model.invoke(quary)
    print(response)

def DoLoad(fileName):
    loader = PyPDFLoader(fileName, extract_images=False)
    pages = loader.load()
    return pages


def DoEmbedding(pages):
    # 문서를 문장으로 분리
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )

    docs = text_splitter.split_documents(pages)

    # 문장을 임베딩으로 변환하고 벡터 저장소에 저장
    embeddings = HuggingFaceEmbeddings(
        model_name='jhgan/ko-sroberta-nli',
        model_kwargs={'device':'cpu'},
        encode_kwargs={'normalize_embeddings':True},
    )

    vectorstore = Chroma.from_documents(docs, embeddings)
    return (vectorstore, docs)

def GetPrompt():
    # Prompt
    template = '''Answer the question based only on the following context:
    {context}

    Question: {question}
    '''

    prompt = ChatPromptTemplate.from_template(template)
    return prompt

def DoFormatDocs(docs):
    return '\n\n'.join([d.page_content for d in docs])

def DoQuestionWithRAGChain(model, retriever, prompt, docs, query):
    # RAG Chain 연결
    rag_chain = (
        {'context': retriever | DoFormatDocs, 'question': RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    response = rag_chain.invoke(query)
    print(response)

if __name__ == "__main__":
    
    model = DoLoadModel()
        
    # 검색 쿼리
    query = "겨울철에 내한성이 강한 나무에는 어떤 것이 있을까요?"
    
    print('-------------------------------------------')
    DoSimpleQuery(model, query)
    
    #pages = DoLoad("./02-rag/경남교육청_수목관리_업무매뉴얼.pdf")
    pages = DoLoad("./02-rag/manual.pdf")
    pages[0].page_content
    vectorstore, docs1 = DoEmbedding(pages)

    # 가장 유사도가 높은 문장을 하나만 추출
    retriever = vectorstore.as_retriever(search_kwargs={'k': 5})

    docs2 = retriever.get_relevant_documents(query)
    print('-------------------------------------------')
    print(f'가장 유사도가 높은 문장을 하나만 추출:')
    #print(len(docs2))
    print(docs2[0].page_content)
    print(docs2[0].metadata)
    
    print('-------------------------------------------')
    print(docs2[1].page_content)
    print(docs2[1].metadata)
    
    print('-------------------------------------------')
    prompt = GetPrompt()
    DoQuestionWithRAGChain(model, retriever, prompt, docs1, query)
