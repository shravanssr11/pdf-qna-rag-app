import streamlit as st
import fitz  # PyMuPDF
with st.sidebar:
    file=st.file_uploader("upload your doc",type="pdf")

query=st.text_input("enter your prompt")

# extracting text from uploaded doc

def extract_text_from_pdf(file_bytes):
    text = ""
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text
if file is not None:
    bytes_data =file.read()

    text = extract_text_from_pdf(bytes_data)

# setting up google api key

    import os
    from dotenv import load_dotenv
    load_dotenv()
    os.environ['HF_TOKEN']=os.getenv('HF_TOKEN')
    groq_api_key=os.getenv('GROQ_API_KEY')

# splitting doc

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])

# create embeddings
    import asyncio
    try:
       asyncio.get_running_loop()
    except RuntimeError:
      asyncio.set_event_loop(asyncio.new_event_loop())

    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    
# create a vector store

    from langchain_community.vectorstores import FAISS
    db=FAISS.from_documents(docs,embeddings)

# creating chat prompt template

    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_template(
    """"
    Answer the following question based on provided context.
    <context>
    {context}
    </context>
    Question:{input}
    """)
# define llm

    from langchain_groq import ChatGroq
    llm=ChatGroq(model="gemma2-9b-it",groq_api_key=groq_api_key,max_tokens=500)
    retriver=db.as_retriever()

# setting up chain

    from langchain.chains.combine_documents import create_stuff_documents_chain
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    from langchain.chains import create_retrieval_chain
    retrieval_chain=create_retrieval_chain(retriver,combine_docs_chain)

#get response from llm

    response=retrieval_chain.invoke({"input":query})

#displaying result

    result=response['answer']
    st.write(result)
    with st.expander('Document similarity result'):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
                 


