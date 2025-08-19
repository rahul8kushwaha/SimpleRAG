import pdfplumber
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
class RAG:
    def __init__(self):
        pass
    def pdf_parsing(self,PATH,start=1,end=None):
        # Open the PDF
        doc=[]
        with pdfplumber.open(PATH) as pdf:
            # Loop through pages
            if end is None:
                end=len(pdf.pages)
            for page_num, page in enumerate(pdf.pages[start:]):
                text = page.extract_text()
                doc.append(text)
        corrected_doc=[]
        for line in doc:
            corrected_doc.extend(line.split('\n'))
        corrected_doc=' '.join(corrected_doc)
        return corrected_doc
    def parsing(self,PATHS):
        docs=[]
        for PATH in PATHS:
            docs.append(self.pdf_parsing(PATH))
        return docs
    def chunking(self,docs):
        token_size=800
        overlap=50
        chunks=[]
        for doc in docs:
            doc=doc.split()
            for i in range(0,len(doc),token_size):
                chunks.append(' '.join(doc[i:i+token_size+overlap]))
        return chunks
    def setup_retriever(self,PATHS):
        docs=self.parsing(PATHS)
        chunks=self.chunking(docs)
        return chunks
    def generate_embedding_vector_db(self,PATH):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        persist_directory = PATH
        self.db = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
    
if __name__=='__main__':
    PATHS=[...]
    PATH=...
    rag=RAG()
    chunks=rag.setup_retriever(PATHS)
    rag.generate_embedding_vector_db(PATH)
    query='Tell me about karma yoga'
    t=rag.db.search(query=query,search_type='similarity_score_threshold')
    t=' '.join(map(lambda a:a.page_content,t))
    llm = Ollama(model='phi3')
    template=f"""Using the given context:{t}

    Answer this question: {query}"""
    response = llm.invoke(template)
    print(response)
