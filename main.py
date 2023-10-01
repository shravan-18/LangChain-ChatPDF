from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain import OpenAI, VectorDBQA
import pinecone

from decouple import Config

config = Config(
    "D:\VIT Material\VIT material\Projects\Langchain Projects\Vector-Databases\.env"
)

pinecone.init(
    environment=config.get("PINECONE_ENVIRONMENT"),
    api_key=config.get("PINECONE_API_KEY"),
)

if __name__ == "__main__":
    print("Hello Vector Store")

    loader = TextLoader(
        r"D:\\VIT Material\\VIT material\\Projects\\Langchain Projects\\Vector-Databases\\medium-blogs\blog-1.txt"
    )
    document = loader.load()
    # print(document)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(document)
    # print(len(texts))

    embeddings = OpenAIEmbeddings(openai_api_key=config.get("OPENAI_API_KEY"))
    decsearch = Pinecone.from_documents(
        texts, embeddings, index_name="medium-blogs-embeddings-test"
    )

    llm = OpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=config.get("OPENAI_API_KEY"),
    )
    qa = VectorDBQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        vectorstore="docsearch",
        return_source_documents=True,
    )

    query = "What is a Vector DB? Explain in 5 lines."
    results = qa({"query": query})
    print(results)
