from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools import BaseTool
from langchain.vectorstores import Chroma


import os


class AutoGluonFAQTools(BaseTool):
    name = "faq"
    description = "useful when you need to answer questions about AutoGluon with latest API documentations, tutorials, and code examples."
    doc_urls_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../data/doc_urls.txt"
    )
    qa_chain: ConversationalRetrievalChain = None

    def get_chat_history(self, inputs) -> str:
        res = []
        for human, ai in inputs:
            res.append(f"Human:{human}\nAI:{ai}")
        return "\n".join(res)

    def setup_qa_chain(self):
        urls = []
        with open(self.doc_urls_path, "r") as file:
            urls.extend([line.rstrip() for line in file])
        loader = UnstructuredURLLoader(urls=urls)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=2000, separator="\n")
        texts = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        # create the vectorestore to use as the index
        db = Chroma.from_documents(texts, embeddings)
        # expose this index in a retriever interface
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        qa = ConversationalRetrievalChain.from_llm(
            llm=OpenAI(),
            retriever=retriever,
            memory=memory,
            verbose=True,
            qa_prompt=QA_PROMPT,
            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
            get_chat_history=self.get_chat_history,
        )
        return qa

    def _run(self, query: str, run_manager=None) -> str:
        """Use the tool."""
        if self.qa_chain is None:
            self.qa_chain = self.setup_qa_chain()
        return self.qa_chain({"question": query})

    async def _arun(self, query: str, run_manager=None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("faq does not support async")