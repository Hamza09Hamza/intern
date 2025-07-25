import logging
from typing import List, Optional
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from typing_extensions import TypedDict
from langgraph.graph import START, StateGraph
import faiss

from config import Config

logger = logging.getLogger(__name__)

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

class RAGService:
    def __init__(self):
        self.llm: Optional[OllamaLLM] = None
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.vector_store: Optional[FAISS] = None
        self.rag_graph = None
        self.prompt: Optional[ChatPromptTemplate] = None
        self.is_initialized = False
        self.document_count = 0
        self.language = Config.LANGUAGE
    
    async def initialize(self):
        logger.info("Initializing RAG system...")
        
        try:
            await self._initialize_llm()
            await self._initialize_embeddings()
            await self._initialize_vector_store()
            await self._load_documents()
            await self._initialize_rag_graph()
            
            self.is_initialized = True
            logger.info("RAG system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise
    
    async def _initialize_llm(self):
        logger.info("Initializing LLM...")
        
        try:
            self.llm = OllamaLLM(model=Config.LLM_MODEL_NAME)
            
            # Test the model
            test_response = self.llm.invoke("Bonjour")
            logger.info(f"LLM test successful: {test_response[:50]}...")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    async def _initialize_embeddings(self):
        logger.info("Initializing embeddings model...")
        
        try:
            model_kwargs = {'device': Config.EMBEDDINGS_DEVICE}
            encode_kwargs = {'normalize_embeddings': Config.EMBEDDINGS_NORMALIZE}
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=Config.EMBEDDINGS_MODEL_NAME,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            
            test_embed = await self.embeddings.aembed_query("test")
            logger.info(f"Embeddings test successful, dimension: {len(test_embed)}")
            
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise
    
    async def _initialize_vector_store(self):
        logger.info("Initializing vector store...")
        
        try:
            test_embed = await self.embeddings.aembed_query("test")
            embedding_dim = len(test_embed)
            
            index = faiss.IndexFlatL2(embedding_dim)
            
            self.vector_store = FAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )
            
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    async def _load_documents(self):
        logger.info("Loading documents...")
        
        try:
            Config.ensure_directories()
            
            loader = DirectoryLoader(
                Config.DATA_DIRECTORY, 
                glob="**/*.txt", 
                show_progress=True
            )
            
            docs = loader.load()
            
            if not docs:
                logger.warning(f"No documents found in {Config.DATA_DIRECTORY}")
                return
            
            logger.info(f"Loaded {len(docs)} documents")
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP,
                add_start_index=True,
            )
            
            all_splits = text_splitter.split_documents(docs)
            logger.info(f"Created {len(all_splits)} document chunks")
            
            if all_splits:
                document_ids = self.vector_store.add_documents(documents=all_splits)
                self.document_count = len(document_ids)
                logger.info(f"Added {self.document_count} documents to vector store")
            
        except Exception as e:
            logger.error(f"Failed to load documents: {e}")
            raise
    
    async def _initialize_rag_graph(self):
        logger.info("Initializing RAG graph...")
        
        try:
            # Create prompt template
            self.prompt = ChatPromptTemplate.from_template(Config.RAG_PROMPT_TEMPLATE)
            
            def retrieve(state: State):
                """Retrieve relevant documents"""
                if self.vector_store is None:
                    logger.warning("Vector store not available")
                    return {"context": []}
                
                try:
                    retrieved_docs = self.vector_store.similarity_search(
                        state["question"], 
                        k=Config.VECTOR_STORE_K
                    )
                    logger.debug(f"Retrieved {len(retrieved_docs)} documents")
                    return {"context": retrieved_docs}
                    
                except Exception as e:
                    logger.error(f"Error in retrieve: {e}")
                    return {"context": []}
            
            def generate(state: State):
                """Generate response using retrieved context"""
                try:
                    # Prepare context
                    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
                    
                    if not docs_content.strip():
                        docs_content = "Aucun contexte pertinent trouvé."
                    
                    # Generate response
                    messages = self.prompt.invoke({
                        "question": state["question"], 
                        "context": docs_content
                    })
                    
                    response = self.llm.invoke(messages)
                    logger.debug(f"Generated response: {response[:100]}...")
                    
                    return {"answer": response}
                    
                except Exception as e:
                    logger.error(f"Error in generate: {e}")
                    return {"answer": "Désolé, une erreur s'est produite lors de la génération de la réponse."}
            
            # Build the graph
            graph_builder = StateGraph(State).add_sequence([retrieve, generate])
            graph_builder.add_edge(START, "retrieve")
            self.rag_graph = graph_builder.compile()
            
            logger.info("RAG graph initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG graph: {e}")
            raise
    
    async def process_question(self, question: str) -> str:
        if not self.is_initialized:
            return "Le système RAG n'est pas initialisé."
        
        if not self.rag_graph:
            return "Le graphique RAG n'est pas disponible."
        
        try:
            logger.info(f"Processing question: '{question}'")
            
            result = self.rag_graph.invoke({"question": question})
            answer = result.get("answer", "Aucune réponse générée.")
            
            logger.info(f"Answer generated successfully")
            return answer
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return "Désolé, une erreur s'est produite lors du traitement de votre question."
    
    def get_status(self) -> dict:
        return {
            "initialized": self.is_initialized,
            "llm_available": self.llm is not None,
            "embeddings_available": self.embeddings is not None,
            "vector_store_available": self.vector_store is not None,
            "rag_graph_available": self.rag_graph is not None,
            "document_count": self.document_count,
            "data_directory": Config.DATA_DIRECTORY
        }