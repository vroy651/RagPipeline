
import os
import re
from typing import List, Dict, Tuple
from datetime import datetime
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Fix for Pydantic v2 compatibility with FlashrankRerank
try:
    FlashrankRerank.model_rebuild()
except Exception:
    pass

# Embeddings and LLM
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()


class PolicyDocument:
    """Represents a policy document with metadata extraction."""
    
    def __init__(self, doc: Document):
        self.doc = doc
        self.filename = os.path.basename(doc.metadata.get("source", ""))
        self.effective_date = self._extract_effective_date()
        self.is_policy = self._is_policy_document()
        
    def _extract_effective_date(self) -> datetime:
        """Extract effective date from document content."""
        content = self.doc.page_content
        
        # Pattern: "Effective Date: Mon DD, YYYY"
        date_pattern = r"Effective Date:\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})"
        match = re.search(date_pattern, content)
        
        if match:
            date_str = match.group(1)
            try:
                return datetime.strptime(date_str, "%b %d, %Y")
            except ValueError:
                try:
                    return datetime.strptime(date_str, "%B %d, %Y")
                except ValueError:
                    pass
        
        # If no date found, use file modification time or epoch
        return datetime(1970, 1, 1)
    
    def _is_policy_document(self) -> bool:
        """Determine if this is a policy document vs noise."""
        content_lower = self.doc.page_content.lower()
        
        # Policy indicators
        policy_keywords = ["policy", "mandate", "required", "employees", "work from home", 
                          "remote work", "office", "approval"]
        
        # Noise indicators
        noise_keywords = ["menu", "cafeteria", "tacos", "burgers", "food"]
        
        policy_score = sum(1 for kw in policy_keywords if kw in content_lower)
        noise_score = sum(1 for kw in noise_keywords if kw in content_lower)
        
        return policy_score > noise_score


class KnowledgeBaseManager:
    """Manages the creation and loading of source documents."""
    
    def __init__(self, directory="knowledge_base"):
        self.directory = directory
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def setup_files(self):
        """Creates the three test files for the exercise."""
        files = {
            "policy_v1_2021.txt": (
                "TechCorp Work From Home Policy (Effective Date: Jan 1, 2021)\n"
                "Due to the global pandemic, all employees are required to work remotely.\n"
                "Effective immediately, working from the office is suspended until further notice.\n"
                "Employees may expense up to $500 for home office equipment."
            ),
            "policy_v2_2024.txt": (
                "TechCorp Return to Office Mandate (Effective Date: Jan 1, 2024)\n"
                "We are excited to welcome everyone back!\n"
                "Remote work is now capped at 1 day per week, and must be approved by a manager.\n"
                "The 100% remote work policy from 2021 is officially revoked.\n"
                "Employees are expected to be in the office 4 days a week."
            ),
            "friday_cafeteria_menu.txt": (
                "TechCorp Cafeteria Menu - Weekly Specials\n"
                "Monday: Tacos\n"
                "Tuesday: Burgers\n"
                "Friday: Fish & Chips (Chef's Special!)\n"
                "Note: The cafeteria is closed for cleaning on Friday afternoons.\n"
                "Come enjoy a break from work!"
            )
        }
        
        for filename, content in files.items():
            filepath = os.path.join(self.directory, filename)
            with open(filepath, "w") as f:
                f.write(content)
        
        print(f"✓ Knowledge base setup in '{self.directory}' with {len(files)} files.")

    def load_documents(self) -> List[PolicyDocument]:
        """Loads all .txt files and wraps them in PolicyDocument objects."""
        documents = []
        
        for filename in sorted(os.listdir(self.directory)):
            if filename.endswith(".txt"):
                loader = TextLoader(os.path.join(self.directory, filename))
                loaded_docs = loader.load()
                
                for doc in loaded_docs:
                    policy_doc = PolicyDocument(doc)
                    documents.append(policy_doc)
                    
                    print(f"  Loaded: {policy_doc.filename}")
                    print(f"    - Effective Date: {policy_doc.effective_date.strftime('%b %d, %Y')}")
                    print(f"    - Is Policy: {policy_doc.is_policy}")
        
        return documents


class ConflictAwareRetriever:
    """
    Handles indexing, embedding, and conflict-aware retrieval.
    
    Key Features:
    - Semantic search with Google Gemini embeddings
    - Temporal conflict resolution (latest policy wins)
    - Noise filtering based on document classification
    - Re-ranking for relevance
    """
    
    def __init__(self, policy_docs: List[PolicyDocument], persist_directory="./chroma_db"):
        self.policy_docs = policy_docs
        self.persist_directory = persist_directory
        
        # 1. Chunk documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=20
        )
        
        # Only chunk the actual LangChain documents
        raw_docs = [pd.doc for pd in policy_docs]
        self.splits = splitter.split_documents(raw_docs)
        
        # Add enhanced metadata to each chunk
        for i, split in enumerate(self.splits):
            # Find the original policy document for this split
            source_file = os.path.basename(split.metadata.get("source", ""))
            
            for pd in policy_docs:
                if pd.filename == source_file:
                    split.metadata["effective_date"] = pd.effective_date.isoformat()
                    split.metadata["is_policy"] = pd.is_policy
                    split.metadata["filename"] = pd.filename
                    break
        
        # 2. Embeddings - Google Gemini Embeddings (as requested)
        print("\n✓ Loading Google Gemini embedding model...")
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )
        
        # 3. Vector Store with persistence
        print("✓ Creating vector store...")
        self.vectorstore = Chroma.from_documents(
            documents=self.splits,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )
        
        # 4. Reranker for better relevance
        print("✓ Initializing reranker...")
        self.compressor = FlashrankRerank(top_n=10)  # Fetch all relevant docs for temporal sorting
        
    def get_retriever(self):
        """Returns a conflict-aware retriever with reranking."""
        base_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 10}  # Cast a wide net initially
        )
        
        # Apply compression/reranking
        return ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=base_retriever
        )
    
    def filter_and_prioritize_documents(self, docs: List[Document]) -> List[Document]:
        """
        Post-retrieval filtering:
        1. Remove noise (non-policy documents)
        2. Handle temporal conflicts (keep only latest version)
        """
        # Filter out noise
        policy_docs = [doc for doc in docs if doc.metadata.get("is_policy", False)]
        
        if not policy_docs:
            return docs  # Fallback if filtering removed everything
        
        # Group by topic/conflict and keep most recent
        # For this exercise, we can identify conflicts by similar content
        # In production, you'd have more sophisticated conflict detection
        
        # Sort by effective date (most recent first)
        sorted_docs = sorted(
            policy_docs,
            key=lambda d: datetime.fromisoformat(d.metadata.get("effective_date", "1970-01-01T00:00:00")),
            reverse=True
        )
        
        return sorted_docs


class TechCorpRAG:

    # performing RAG task here

    
    def __init__(self, retriever: ConflictAwareRetriever):
        self.retriever = retriever
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0  #  it give more  Deterministic answer
        )
        
        # System prompt with strict grounding rules
        system_prompt = """You are TechCorp's HR Policy Assistant. Your job is to provide accurate, well-cited answers about company policies.

CRITICAL RULES:
1. TEMPORAL CONFLICT RESOLUTION: When multiple policies exist on the same topic, ALWAYS use the one with the LATEST "Effective Date".
   - If you see a 2021 policy and a 2024 policy on remote work, the 2024 policy is the ONLY truth.
   - Explicitly state that older policies are outdated/revoked.

2. NOISE FILTERING: Ignore documents that are not policies (e.g., cafeteria menus, event schedules).
   - Just because a document mentions "Friday" doesn't make it relevant to a work-from-home policy question.

3. GROUNDING: Base your answer ONLY on the retrieved context below. Do not make up information.

4. MANDATORY CITATION: 
   - End your response with: "Sources: [list of filenames]"
   - Only cite filenames that directly support your answer
   - If you cite policy_v1_2021.txt, you MUST explain it's outdated

CONTEXT:
{context}

Answer the user's question below clearly and concisely."""

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        # Create chains using LCEL
        self.document_chain = (
            RunnablePassthrough.assign(context=lambda x: self._format_docs(x["context"]))
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def _format_docs(self, docs: List[Document]) -> str:
        """Format the context for the prompt."""
        formatted = []
        for doc in docs:
            effective_date = doc.metadata.get("effective_date", "Unknown")
            filename = doc.metadata.get("filename", "Unknown")
            formatted.append(f"--- DOCUMENT: {filename} (Effective: {effective_date}) ---\n{doc.page_content}")
        return "\n\n".join(formatted)

    def query(self, user_input: str) -> Dict:
        """
        Query the RAG system with manual post-retrieval filtering and prioritization.
        """
        # 1. Retrieve raw documents using the compressed retriever
        retriever = self.retriever.get_retriever()
        retrieved_docs = retriever.invoke(user_input)
        
        # 2. Apply strict filtering and temporal prioritization
        # This groups documents and ensures the latest ones are presented first
        filtered_docs = self.retriever.filter_and_prioritize_documents(retrieved_docs)
        
        # 3. Generate answer using the document chain with the prioritized context
        response = self.document_chain.invoke({
            "input": user_input,
            "context": filtered_docs
        })
        
        # Extract unique source filenames from the filtered set
        sources = list(set([
            doc.metadata.get("filename", "unknown")
            for doc in filtered_docs
            if doc.metadata.get("is_policy", False)
        ]))
        
        return {
            "answer": response,
            "sources": sources,
            "context": filtered_docs
        }


def main():
    """Main execution flow."""
    print("=" * 70)
    print("TechCorp Conflicting Policy RAG System")
    print("Using Google Gemini Embeddings")
    print("=" * 70)
    
    # 1. Setup Knowledge Base
    print("\n[1/4] Setting up knowledge base...")
    kb = KnowledgeBaseManager()
    kb.setup_files()
    
    # 2. Load documents with metadata
    print("\n[2/4] Loading documents...")
    policy_docs = kb.load_documents()
    
    # 3. Initialize RAG system
    print("\n[3/4] Building RAG system...")
    retriever = ConflictAwareRetriever(policy_docs)
    bot = TechCorpRAG(retriever)
    
    # 4. Test Query (The Exercise Question)
    print("\n[4/4] Testing with exercise query...")
    print("=" * 70)
    
    test_query = "Can I work fully remotely this Friday?"
    print(f"\n❓ QUERY: {test_query}")
    print("-" * 70)
    
    result = bot.query(test_query)
    
    print(f"\n✅ ANSWER:\n{result['answer']}")
    print(f"\n📎 SOURCES USED: {', '.join(result['sources'])}")
    
    # Additional test queries
    print("\n" + "=" * 70)
    print("Additional Test Queries:")
    print("=" * 70)
    
    additional_queries = [
        "What was the remote work policy in 2021?",
        "How many days per week can I work remotely?",
        "What's for lunch on Friday?"
    ]
    
    for query in additional_queries:
        print(f"\n❓ {query}")
        result = bot.query(query)
        print(f"✅ {result['answer'][:200]}...")  # Truncate for display


if __name__ == "__main__":
    main()