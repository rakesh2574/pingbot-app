import json
import os
# FIX: Updated imports for modern LangChain (v0.2/v0.3)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from config import SYSTEM_PROMPT, BLOCKED_INTENTS, LLM_MODEL


class PingbhaiRAG:
    def __init__(self, api_key=None, kb_path="knowledge_base.json", history_file="temp_history.json"):
        self.kb_path = kb_path
        self.history_file = history_file

        # LOGIC: Use provided key -> OR -> Use Environment/Secrets key
        self.api_key = api_key if api_key else os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("No OpenAI API Key provided. Please enter one or set it in Secrets.")

        self.embeddings = OpenAIEmbeddings(api_key=self.api_key)
        self.vector_store = None
        self.chain = None
        self.model = ChatOpenAI(
            model_name=LLM_MODEL,
            temperature=0.3,
            api_key=self.api_key
        )

    # --- MEMORY MANAGEMENT ---
    def _load_history(self):
        if not os.path.exists(self.history_file):
            return ""
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
            conversation_text = ""
            for turn in history:
                conversation_text += f"User: {turn['user']}\nAI: {turn['ai']}\n"
            return conversation_text
        except:
            return ""

    def _save_interaction(self, user_query, ai_response):
        history = []
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except:
                history = []

        history.append({"user": user_query, "ai": ai_response})
        if len(history) > 10:
            history = history[-10:]

        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)

    def clear_history(self):
        if os.path.exists(self.history_file):
            try:
                os.remove(self.history_file)
            except:
                pass

    # --- HELPER: Extract text from nested content ---
    def _flatten_content(self, content, prefix=""):
        """Recursively flatten nested dicts/lists into readable text."""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            parts = []
            for item in content:
                parts.append(self._flatten_content(item, prefix))
            return "\n".join(parts)
        elif isinstance(content, dict):
            parts = []
            for key, value in content.items():
                flat_value = self._flatten_content(value, f"{key}: ")
                if isinstance(value, (dict, list)):
                    parts.append(f"{key}:\n{flat_value}")
                else:
                    parts.append(f"{key}: {flat_value}")
            return "\n".join(parts)
        else:
            return str(content)

    # --- CORE RAG ---
    def load_and_process_data(self):
        if not os.path.exists(self.kb_path):
            raise FileNotFoundError(f"Knowledge base not found at {self.kb_path}")

        with open(self.kb_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        documents = []

        # Handle list-based JSON structure (array of page objects)
        if isinstance(data, list):
            for page in data:
                page_title = page.get("page_title", "Unknown Page")
                source_url = page.get("source_url", "")
                content = page.get("content", "")

                # Flatten the content (handles nested dicts/lists)
                flat_content = self._flatten_content(content)

                # Build a document entry
                doc_text = f"Page: {page_title}\nSource: {source_url}\n\n{flat_content}"
                documents.append(doc_text)

        # Handle dict-based JSON structure (original expected format)
        elif isinstance(data, dict):
            # 1. Company
            company = data.get("company", {})
            if company:
                documents.append(
                    f"Company: {company.get('name', 'N/A')}\n"
                    f"Overview: {company.get('overview', 'N/A')}\n"
                    f"Mission: {company.get('mission', 'N/A')}\n"
                    f"Values: {', '.join(company.get('core_values', []))}"
                )

            # 2. Products
            for product in data.get("products", []):
                content = (
                    f"Product: {product.get('name', 'N/A')}\n"
                    f"Brand: {product.get('brand', 'N/A')} | Model: {product.get('model', 'N/A')}\n"
                    f"Category: {product.get('category', 'N/A')}\n"
                    f"Description: {product.get('description', 'N/A')}\n"
                    f"Features: {', '.join(product.get('features', []))}\n"
                )
                if "technical_specs" in product:
                    specs = product['technical_specs']
                    spec_str = ", ".join([f"{k}: {v}" for k, v in specs.items()])
                    content += f"Specs: {spec_str}\n"
                content += (
                    f"Warranty: {product.get('warranty', 'N/A')}\n"
                    f"Customers: {product.get('total_customers', 'N/A')}\n"
                    f"Origin: {product.get('country_of_make', 'N/A')}\n"
                )
                documents.append(content)

            # 3. Features & Resources
            for feat in data.get("platform_features", []):
                documents.append(f"Feature: {feat.get('feature', 'N/A')}\nBenefit: {feat.get('benefit', 'N/A')}")

            buyer_res = data.get("buyer_resources", {})
            for faq in buyer_res.get("faqs", []):
                documents.append(f"Buyer FAQ: Q: {faq.get('q', '')} A: {faq.get('a', '')}")

            seller_res = data.get("seller_resources", {})
            if seller_res.get('onboarding_process'):
                documents.append(f"Seller Onboarding: {seller_res.get('onboarding_process')}")
            for faq in seller_res.get("faqs", []):
                documents.append(f"Seller FAQ: Q: {faq.get('q', '')} A: {faq.get('a', '')}")

            for partner in data.get("partners", []):
                documents.append(
                    f"Partner: {partner.get('name', 'N/A')}\n"
                    f"Role: {partner.get('role', 'N/A')}\n"
                    f"Details: {partner.get('details', 'N/A')}"
                )
        else:
            raise ValueError(f"Unsupported knowledge base format: expected list or dict, got {type(data).__name__}")

        if not documents:
            raise ValueError("No documents were extracted from the knowledge base.")

        # Build Vector Store
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.create_documents(documents)

        self.vector_store = FAISS.from_documents(texts, self.embeddings)
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})

        prompt = PromptTemplate(
            template=SYSTEM_PROMPT,
            input_variables=["context", "chat_history", "question"]
        )

        self.chain = (
                {
                    "context": retriever,
                    "question": RunnablePassthrough(),
                    "chat_history": lambda x: self._load_history()
                }
                | prompt
                | self.model
                | StrOutputParser()
        )

    def is_obvious_spam(self, query: str) -> bool:
        query_lower = query.lower()
        if any(intent in query_lower for intent in BLOCKED_INTENTS):
            return True
        return False

    def query(self, user_input: str):
        if not self.vector_store:
            self.load_and_process_data()

        if self.is_obvious_spam(user_input):
            return "I'm PingBot, specialized in Pingbhai's electronics equipment. I can't help with that topic."

        try:
            response = self.chain.invoke(user_input)
            self._save_interaction(user_input, response)
            return response
        except Exception as e:
            return f"Error: {str(e)}"