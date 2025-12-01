import json
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
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

    # --- CORE RAG ---
    def load_and_process_data(self):
        if not os.path.exists(self.kb_path):
            raise FileNotFoundError(f"Knowledge base not found at {self.kb_path}")

        with open(self.kb_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        documents = []

        # 1. Company
        company = data.get("company", {})
        documents.append(
            f"Company: {company.get('name')}\nOverview: {company.get('overview')}\nMission: {company.get('mission')}\nValues: {', '.join(company.get('core_values', []))}")

        # 2. Products
        for product in data.get("products", []):
            content = f"Product: {product['name']}\nBrand: {product.get('brand')} | Model: {product.get('model')}\nCategory: {product['category']}\nDescription: {product['description']}\nFeatures: {', '.join(product.get('features', []))}\n"
            if "technical_specs" in product:
                specs = product['technical_specs']
                spec_str = ", ".join([f"{k}: {v}" for k, v in specs.items()])
                content += f"Specs: {spec_str}\n"
            content += f"Warranty: {product.get('warranty', 'N/A')}\nCustomers: {product.get('total_customers', 'N/A')}\nOrigin: {product.get('country_of_make', 'N/A')}\n"
            documents.append(content)

        # 3. Features & Resources
        for feat in data.get("platform_features", []):
            documents.append(f"Feature: {feat['feature']}\nBenefit: {feat['benefit']}")

        buyer_res = data.get("buyer_resources", {})
        for faq in buyer_res.get("faqs", []):
            documents.append(f"Buyer FAQ: Q: {faq['q']} A: {faq['a']}")

        seller_res = data.get("seller_resources", {})
        documents.append(f"Seller Onboarding: {seller_res.get('onboarding_process')}")
        for faq in seller_res.get("faqs", []):
            documents.append(f"Seller FAQ: Q: {faq['q']} A: {faq['a']}")

        for partner in data.get("partners", []):
            documents.append(f"Partner: {partner['name']}\nRole: {partner['role']}\nDetails: {partner['details']}")

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