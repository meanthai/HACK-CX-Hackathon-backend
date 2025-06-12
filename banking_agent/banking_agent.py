# import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from pydantic import BaseModel
from google import genai
from google.genai import types
from user_db_manager import DatabaseManager
from transformers import AutoTokenizer
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import TokenTextSplitter
from typing import List
import os
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.tools import FunctionTool
from llama_index.core.storage.docstore.simple_docstore import (
    SimpleDocumentStore,
)
from llama_index.core.extractors import DocumentContextExtractor

# ------------------

from .prompts import AGENT_RECOMMENDATION_RESPONSE_PROMPT, SUMMARIZATION_RESPONSE_PROMPT, AGENT_CONVO_SYSTEM_PROMPT, AGENT_CONVO_RESPONSE_PROMPT, AGENT_ORCHESTRATION_PROMPT
from .tools import calculate_topic_care_weights_description, get_promotional_policies, search_internet_func, get_used_products, get_recommended_eligible_products

class SummarizationResponse(BaseModel):
    topics_of_interest: List[str]

class RecommendationQuestion(BaseModel):
    recommendations: List[str]

class BankingRelatedQuestion(BaseModel):
    related: bool

load_dotenv()

class BankingAgent:
    def __init__(self):
        try:
            self.model_type = "gemini-2.0-flash"
            self.embed_model = SentenceTransformer('BAAI/bge-m3')
            self.qdrant_client = QdrantClient(url="qdrant_all:6333")
            self.user_db_manager = DatabaseManager()
            self.tokenizer = AutoTokenizer.from_pretrained("Cloyne/vietnamese-embedding_finetuned_pair")
            Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3", cache_folder="/myProject/.cache")
            self.llm = Gemini(model=f"models/{self.model_type}")
            Settings.llm = self.llm
            self.agent_convo_context = AGENT_CONVO_SYSTEM_PROMPT
    
            self.text_splitter = TokenTextSplitter(
                separator=" ", chunk_size=2048, chunk_overlap=64
            )

            self.search_internet = FunctionTool.from_defaults(
                fn=search_internet_func,
                name="search_internet_func",
                description="Bạn có thể sử dụng công cụ này để tìm thông tin về tin tức tài chính hoặc bất kỳ chủ đề khác liên quan ngân hàng, những thông tin cần thiết",
            )

            self.orchestrator_agent = genai.Client() # Orchestrator agent for managing the flow of conversation and tool usage
            self.behavior_analysis_agent = None # Agent for analyzing user behavior and generating recommendations
            self.recommendation_agent = genai.Client() # Agent for recommendations generation based on user behavior data, user's topic of interest and bank policies
            self.base_convo_agent = self.create_banking_agent() # Agent for handling conversations and using tools

            self.convo_agent = {} # support multi-user

            print("Successfully initialized BankingAgent with all components!")
        except Exception as e:
            print(f"Error initializing BankingAgent: {e}")
            raise e
    
    def create_banking_agent(self, bank_promotional_policies_path="banking_agent/data/banking_promotional_policies.txt"):
        try:
            print("data path exist?: ", os.path.isfile(bank_promotional_policies_path))
            documents = SimpleDirectoryReader(input_files=[bank_promotional_policies_path]).load_data()
        
            vector_store = QdrantVectorStore(client=self.qdrant_client, collection_name="PromotionalPolicies")

            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            docstore = SimpleDocumentStore()

            context_extractor = DocumentContextExtractor(
                docstore=docstore,
                max_context_length=128000, 
                oversized_document_strategy="warn",
                max_output_tokens=512,
                llm=self.llm,
                # key=user_id,
                prompt=DocumentContextExtractor.SUCCINCT_CONTEXT_PROMPT,
            )

            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                transformations=[self.text_splitter, context_extractor]
            )

            query_engine = index.as_query_engine(llm=self.llm, text_splitter=self.text_splitter)

            tools = [
                QueryEngineTool(
                    query_engine=query_engine,
                    metadata=ToolMetadata(
                        name="Banking_Promotional_Policies_Query_Engine",
                        description="Công cụ này cung cấp các chính sách khuyến mãi của ngân hàng dựa theo thông tin hoặc câu hỏi của người dùng. Nó có thể trả lời các câu hỏi về các ưu đãi ngân hàng hiện tại, sản phẩm tài chính, và các chương trình khuyến mãi mới nhất cũng như các thông tin liên quan khác của ngân hàng.",
                    )
                ),
                self.search_internet,
            ]

            del documents
            del vector_store
            del storage_context
            del context_extractor
            del docstore
            del query_engine

            convo_agent = ReActAgent.from_tools(tools=tools, llm=self.llm, verbose=True, context=self.agent_convo_context)
            print("Successfully created banking conversation agent!")

            return convo_agent

        except Exception as e:
            print("Error from creating_agent function: ", e)
            return {"success": False, "message": f"Error while creating banking agent {str(e)}"}
        
    def update_user_conversation(self, user_id, new_convo=""):
        try:
            if not user_id:
                return 

            user_info = self.user_db_manager.get_user_by_id(user_id)
            if not user_info:
                return 

            user_info['past_conversations'] = user_info.get('past_conversations', '') + new_convo

            self.user_db_manager.update_user_info(user_id, user_info)
        except Exception as e:
            print(f"Error updating user conversation: {e}")

    def orchestrate(self, user_input) -> dict:
        try:
            final_prompt = AGENT_ORCHESTRATION_PROMPT.format(
                user_question=user_input
            )
            response = self.recommendation_agent.models.generate_content(
                model=self.model_type,
                config={
                'response_mime_type': 'application/json',
                'response_schema': BankingRelatedQuestion,
                },
                contents=final_prompt
            )

            return {
                "is_related": response.parsed.related,
            }
        except Exception as e:
            print(f"Error orchestrating conversation: {e}")
            return {"success": False, "message": str(e)}
                
    def get_summarization_topics_of_interest_past_convo(self, user_info) -> str:
        past_conversations = user_info.get('past_conversations', '')

        if not past_conversations:
            return "Không tồn tại bất kì cuộc hội thoại nào để tóm tắt."

        final_prompt = SUMMARIZATION_RESPONSE_PROMPT.format(
            past_conversations=past_conversations
        )

        print("Final prompt for summarization:", final_prompt)

        try:
            response = self.recommendation_agent.models.generate_content(
                model=self.model_type,
                config={
                'response_mime_type': 'application/json',
                'response_schema': SummarizationResponse,
                },
                contents=final_prompt
            )

            topics_of_interest = response.parsed.topics_of_interest

            print("Summarized topics of interest:", topics_of_interest)

            return ", ".join(topics_of_interest)
        except Exception as e:
            print(f"Error generating content: {e}")
            return "No summarization available at the moment."
    

    def agent_recommendation_response(self, user_id, top_k_questions=5) -> dict:
        try:
            user_info = self.user_db_manager.get_user_by_id(user_id)
            if not user_info:
                return {
                    "success": False,
                    "response": "Không tìm thấy người dùng trong dữ liệu!"
                }
            
            current_banking_promotional_policies = get_promotional_policies() or ""
            
            summarization_past_convo = self.get_summarization_topics_of_interest_past_convo(user_info) or ""

            topic_care_weights_description = calculate_topic_care_weights_description(user_info) or ""

            current_financial_state = f"Số dư tài khoản hiện tại của người dùng: ${user_info.get('user_current_acc_balance', 0)}. Số dư nợ hiện tại của người dùng: ${user_info.get('user_current_acc_debit', 0)}."

            used_products = get_used_products(user_info)
            recommended_eligible_products = get_recommended_eligible_products(user_info)

            income_tier = user_info.get("income_tier", "")

            final_prompt = AGENT_RECOMMENDATION_RESPONSE_PROMPT.format(
                topics_of_interest_from_past_conversations=summarization_past_convo,
                topic_care_weights_description=topic_care_weights_description,
                current_financial_state=current_financial_state,
                current_banking_promotional_policies=current_banking_promotional_policies,
                used_products=used_products,
                income_tier=income_tier,
                recommended_eligible_products=recommended_eligible_products,
                user_type="regular"
            )

            print("Final prompt for recommendation:", final_prompt)

            response = self.recommendation_agent.models.generate_content(
                model=self.model_type,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': RecommendationQuestion,
                },
                contents=final_prompt
            )

            recommendations = response.parsed.recommendations[:top_k_questions]

            print("Generated recommendation response:", recommendations)
            return {
                "success": True,
                "response": recommendations
            }
        except Exception as e:
            print(f"Error generating recommendation response: {e}")
            return {
                "success": False,
                "message": str(e)
            }
    
    def agent_convo_response(self, user_input, user_id) -> dict:
        try:
            if user_id not in self.convo_agent:
                self.convo_agent[user_id] = self.base_convo_agent

            if not user_input:
                return {"success": False, "message": "No user input provided."}
            
            if not user_id:
                return {"success": False, "message": "No user ID provided."}
            
            # is_related = self.orchestrate(user_input)
            
            tracking_convo = "Vai trò: Người dùng\nNội dung: " + user_input + "\n"

            user_info = self.user_db_manager.get_user_by_id(user_id)

            current_financial_state = f"Số dư tài khoản hiện tại của người dùng: ${user_info.get('user_current_acc_balance') if user_info.get('user_current_acc_balance') else 'Không được tiết lộ'}. Số dư nợ hiện tại của người dùng: ${user_info.get('user_current_acc_debit') if user_info.get('user_current_acc_debit') else 'Không được tiết lộ'}."

            final_prompt = AGENT_CONVO_RESPONSE_PROMPT.format(
                user_question=user_input,
                current_financial_state=current_financial_state,
            )

            response = self.convo_agent[user_id].chat(final_prompt).response.strip()

            tracking_convo += "Vai trò: Trợ lý ngân hàng: " + response + "\n"

            self.update_user_conversation(user_id, tracking_convo)

            return {
                "success": True,
                "response": response
            }
        except Exception as e:
            print(f"Error retrieving promotional policies: {e}")
            return {"success": False, "message": str(e)}