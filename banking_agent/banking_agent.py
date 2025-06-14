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
from typing import Optional, Literal
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
import base64

# ------------------------------------------------------------------------------------------------------------

from .prompts import AGENT_RECOMMENDATION_RESPONSE_PROMPT, SUMMARIZATION_RESPONSE_PROMPT, AGENT_CONVO_SYSTEM_PROMPT, AGENT_CONVO_RESPONSE_PROMPT, AGENT_ORCHESTRATION_PROMPT
from .tools import get_promotional_policies, search_internet_func, get_personal_info_and_behaviour_data, get_available_eligible_products, get_used_products, draw_customer_behaviour_analysis
from data_analysis_manager import calculate_credit_score, get_top_n_recommendations_new_customer 

class SummarizationResponse(BaseModel):
    topics_of_interest: List[str]

class RecommendationQuestion(BaseModel):
    recommendations: List[str]

class PaymentMetadata(BaseModel):
    target_acc_id: str
    amount: float
    account_name: str

class NavigationJump(BaseModel):
    jump_to_other_pages: bool
    jumping_page: Optional[Literal["payment"]] = None
    payment_metadata: Optional[PaymentMetadata] = None 

load_dotenv()

class BankingAgent:
    def __init__(self):
        try:
            self.model_type = "gemini-2.5-flash-preview-05-20"
            self.embed_model = SentenceTransformer('BAAI/bge-m3')
            self.qdrant_client = QdrantClient(url="qdrant_all:6333")
            self.user_db_manager = DatabaseManager()
            self.tokenizer = AutoTokenizer.from_pretrained("Cloyne/vietnamese-embedding_finetuned_pair")
            Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3", cache_folder="/myProject/.cache")
            self.llm = Gemini(model=f"models/{self.model_type}")
            Settings.llm = self.llm
            self.agent_convo_context = AGENT_CONVO_SYSTEM_PROMPT
            self.agent_behavior_analysis_context = "None"
    
            self.text_splitter = TokenTextSplitter(
                separator=" ", chunk_size=2048, chunk_overlap=64
            )

            self.search_internet_tool = FunctionTool.from_defaults(
                fn=search_internet_func,
                name="search_internet_tool",
                description="Bạn có thể sử dụng công cụ này để tìm thông tin về tin tức tài chính hoặc bất kỳ chủ đề khác liên quan ngân hàng, những thông tin cần thiết",
            )

            self.analysis_tool = FunctionTool.from_defaults(
                fn=get_personal_info_and_behaviour_data,
                name="get_personal_info_and_behaviour_data_tool",
                description="Bạn có thể sử dụng công cụ này để tìm thông tin về những chủ đề người dùng quan tâm theo mức độ phần trăm; thông tin về tài khoản ngân hàng như số dư tài khoản, số nợ,...; Và thu nhập của người dùng",
            )

            self.draw_customer_behaviour_analysis_tool = FunctionTool.from_defaults(
                fn=draw_customer_behaviour_analysis,
                name="draw_customer_behaviour_analysis_tool",
                description="Sử dụng công cụ này để trực quan hóa mức độ quan tâm của khách hàng đối với các sản phẩm tài chính qua biểu đồ hình tròn."
            )

            self.orchestrator_agent = genai.Client() # Orchestrator agent for managing the flow of conversation and tool usage
            self.behavior_analysis_agent = self.create_behavior_analysis_agent() # Agent for analyzing user behavior and generating recommendations
            self.recommendation_agent = genai.Client() # Agent for recommendations generation based on user behavior data, user's topic of interest and bank policies
            self.base_convo_agent = self.create_convo_agent() # Agent for handling conversations and using tools

            self.convo_agent = {} # support multi-user

            print("Successfully initialized BankingAgent with all components!")
        except Exception as e:
            print(f"Error initializing BankingAgent: {e}")
            raise e
    
    def create_convo_agent(self, bank_promotional_policies_path="banking_agent/data/banking_promotional_policies.txt"):
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
                self.search_internet_tool,
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
        
    def create_behavior_analysis_agent(self):
        try:
            tools = [
                self.analysis_tool,
                self.draw_customer_behaviour_analysis_tool
            ]

            behavior_analysis_agent = ReActAgent.from_tools(tools=tools, llm=self.llm, verbose=True, context=self.agent_behavior_analysis_context)
            print("Successfully created banking conversation agent!")

            return behavior_analysis_agent

        except Exception as e:
            print("Error from creating_agent function: ", e)
            return {"success": False, "message": f"Error while creating banking agent {str(e)}"}
        
    def update_user_conversation(self, user_id, new_convo: List[dict] = {}):
        try:
            if not user_id:
                return 

            user_info = self.user_db_manager.get_user_by_id(user_id)
            if not user_info:
                return 

            new_chat = user_info.get('past_conversations', '')
            new_chat.extend(new_convo)

            user_info['past_conversations'] = new_chat

            self.user_db_manager.update_user_info(user_id, user_info)

        except Exception as e:
            print(f"Error updating user conversation: {e}")

    def orchestrate(self, user_input):
        try:
            final_prompt = AGENT_ORCHESTRATION_PROMPT.format(
                user_question=user_input
            )
            response = self.recommendation_agent.models.generate_content(
                model=self.model_type,
                config={
                'response_mime_type': 'application/json',
                'response_schema': NavigationJump,
                },
                contents=final_prompt
            )

            return response.parsed
        except Exception as e:
            print(f"Error orchestrating conversation: {e}")
            return {"success": False, "message": str(e)}
                
    def get_summarization_topics_of_interest_past_convo(self, user_info) -> str:
        past_conversations = str(user_info.get('past_conversations', '')) # past_convo is a list of dict objects :D, convert to string

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
    

    def agent_recommendation_response(self, user_id, top_k_questions=3) -> dict:
        try:
            user_info = self.user_db_manager.get_user_by_id(user_id)
            if not user_info:
                return {
                    "success": False,
                    "response": "Không tìm thấy người dùng trong dữ liệu!"
                }
            
            current_banking_promotional_policies = get_promotional_policies() or ""
            
            summarization_past_convo = self.get_summarization_topics_of_interest_past_convo(user_info) or ""

            topic_of_interest_probs, current_financial_state, income_tier = get_personal_info_and_behaviour_data(user_info)

            used_products = get_used_products(user_info)
            available_eligible_products = get_available_eligible_products(user_info)

            final_prompt = AGENT_RECOMMENDATION_RESPONSE_PROMPT.format(
                topics_of_interest_from_past_conversations=summarization_past_convo,
                topic_care_weights_description=topic_of_interest_probs,
                current_financial_state=current_financial_state,
                current_banking_promotional_policies=current_banking_promotional_policies,
                used_products=used_products,
                income_tier=income_tier,
                recommended_eligible_products=available_eligible_products,
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
            if not user_id:
                return {"success": False, "message": "No user ID provided."}
            
            if user_id not in self.convo_agent:
                self.convo_agent[user_id] = self.base_convo_agent

            if not user_input:
                return {"success": False, "message": "No user input provided."}
            
            checking_jumping = self.orchestrate(user_input)

            if checking_jumping and checking_jumping.jump_to_other_pages:
                return {
                    "success": True,
                    "jump_to_other_pages": True,
                    "jumping_page": checking_jumping.jumping_page,
                    "payment_metadata": checking_jumping.payment_metadata,
                    "response": f"Chắc chắn rồi! Mình sẽ giúp bạn thực hiện giao dịch tới số tài khoản {checking_jumping.payment_metadata.target_acc_id} với tên của người nhận là {checking_jumping.payment_metadata.account_name} với số tiền là {checking_jumping.payment_metadata.amount}"
                }
            
            tracking_convo = [{"Role": "User", "Content": user_input}]

            user_info = self.user_db_manager.get_user_by_id(user_id)

            topic_of_interest_probs, current_financial_state, income_tier = get_personal_info_and_behaviour_data(user_info)

            final_prompt = AGENT_CONVO_RESPONSE_PROMPT.format(
                user_question=user_input,
                current_financial_state=current_financial_state,
                topic_care_weights_description=topic_of_interest_probs,
                income_tier=income_tier
            )

            response = self.convo_agent[user_id].chat(final_prompt).response.strip()

            tracking_convo.append({"Role": "Assistant", "Content": response})

            self.update_user_conversation(user_id, tracking_convo)

            return {
                "success": True,
                "jump_to_other_pages": False,
                "jumping_page": None,
                "payment_metadata": None,
                "response": response
            }
        except Exception as e:
            print(f"Error while Convo Agent responding: {e}")
            return {"success": False, "message": str(e)}
        
    def agent_draw_customer_behaviour_analysis(self, user_id, save_path="banking_agent/customer_behaviour_analysis/banking_product_interest_percentage.jpg"):
        try:
            if not user_id:
                return {"success": False, "message": "No user ID provided."}

            user_info = self.user_db_manager.get_user_by_id(user_id)

            # final_prompt = f"Given this user_info: {user_info}, please use the draw_customer_behaviour_analysis_tool to draw customer's topics of interest diagram. Input the tool function like this:\n"

            # formatted_input = "{user_info: user_info}"

            # response = self.behavior_analysis_agent.chat(final_prompt).response.strip()

            draw_customer_behaviour_analysis(user_info=user_info, save_path=save_path)
            image = Image.open(save_path)

            buffer = BytesIO()
            image.save(buffer, format="PNG")
            encoded_string = base64.b64encode(buffer.getvalue()).decode("utf-8")

            return {
                "success": True,
                "image": f"data:image/png;base64,{encoded_string}"
            }
        except Exception as e:
            print(f"Error retrieving promotional policies: {e}")
            return {"success": False, "message": str(e)}
