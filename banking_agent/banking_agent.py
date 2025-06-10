# import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from google import genai
from pydantic import BaseModel
from genai import types
from prompts import RECOMMENDATION_SYSTEM_PROMPT, RECOMMENDATION_RESPONSE_PROMPT, SUMMARIZATION_RESPONSE_PROMPT, RAG_LLM_SYSTEM_PROMPT, RAG_LLM_RESPONSE_PROMPT
from user_db_manager import DatabaseManager
from tools import calculate_topic_care_weights_description, get_promotional_policies
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import Distance, VectorParams

class SummarizationResponse(BaseModel):
    topics_of_interest: str

load_dotenv()

class BankingAgent:
    def __init__(self):
        try:
            self.model_type = "gemini-1.5-flash"
            self.gemini_client = genai.Client()
            self.chatbot = self.chatbot_client.chats.create(model=self.model_type)
            self.embed_model = SentenceTransformer('BAAI/bge-m3', cache_folder=".cache")
            self.qdrant_client = QdrantClient(url="qdrant_base:6333")
            self.user_db_manager = DatabaseManager()
            self.tokenizer = AutoTokenizer.from_pretrained("Cloyne/vietnamese-embedding_finetuned_pair")
    
            self.text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                self.tokenizer,
                chunk_size = 1024,    
                chunk_overlap = int(64),
                separators=["\n\n", "\n", "; ", ".", ","],
                is_separator_regex=False
            )

            print("Successfully initialized BankingAgent with all components!")
        except Exception as e:
            print(f"Error initializing BankingAgent: {e}")
            raise e
    
    def create_qdrant_vector_collection(self, embeddings, description_chunks_obj, dim=1024, collection_name="PromotionalPolicies"):
        if self.qdrant_client.collection_exists(collection_name):
             self.qdrant_client.delete_collection(collection_name=collection_name)

        self.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

        self.qdrant_client.upload_collection(
            collection_name=collection_name,
            vectors=embeddings,
            payload=description_chunks_obj,
            ids=None,   
            batch_size=4,  # How many vectors will be uploaded in a single request?
        )

        print("Successfully upload the collection!")

    def embedding_promotional_policies(self) -> dict:
        try:
            promotional_policies = get_promotional_policies()
            if not promotional_policies:
                return {"success": False, "message": "No promotional policies provided."}

            chunks = self.text_splitter.split_text(promotional_policies)

            embeddings = [self.embed_model.encode(chunk, normalize_embeddings=True) for chunk in chunks]

            description_chunks_obj = [{"text": chunk} for chunk in chunks]

            self.create_qdrant_vector_collection(embeddings, description_chunks_obj)
            print("Successfully created Qdrant vector collection with embeddings!")

            return {"success": True, "message": "Embeddings created and uploaded to Qdrant."}
        except Exception as e:
            print(f"Error in embedding promotional policies: {e}")
            return {"success": False, "message": str(e)}

    def retrieve_relevant_promotional_policies(self, user_input, top_k=2, collection_name="PromotionalPolicies") -> dict:
        try:
            if not user_input:
                return {"success": False, "message": "No user input provided."}

            query_embedding = self.embed_model.encode(user_input, normalize_embeddings=True)

            search_results = self.qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=top_k, # take k most relevant results,
                query_filter=None,  
            )

            if not search_results:
                return {"success": False, "relevant_policies": "Không tìm thấy chính sách ưu đãi nào liên quan."}

            relevant_policies = ""
            for result in search_results:
                relevant_policies += result.payload['text'] 

            print("Retrieved relevant promotional policies:", relevant_policies)

            return {"success": True, "relevant_policies": relevant_policies}
        except Exception as e:
            print(f"Error retrieving promotional policies: {e}")
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
            response = self.client.models.generate_content(
                model=self.model_type,
                config={
                'response_mime_type': 'application/json',
                'response_schema': SummarizationResponse,
                },
                contents=final_prompt
            )

            topics_of_interest = response.parsed.topics_of_interest.strip()

            print("Summarized topics of interest:", topics_of_interest)

            return topics_of_interest
        except Exception as e:
            print(f"Error generating content: {e}")
            return "No summarization available at the moment."
        
    def update_user_conversation(self, user_id):
        try:
            if not user_id:
                return 

            user_info = self.user_db_manager.get_user_by_id(user_id)
            if not user_info:
                return 

            new_convo = ""

            for message in self.chatbot.get_history():
                new_convo += f'Role - {message.role}: {message.parts[0].text}\n'

            user_info['past_conversations'] = user_info.get('past_conversations', '') + new_convo

            self.user_db_manager.update_user_info(user_id, user_info)
        except Exception as e:
            print(f"Error updating user conversation: {e}")

    def get_recommendation(self, user_id) -> dict:
        user_info = self.user_db_manager.get_user_by_id(user_id)
        if not user_info:
            return {
                "success": False,
                "response": "Không tìm thấy người dùng trong dữ liệu!"
            }
        
        current_banking_promotional_policies = get_promotional_policies() | ""
        
        summarization_past_convo = self.get_summarization_topics_of_interest_past_convo(user_info) | ""

        topic_care_weights_description = calculate_topic_care_weights_description(user_info) | ""

        current_financial_state = f"Số dư tài khoản hiện tại của người dùng: ${user_info.get('user_current_acc_balance', 0)}."

        final_prompt = RECOMMENDATION_RESPONSE_PROMPT.format(
            topics_of_interest_from_past_conversations=summarization_past_convo,
            topic_care_weights_description=topic_care_weights_description,
            current_financial_state=current_financial_state,
            current_banking_promotional_policies=current_banking_promotional_policies,
            user_type="regular"
        )

        print("Final prompt for recommendation:", final_prompt)

        response = self.client.models.generate_content(
            model=self.model_type,
            config=types.GenerateContentConfig(
                system_instruction=RECOMMENDATION_SYSTEM_PROMPT),
            contents=final_prompt
        )

        print("Generated recommendation response:", response)
        return {
            "success": True,
            "response": response.text.strip()
        }

    def rag_response(self, user_input, user_id) -> dict:
        try:
            if not user_input:
                return {"success": False, "message": "No user input provided."}
            
            if not user_id:
                return {"success": False, "message": "No user ID provided."}
            
            user_info = self.user_db_manager.get_user_by_id(user_id)

            relevant_policies = self.retrieve_relevant_promotional_policies(user_input)
            relevant_policies = "Không tìm thấy chính sách khuyến mãi liên quan nào." if not relevant_policies.get("success") else relevant_policies.get("relevant_policies")

            current_financial_state = f"Số dư tài khoản hiện tại của người dùng: ${user_info.get('user_current_acc_balance', 0)}."


            final_prompt = RAG_LLM_RESPONSE_PROMPT.format(
                user_question=user_input,
                current_financial_state=current_financial_state,
                relevant_banking_info_policies=relevant_policies
            )

            response = self.client.models.generate_content(
                model=self.model_type,
                config=types.GenerateContentConfig(
                    system_instruction=RAG_LLM_SYSTEM_PROMPT),
                contents=final_prompt
            )

            self.update_user_conversation(user_id)

            return {
                "success": True,
                "response": response.text.strip()
            }
        except Exception as e:
            print(f"Error retrieving promotional policies: {e}")
            return {"success": False, "message": str(e)}