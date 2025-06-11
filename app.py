from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
import nest_asyncio
from pydantic import BaseModel
from qdrant_client import QdrantClient
from apscheduler.schedulers.background import BackgroundScheduler
from banking_agent import BankingAgent
from user_db_manager import UserSchema
nest_asyncio.apply()

class InputPrompt(BaseModel):
    user_input: str
    user_id: str

class UserID(BaseModel):
    user_id: str

class UpdateFrequency(BaseModel):
    user_id: str
    product: str # e.g., "deposit", "credit_loan", "stock_investment"

load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/api/agent/update_user_info")
def update_user_info(user_data: UserSchema):
    try:
        user_id = user_data.user_id
        if not user_id:
            return {"success": False, "message": "User ID is required for updating user info"}

        response = banking_agent.user_db_manager.update_user_info(user_id, user_data.model_dump())
        if response.get("success"):
            return {"success": True, "message": "User info updated successfully"}
        else:
            return {"success": False, "message": response.get("message", "Failed to update user info")}
    except Exception as e:
        print("Error in update_user_info: ", e)
        return {"success": False, "message": str(e)}

@app.post("/api/agent/get_user_info")
def get_user_info(user_id_input: UserID):
    try:
        user_id = user_id_input.user_id
        if not user_id:
            return {"success": False, "message": "User ID is required for getting user info"}

        user_info = banking_agent.user_db_manager.get_user_by_id(user_id)
        if not user_info:
            return {"success": False, "message": "User not found"}

        return {"success": True, "user_info": user_info}
    except Exception as e:
        print("Error in get_user_info: ", e)
        return {"success": False, "message": str(e)}

@app.post("/api/agent/create_user")
def create_user(user_data: UserSchema):
    try:
        user_id = user_data.user_id
        if not user_id:
            return {"success": False, "message": "User ID is required for creating a user"}

        response = banking_agent.user_db_manager.create_user(user_data.model_dump())
        if response.get("success"):
            return {"success": True, "message": "User created successfully"}
        else:
            return {"success": False, "message": response.get("message", "Failed to create user")}
    except Exception as e:
        print("Error in create_user: ", e)
        return {"success": False, "message": str(e)}

@app.post("/api/agent/update_frequency")
def update_frequency(update_freq: UpdateFrequency):
    try:
        product_type = update_freq.product.lower()
        if product_type not in ["deposit", "credit_loan", "stock_investment"]:
            return {"success": False, "message": "invalid type for updating"}
        user_id = update_freq.user_id
        user_info = banking_agent.user_db_manager.get_user_by_id(user_id)
        if not user_info:
            return {"success": False, "message": "User not found for updating frequency"}

        user_info[f"total_freq_{product_type}"] += 1

        response = banking_agent.user_db_manager.update_user_info(user_id, user_info)

        if response.get("success"):
            return {"success": True, "message": f"Frequency for {product_type} updated successfully"}
    except Exception as e:
        print("Error in update_frequency: ", e)

@app.post("/api/agent/get_recommendation")
def get_recommendation(user_id_input: UserID):
    try:
        user_id = user_id_input.user_id
        if not user_id:
            return {"success": False, "message": "User not found for getting recommendations"}

        recommendations = banking_agent.get_recommendation(user_id)
        if not recommendations.get("success"):
            return {"success": False, "message": recommendations.get("response", "Không khuyến nghị nào được tìm thấy")}

        return {"success": True, "recommendations": recommendations.get("response")}
    except Exception as e:
        return {"success": False, "message": f"An error occurred while processing the request: {str(e)}"}

@app.post("/api/agent/agent_response")
def agent_response(input_prompt: InputPrompt):
    try:
        user_input = input_prompt.user_input
        user_id = input_prompt.user_id
        
        if not user_input:
            return {"success": False, "message": "User input is required for getting responses"}

        if not user_id:
            return {"success": False, "message": "User not found for getting responses"}

        response = banking_agent.agent_response(user_input, user_id)
        if not response.get("success"):
            return {"success": False, "message": response.get("message", "No response found")}

        return {"success": True, "response": response.get("response")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":

    qdrant_client = QdrantClient(url='qdrant_all:6333')

    banking_agent = BankingAgent()
    
    # scheduler = BackgroundScheduler()
    # scheduler.add_job(code_generative_agent.delete_inactive_users, "interval", minutes=10)  # Check every 10 minutes
    # scheduler.start()

    print("successfully initialize everything!")
    try:
        uvicorn.run(app, host="0.0.0.0", port=8053, loop="asyncio")
        print("successfully initialize the fastapi application")
    except Exception as e:
        print("Errors with the fastapi app initialization: ", e)
