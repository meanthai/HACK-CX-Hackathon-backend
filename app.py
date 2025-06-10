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

nest_asyncio.apply()

class InputPrompt(BaseModel):
    user_input: str
    user_id: str

class UserID(BaseModel):
    user_id: str

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


@app.post("/api/agent/get_recommendation")
def get_recommendation(user_id_input: UserID):
    try:
        user_id = user_id_input.user_id
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID is required")

        recommendations = banking_agent.get_recommendation(user_id)
        if not recommendations.get("success"):
            return {"success": False, "message": recommendations.get("response", "Không khuyến nghị nào được tìm thấy")}

        return {"success": True, "recommendations": recommendations.get("response")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        return {"success": False, "message": "An error occurred while processing the request"}

@app.post("/api/agent/rag_response")
def rag_response(input_prompt: InputPrompt):
    try:
        user_input = input_prompt.user_input
        user_id = input_prompt.user_id

        if not user_input or not user_id:
            raise HTTPException(status_code=400, detail="User input and User ID are required")

        response = banking_agent.rag_response(user_input, user_id)
        if not response.get("success"):
            return {"success": False, "message": response.get("message", "Không có phản hồi nào được tìm thấy")}

        return {"success": True, "response": response.get("response")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":

    qdrant_client = QdrantClient(url='qdrant_all:6333')

    banking_agent = BankingAgent()
    banking_agent.embedding_promotional_policies()
    
    # scheduler = BackgroundScheduler()
    # scheduler.add_job(code_generative_agent.delete_inactive_users, "interval", minutes=10)  # Check every 10 minutes
    # scheduler.start()

    print("successfully initialize everything!")
    try:
        uvicorn.run(app, host="0.0.0.0", port=8053, loop="asyncio")
        print("successfully initialize the fastapi application")
    except Exception as e:
        print("Errors with the fastapi app initialization: ", e)
