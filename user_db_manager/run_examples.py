# from .user_db_manager import DatabaseManager
from datetime  import datetime

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os
from typing import Optional, List
# ------------------------------------------------------------------
# from .user_schema import UserSchema
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
class UserSchema(BaseModel):
    """
    User Schema:
    - user_name: str (required)
    - user_id: str (required, unique)
    - user_current_acc_balance: float (default: 0.0)
    - created_at: datetime (auto-generated)
    - updated_at: datetime (auto-updated)
    """
    user_name: str
    user_id: str
    user_email: str
    user_phone_number: str
    user_age: int = 16 # default
    
    income_tier: str 
    user_occupation: str
    user_type: str = "new_user"  # Default user type

    credit_score: float = 0.0
    current_acc_balance: float = 0.0
    current_acc_debit: float = 0.0

    total_freq_deposit_account: int = 0
    total_freq_saving: int = 0
    total_freq_credit_card: int = 0
    total_freq_mortgage: int = 0
    total_freq_investment_fund: int = 0
    total_freq_insurance: int = 0
    total_freq_personal_loan: int = 0
    total_freq_fx_transfer: int = 0

    last_deposit_account_timestamp: Optional[datetime] = None
    last_saving_timestamp: Optional[datetime] = None
    last_credit_card_timestamp: Optional[datetime] = None
    last_mortgage_timestamp: Optional[datetime] = None
    last_investment_fund_timestamp: Optional[datetime] = None
    last_insurance_timestamp: Optional[datetime] = None
    last_personal_loan_timestamp: Optional[datetime] = None
    last_fx_transfer_timestamp: Optional[datetime] = None

    used_deposit_account: bool = 0
    used_saving: bool = 0
    used_credit_card: bool = 0
    used_mortgage: bool = 0
    used_investment_fund: bool = 0
    used_insurance: bool = 0
    used_personal_loan: bool = 0
    used_fx_transfer: bool = 0
    
    past_conversations: List[dict] = Field(description="Past conversations with the user")
    
class DatabaseManager:
    def __init__(self, connection_str = "MONGODB_CONNECTION_STRING", db_name="HACK-CX-Hackathon", collection_name="users"):
        try:
            load_dotenv()
            self.uri = os.getenv(connection_str)
            self.client = MongoClient(self.uri, server_api=ServerApi('1'))
            self.client.admin.command('ping')
            self.collection = self.client.get_database(db_name).get_collection(collection_name)

            print("Successfully initialized DatabaseManager and connected to MongoDB!")
        except Exception as e:
            print(f"Error connecting to MongoDB: {e}")

    def create_user(self, user_data) -> dict:
        """Create a new user"""
        try:
            existing_user = self.collection.find_one({"user_id": user_data["user_id"]})
            if existing_user:
                print(f"Error: User with user_id '{user_data['user_id']}' already exists!")
                return {"success": False, "message": f"User with user_id '{user_data['user_id']}' already exists!"}
            
            result = self.collection.insert_one(user_data)
            print(f"User created successfully with ID: {result.inserted_id}")
            return {"success": True, "message": f"User created successfully with ID: {result.inserted_id}"}
        
        except Exception as e:
            print(f"Error creating user: {e}")
            return {"success": False, "message": str(e)}

    def get_user_by_id(self, user_id) -> Optional[UserSchema]:
        try:
            user = self.collection.find_one({"user_id": user_id})
            if user:
                print(f"User found: {user}")
                return user
            else:
                print(f"No user found with user_id: {user_id}")
                return None
        except Exception as e:
            print(f"Error finding user: {e}")
            return None

    def get_all_users(self) -> List[UserSchema]:
        try:
            users = list(self.collection.find())
            print(f"Found {len(users)} users:")
            return users
        except Exception as e:
            print(f"Error getting users: {e}")
            return []

    def update_user_info(self, user_id, update_data) -> dict:
        try:        
            result = self.collection.update_one(
                {"user_id": user_id},
                {"$set": update_data}
            )
            
            if result.matched_count > 0:
                print(f"Successfully updated user '{user_id}' with data: {update_data}")
                return {"success": True, "message": f"User '{user_id}' updated successfully."}
            else:
                print(f"No user found with user_id: {user_id}")
                return {"success": False, "message": f"No user found with user_id: {user_id}"}
        
        except Exception as e:
            print(f"Error updating user: {e}")
            return {"success": False, "message": str(e)}
        
    def delete_user(self, user_id) -> dict:
        try:
            result = self.collection.delete_one({"user_id": user_id})
            if result.deleted_count > 0:
                print(f"Successfully deleted user with user_id: {user_id}")
                return {"success": True, "message": f"User '{user_id}' deleted successfully."}
            else:
                print(f"No user found with user_id: {user_id}")
                return {"success": False, "message": f"No user found with user_id: {user_id}"}
        except Exception as e:
            print(f"Error deleting user: {e}")
            return {"success": False, "message": str(e)}

def run_examples(db_manager=None):
    """Run example CRUD operations"""
    print("\n" + "="*50)
    print("RUNNING CRUD EXAMPLES")
    print("="*50)
    
    print("\n1. CREATING SAMPLE USERS:")
    print("-" * 30)
    
    sample_users = [
        {
        "user_name": "Nguyen Van A",
        "user_id": "user_001",
        "user_email": "nguyenvana@example.com",
        "user_phone_number": "0912345678",
        "user_age": 28,

        "income_tier": "middle",
        "user_occupation": "Software Engineer",
        "user_type": "new_user",

        "credit_score": 720.5,
        "current_acc_balance": 15000000.0,
        "current_acc_debit": 5000000.0,

        "total_freq_deposit_account": 3,
        "total_freq_saving": 12,
        "total_freq_credit_card": 5,
        "total_freq_mortgage": 0,
        "total_freq_investment_fund": 1,
        "total_freq_insurance": 0,
        "total_freq_personal_loan": 0,
        "total_freq_fx_transfer": 4,

        "last_deposit_account_timestamp": datetime(2025, 5, 1, 14, 30),
        "last_saving_timestamp": datetime(2025, 4, 15, 10, 0),
        "last_credit_card_timestamp": datetime(2025, 6, 1, 8, 0),
        "last_mortgage_timestamp": None,
        "last_investment_fund_timestamp": datetime(2025, 3, 10, 9, 15),
        "last_insurance_timestamp": None,
        "last_personal_loan_timestamp": None,
        "last_fx_transfer_timestamp": datetime(2025, 6, 10, 13, 0),

        "used_deposit_account": 1,
        "used_saving": 1,
        "used_credit_card": 1,
        "used_mortgage": 0,
        "used_investment_fund": 1,
        "used_insurance": 0,
        "used_personal_loan": 0,
        "used_fx_transfer": 1,

        "past_conversations": [{
            "Role": "User",
            "Content": "Hi, can you help me understand what a credit score of 720.5 means?"
        },
        {
            "Role": "Assistant",
            "Content": "A credit score of 720.5 is considered a good score. It typically indicates that you are a low-risk borrower, which means you're more likely to be approved for loans and may receive better interest rates. Generally, scores above 700 are viewed positively by financial institutions."
        },
        {
            "Role": "User",
            "Content": "I recently used my credit card. Can I know how this affects my credit score?"
        },
        {
            "Role": "Assistant",
            "Content": """Using your credit card can impact your credit score in several ways. If you pay your bills on time and maintain a low credit utilization ratio (below 30% of your credit limit), it can improve your score. However, late payments or maxing out your card can lower your score."""
        }
        ]
    }
    ]
    
    deleted_user_id = "user_001"

    db_manager.delete_user(deleted_user_id)  

    for user in sample_users:
        db_manager.create_user(user)
    
    # print("\n2. READING USERS:")
    # print("-" * 30)
    
    # db_manager.get_all_users()
    
    # print("\nGetting specific user:")
    # db_manager.get_user_by_id("john_doe_001")
    
    # print("\n3. UPDATING USERS:")
    # print("-" * 30)
    

    # db_manager.update_user_info("jane_smith_002", {
    #     "user_name": "Jane Smith-Wilson",
    #     "user_current_acc_balance": 3000.00
    # })
    
    # print("\nUsers after updates:")
    # db_manager.get_all_users()
    
    # print("\n4. DELETING USER:")
    # print("-" * 30)
        
    # print("\nFinal user list:")
    # db_manager.get_all_users()


if __name__ == "__main__":
    db_manager = DatabaseManager()

    run_examples(db_manager=db_manager)