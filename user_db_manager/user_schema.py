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
    user_age: int = 18 # default
    
    income_tier: str 
    user_occupation: str
    user_type: str = "new_user"  # Default user type

    credit_point: float = 0.0
    current_acc_balance: float = 0.0
    current_acc_debit: float = 0.0
    credit_score: float = 0.0

    total_freq_deposit: int = 0
    total_freq_credit_loan: int = 0
    total_freq_stock_investment: int = 0

    last_deposit_timestamp: Optional[datetime] = None
    last_credit_loan_timestamp: Optional[datetime] = None
    last_stock_investment_timestamp: Optional[datetime] = None
    
    past_conversations: str = Field(description="Past conversations with the user")
    
    used_deposit_account: bool = 0
    used_saving: bool = 0
    used_credit_card: bool = 0
    used_mortgage: bool = 0
    used_investment_fund: bool = 0
    used_insurance: bool = 0
    used_personal_loan: bool = 0
    used_fx_transfer: bool = 0