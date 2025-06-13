from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from typing import List
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
    user_bank_account_id: str
    
    income_tier: str 
    user_occupation: str
    user_type: str = "new_user"  # Default user type

    credit_score: float = 0.0
    current_acc_balance: float = 0.0
    current_acc_debit: float = 0.0
    # tenure_years: float = 0.0

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
    