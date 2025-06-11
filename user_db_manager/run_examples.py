from .user_db_manager import DatabaseManager

def run_examples(db_manager=None):
    """Run example CRUD operations"""
    print("\n" + "="*50)
    print("RUNNING CRUD EXAMPLES")
    print("="*50)
    
    print("\n1. CREATING SAMPLE USERS:")
    print("-" * 30)
    
    sample_users = [
        {
            "user_name": "John Doe",
            "user_id": "john_doe_001",
            "user_type": "new_user",
            "user_current_acc_balance": 1500.50,
            "user_current_acc_debit": 0.0,
            "total_freq_deposit"   : 13,
            "total_freq_credit_loan": 2,
            "total_freq_stock_investment": 3,
            "last_deposit_timestamp": datetime.now(),
            "last_credit_loan_timestamp": datetime.now(),
            "last_stock_investment_timestamp": datetime.now(),
            "past_conversations": """Vai trò - Trợ lý ngân hàng: Tôi là trợ lý ngân hàng của bạn, tôi có thể giúp gì cho bạn hôm nay?Người dùng: Tôi muốn biết số dư tài khoản của mình. Trợ lý ngân hàng: Số dư tài khoản hiện tại của bạn là 1500.50$."""
        }
    ]
    
    # db_manager.delete_user("john_doe_001")  # Clean up if user already exists

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