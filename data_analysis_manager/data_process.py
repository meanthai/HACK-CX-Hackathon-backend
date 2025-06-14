import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Process data for credit score calculation
user_data = pd.read_excel("HACK-CX-Hackathon-backend/Data Analytics/raw_data/customers.xlsx")

user_data_extracted = user_data[['income_tier', 'tenure_years', 'avg_balance', 'mortgage_outstanding', 'investments_aum',
                                 ]].copy()

# Handle missing values
user_data_extracted.dropna(inplace=True)

# Handle duplicate entries
user_data_extracted.drop_duplicates(inplace=True)

# Handle outliers in numerical columns
numerical_cols_name = user_data_extracted.select_dtypes(include=['int64', 'float64']).columns.tolist()
for col_name in numerical_cols_name:
    Q1 = user_data_extracted[col_name].quantile(0.25)
    Q3 = user_data_extracted[col_name].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    user_data_extracted = user_data_extracted[(user_data_extracted[col_name] >= lower) & (user_data_extracted[col_name] <= upper)]
    

# Handle income_tier ranking
def rank_income_tier(income_tier):
    income_tier = income_tier.strip().lower().replace(' ', '-')
    if income_tier == 'low':
        return 0.2
    elif income_tier == 'lower-middle':
        return 0.4
    elif income_tier == 'upper-middle':
        return 0.6
    elif income_tier == 'high':
        return 0.8
    else:
        return 1.0
    
user_data_extracted['income_tier'] = user_data_extracted['income_tier'].apply(lambda x: rank_income_tier(x))

user_data_extracted.to_csv("HACK-CX-Hackathon-backend/Data Analytics/processed_data/user_data_credit_score.csv", index=False)

#----------------------------------------------------------------------#
# Process data for NBO model
customer_data = user_data
product_data = pd.read_excel('HACK-CX-Hackathon-backend/Data Analytics/raw_data/products.xlsx')
adoption_data = pd.read_excel('HACK-CX-Hackathon-backend/Data Analytics/raw_data/adoption_logs.xlsx')

pop = adoption_data.pop('user_id')
adoption_data.insert(0, 'user_id', pop)

final_data = pd.merge(adoption_data, product_data,  on='product_id', how='inner').copy()
final_data = pd.merge(customer_data, final_data, on='user_id', how='inner')

extrac_data = final_data[['age', 'clv_score', 'avg_balance', 'monetary_volume', 'mortgage_outstanding', 'activity_intensity',
            'investments_aum', 'recency_days', 'category']].copy()

extrac_data_filter = extrac_data[(extrac_data['category'] != 'Mortgage') & (extrac_data['category'] != 'Overdraft')]

# Handle Duplicate entries
extrac_data_filter = extrac_data_filter.drop_duplicates().copy()

# Handle missing values
extrac_data_filter = extrac_data_filter.dropna().copy()

# Sample data for equal target lablels
FXTransfer = extrac_data_filter[extrac_data_filter['category'] == 'FXTransfer'].sample(4500, random_state=42, replace=False)
SavingsAccount = extrac_data_filter[extrac_data_filter['category'] == 'SavingsAccount'].sample(4500, random_state=42, replace=False)
PersonalLoan = extrac_data_filter[extrac_data_filter['category'] == 'PersonalLoan'].sample(4500, random_state=42, replace=False)
DebitCard = extrac_data_filter[extrac_data_filter['category'] == 'DebitCard'].sample(4500, random_state=42, replace=False)
InvestmentFund = extrac_data_filter[extrac_data_filter['category'] == 'InvestmentFund'].sample(4500, random_state=42, replace=False)
FixedDeposit = extrac_data_filter[extrac_data_filter['category'] == 'FixedDeposit'].sample(4500, random_state=42, replace=False)
CreditCard = extrac_data_filter[extrac_data_filter['category'] == 'CreditCard'].sample(4500, random_state=42, replace=False)
Insurance = extrac_data_filter[extrac_data_filter['category'] == 'Insurance'].sample(4500, random_state=42, replace=False)

# Concatenate the sampled data
final_sampled_data = pd.concat([FXTransfer, SavingsAccount, PersonalLoan, DebitCard, InvestmentFund,
                                 FixedDeposit, CreditCard, Insurance], ignore_index=True, axis=0)

# Handle outliers in numerical columns
numerical_cols_name = final_sampled_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
for col_name in numerical_cols_name:
    Q1 = final_sampled_data[col_name].quantile(0.25)
    Q3 = final_sampled_data[col_name].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    final_sampled_data = final_sampled_data[(final_sampled_data[col_name] >= lower) & (final_sampled_data[col_name] <= upper)]

# Shuffle the final sampled data
final_sampled_data = final_sampled_data.sample(frac=1, random_state=42, replace=False).reset_index(drop=True)

# Save the final processed data
final_sampled_data.to_csv("HACK-CX-Hackathon-backend/Data Analytics/processed_data/history_data_nbo_filter_outlier.csv", index=False)