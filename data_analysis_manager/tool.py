import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression

def calculate_credit_score(new_income_tier, new_tenure_years, new_avg_balance, new_mortgage_outstanding, new_investments_aum, historical_data):
    # preprocess tenure_years
    scaler_tenure = MinMaxScaler()
    historical_data[['tenure_years']] = scaler_tenure.fit_transform(historical_data[['tenure_years']].values)

    # preprocess avg_balance
    scaler_avg_balance = MinMaxScaler()
    historical_data[['avg_balance']] = scaler_avg_balance.fit_transform(historical_data[['avg_balance']].values)

    # preprocess mortgage_outstanding
    scaler_mortgage = MinMaxScaler()
    historical_data[['mortgage_outstanding']] = scaler_mortgage.fit_transform(historical_data[['mortgage_outstanding']].values)

    # preprocess investments_aum
    scaler_investment = MinMaxScaler()
    historical_data[['investments_aum']] = scaler_investment.fit_transform(historical_data[['investments_aum']].values)

    #-----------#
    # calculate value for new values
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

    # rank income_tier
    new_income_tier = rank_income_tier(new_income_tier)

    # scale tenure_years
    new_tenure_years = scaler_tenure.transform(np.array([[new_tenure_years]]))[0][0]

    # scale avg_balance
    new_avg_balance = scaler_avg_balance.transform(np.array([[new_avg_balance]]))[0][0]

    # scale mortgage_outstanding
    new_mortgage_outstanding = scaler_mortgage.transform(np.array([[new_mortgage_outstanding]]))[0][0]
    
    # scale investments_aum
    new_investments_aum = scaler_investment.transform(np.array([[new_investments_aum]]))[0][0]
    #-----------#

    # calculate credit score
    credit_score = (new_income_tier * 0.3 + 
                    new_tenure_years * 0.05 + 
                    new_avg_balance * 0.1 + 
                    (1 - new_mortgage_outstanding) * 0.35 + 
                    new_investments_aum * 0.2) * 1000
    
    return credit_score.item()
#--------------------#

def get_top_n_recommendations_new_customer(new_age, new_clv_score, new_avg_balance, new_monetary_volume,
                                           new_mortgage_outstanding, new_activity_intensity, new_investments_aum,
                                           new_recency_days, history_data_nbo, top_n):
    
    mapping = {'FXTransfer': 0, 'SavingsAccount': 1, 'PersonalLoan': 2, 'DebitCard': 3,
                'InvestmentFund': 4, 'FixedDeposit': 5, 'CreditCard': 6, 'Insurance': 7}
    
    def get_categories_high_probability(pred_proba, top_n = 4):
        dict_cate = {}
        dict_map = {index: proba for index, proba in enumerate(pred_proba)}

        sorted_dict = sorted(dict_map.items(), key=lambda item: item[1], reverse=True)
        for i in range(top_n):
            dict_cate[sorted_dict[i][0]] = sorted_dict[i][1]

        mapping_reverse = {v: k for k, v in mapping.items()}

        new_dict_cate = {}
        for index, proba in dict_cate.items():
            category = mapping_reverse[index]
            new_dict_cate[category] = proba

        return new_dict_cate
    
    
    
    shuffle_history_data_nbo = history_data_nbo.sample(frac=1, random_state=42).reset_index(drop=True)
    
    scaler = StandardScaler()
    scaler.fit(shuffle_history_data_nbo[['age', 'clv_score', 'avg_balance', 'monetary_volume', 'mortgage_outstanding', 
                                                      'activity_intensity', 'investments_aum', 'recency_days']].values)
    
    X = scaler.transform(shuffle_history_data_nbo[['age', 'clv_score', 'avg_balance', 'monetary_volume', 'mortgage_outstanding', 
                                                      'activity_intensity', 'investments_aum', 'recency_days']].values)
    y = shuffle_history_data_nbo['category'].map(mapping)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X, y)
    
    new_data = np.array([[new_age, new_clv_score, new_avg_balance, new_monetary_volume,
                          new_mortgage_outstanding, new_activity_intensity, new_investments_aum,
                          new_recency_days]])
    
    new_data_scaled = scaler.transform(new_data)
    pred_proba = lr.predict_proba(new_data_scaled)[0]
    categories_high_prob = get_categories_high_probability(pred_proba, top_n=top_n)

    return categories_high_prob
    
