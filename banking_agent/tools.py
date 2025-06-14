import math
from datetime import datetime
import os
from .utils_setting import financial_terms_vi, default_credit_constraints, banking_products, default_age_constraints, default_avg_balance_constraints
from user_db_manager import UserSchema

def search_internet_func(query: str, num_results: int = 2) -> dict:
    """
    This function searches the internet for a given query and returns the results.
    You could use this tool to find information about banking products, financial news, or any other relevant topic.
    Args:
        query: The search query string
        num_results: Number of results to return (default: 5)
        
    Returns:
        A dictionary containing search results or error information
    """
    try:
        if not query or not isinstance(query, str):
            return {"error": "Query must be a non-empty string"}
        
        api_key = os.environ.get("SERP_API_KEY")
        if not api_key:
            return {"error": "Search API key not configured"}
        
        from serpapi import GoogleSearch
        
        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "num": num_results
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        if "error" in results:
            return {"error": results["error"]}
        
        formatted_results = []
        if "organic_results" in results:
            for result in results["organic_results"][:num_results]:
                formatted_results.append({
                    "title": result.get("title", ""),
                    "snippet": result.get("snippet", ""),
                })
        
        return {
            "search_results": formatted_results,
            "query": query,
            "total_results_found": len(formatted_results)
        }
    
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}

def get_promotional_policies(policies_path = 'banking_agent/data/banking_promotional_policies.txt') -> str:
    content = ""
    with open(policies_path, 'r') as file:
        content = file.read()
    return content

def softmax(weight_dict):
    exps = {k: math.exp(v) for k, v in weight_dict.items()}
    sum_exps = sum(exps.values())
    return {k: v / sum_exps for k, v in exps.items()}

def format_softmax_weights(weight_dict):
    prob_weights = softmax(weight_dict)
    final_str = ""
    for key, prob in prob_weights.items():
        final_str += f"{round(prob * 100, 2)}% mức độ quan tâm cho sản phẩm: {key}\n"
    return prob_weights, final_str

def calculate_topics_of_interest_probs(user_info, alpha=0.6, beta=0.4, tau=1440) -> str:
    now = datetime.now()
    
    freqs = {
        product: user_info[f"total_freq_{product}"]
        for product in banking_products
    }
    max_freq = max(freqs.values()) if max(freqs.values()) > 0 else 1  

    timestamps = {
        product: user_info[f"last_{product}_timestamp"]
        for product in banking_products
    }

    weights = {}
    for key in freqs:
        freq = freqs[key]
        normalized_freq = freq / max_freq

        last_time = timestamps[key]
        if last_time:
            delta_minutes = (now - last_time).total_seconds() / 60
            recency_score = math.exp(-delta_minutes / tau)
        else:
            recency_score = 0  # if never used

        weight = alpha * normalized_freq + beta * recency_score
        weights[key] = round(weight, 4)  

    prob_weights, topic_of_interest_probs = format_softmax_weights(weights)

    return prob_weights, topic_of_interest_probs 

import plotly.graph_objects as go

def draw_customer_behaviour_analysis(user_info: UserSchema, banking_products=banking_products, save_path="banking_agent/customer_behaviour_analysis/banking_product_interest_percentage.jpg"):
    """
    Draws and saves a circular pie chart using Plotly to visualize customer interest 
    percentages towards various banking products based on given information about user_info.
    
    Args:
        user_info : User's information for analyzing the topics of interest probabilities distribution
        banking_products (list): List of banking product keys (matching `probs`).
        save_path (str): File path to save the resulting chart image.
    """
    prob_weights, _ = calculate_topics_of_interest_probs(user_info)
    labels = [product.replace("_", " ").title() for product in banking_products]
    values = [prob_weights.get(product, 0) for product in banking_products]

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(
        title_text="Customer Interest in Banking Products (Percentage)",
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent outer background
        plot_bgcolor='rgba(0,0,0,0)',   # Transparent plot background
    )
    
    fig.write_image(save_path)

def get_used_products(user_info) -> str:
    used_products = []
    for key, val in financial_terms_vi.items():
        product = f"used_{key}"
        if product in user_info and user_info[product]:
            used_products.append(val)

    return ", ".join(used_products) if used_products else "Không có sản phẩm nào được sử dụng gần đây."

def get_available_eligible_products(user_info):
    unused_products = []
    recommended_eligible_products = []

    for key in banking_products:
        product = f"used_{key}"
        if product in user_info and not user_info[product]:
            unused_products.append(key)

    for product in unused_products:
        if user_info["user_age"] >= default_age_constraints[product] and user_info["credit_score"] >= default_credit_constraints[product] and user_info['current_acc_balance'] >= default_avg_balance_constraints[product]:
            recommended_eligible_products.append(financial_terms_vi[product])

    return ", ".join(recommended_eligible_products) if recommended_eligible_products else "Hãy tự tìm các sản phẩm thích hợp với khách hàng."

def get_personal_info_and_behaviour_data(user_info):
    _, topic_of_interest_probs = calculate_topics_of_interest_probs(user_info) or ""

    current_financial_state = f"Số dư tài khoản hiện tại của người dùng: ${user_info.get('user_current_acc_balance', 0)}. Số dư nợ hiện tại của người dùng: ${user_info.get('user_current_acc_debit', 0)}."

    income_tier = user_info.get("income_tier", "")
    return topic_of_interest_probs, current_financial_state, income_tier