import math
from datetime import datetime
from typing import Dict
from user_db_manager import UserSchema
import os

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

def get_promotional_policies(policies_path = 'banking_agent/banking_promotional_policies.txt') -> str:
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
    return final_str

def calculate_topic_care_weights_description(user: UserSchema, alpha=0.5, beta=0.5, tau=1440) -> str:
        now = datetime.now()
        
        freqs = {
            'deposit': user.total_freq_deposit,
            'credit_loan': user.total_freq_credit_loan,
            'stock_investment': user.total_freq_stock_investment
        }
        max_freq = max(freqs.values()) if max(freqs.values()) > 0 else 1  # avoid division by zero

        timestamps = {
            'deposit': user.last_deposit_timestamp,
            'credit_loan': user.last_credit_loan_timestamp,
            'stock_investment': user.last_stock_investment_timestamp
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

        care_weight_description = format_softmax_weights(weights)

        return care_weight_description