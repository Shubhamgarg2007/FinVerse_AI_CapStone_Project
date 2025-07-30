import re

def parse_natural_language(message: str) -> dict:
    """
    Parses a natural language string to extract financial profile details.
    Returns a dictionary with found values, or None for fields it cannot find.
    """
    message = message.lower() 
    extracted = {}

    # --- Regex for Age ---
    age_match = re.search(r"\b(\d{2})\b", message)
    if age_match:
        extracted["age"] = int(age_match.group(1))

    # --- Regex for Income ---
    income_match = re.search(r"(?:₹|rs\.?|inr)?\s?(\d+\.?\d*)\s?(l|lakhs?|lpa|k|thousands?)?", message)
    if income_match:
        value = float(income_match.group(1))
        suffix = income_match.group(2)
        if suffix and suffix.startswith('l'):
            value *= 100000
        elif suffix and suffix.startswith('k'):
            value *= 1000
        extracted["annual_income"] = int(value)

    # --- Keywords for Risk Appetite ---
    if re.search(r"\b(high risk|aggressive|maximize returns|volatility)\b", message):
        extracted["risk_appetite"] = "High"
    elif re.search(r"\b(low risk|conservative|protecting capital|safe|cautious)\b", message):
        extracted["risk_appetite"] = "Low"
    elif re.search(r"\b(medium risk|balanced|moderate)\b", message):
        extracted["risk_appetite"] = "Medium"
        
    # --- Keywords for Investment Goal ---
    if re.search(r"\b(retire|retirement)\b", message):
        extracted["investment_goal"] = "Retirement"
    elif re.search(r"\b(wealth creation|grow my money)\b", message):
        extracted["investment_goal"] = "Wealth Creation"
    elif re.search(r"\b(tax|80c|tax saving)\b", message):
        extracted["investment_goal"] = "Tax Saving"
    elif re.search(r"\b(short-term|down payment|car|house in \d+ years)\b", message):
        extracted["investment_goal"] = "Short-Term"
        
    return extracted
