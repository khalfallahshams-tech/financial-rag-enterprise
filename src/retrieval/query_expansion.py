"""
Financial Domain Acronym Expansion Module.
Bridges lexical-semantic disconnects by augmenting financial abbreviations with formal industry definitions.
"""

import re
from typing import Dict, List, Tuple

FINANCIAL_ACRONYM_MAP: Dict[str, str] = {
    "DBA": "Doing Business As",
    "EIN": "Employer Identification Number",
    "APR": "Annual Percentage Rate",
    "APY": "Annual Percentage Yield",
    "HELOC": "Home Equity Line of Credit",
    "PMI": "Private Mortgage Insurance",
    "CD": "Certificate of Deposit",
    "ETF": "Exchange Traded Fund",
    "REIT": "Real Estate Investment Trust",
    "IRA": "Individual Retirement Account",
    "HSA": "Health Savings Account",
    "FSA": "Flexible Spending Account",
    "P/E": "Price to Earnings Ratio",
    "NAV": "Net Asset Value",
    "LLC": "Limited Liability Company",
    "USPS": "United States Postal Service",
    "KYC": "Know Your Customer",
    "AML": "Anti-Money Laundering",
    "SEPP": "Substantially Equal Periodic Payments",
}


class FinancialAcronymExpander:
    """
    Expands financial acronyms into parenthetical descriptions.
    Example:
        'Can a DBA open a 401k?' -> 'Can a DBA (Doing Business As) open a 401k?'
    """
    def __init__(self, acronym_map: Dict[str, str] = None):
        self.acronym_map = acronym_map or FINANCIAL_ACRONYM_MAP

    def expand(self, query: str) -> Tuple[str, List[str]]:
        """
        Expands recognized financial acronyms in the query.
        Returns:
            (expanded_query, list_of_expanded_acronyms)
        """
        expanded_acronyms = []
        
        def replace_match(match: re.Match) -> str:
            token = match.group(0)
            token_upper = token.upper()
            if token_upper in self.acronym_map:
                expanded_acronyms.append(token_upper)
                return f"{token} ({self.acronym_map[token_upper]})"
            return token

        # Word boundary pattern for acronyms
        pattern = r"\b[A-Za-z/]{2,6}\b"
        expanded_query = re.sub(pattern, replace_match, query)

        return expanded_query, expanded_acronyms

