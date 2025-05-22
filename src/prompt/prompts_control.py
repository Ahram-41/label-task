"""
This module contains prompt templates used for various LLM tasks.
"""
# Task 1
FOUNDERS_BACKGROUND = """Based on the AI startups' names and business descriptions, collect information on its FOUNDER team members and construct background variables. 

COMPANY NAME: {company_name}
DESCRIPTION: {description}

{format_instructions}
"""

COMPANY_EXECUTIVES = """You are a research analysis assistant for AI startup data collection. Based on the AI startups’ names or long company business descriptions, collect information on all top executives’ information and construct variables at the startup level. 
Measures the number of top executives belonging to each AI startup. The top executives include CEO, CFO, COO, CTO, Founder, President, Director, Board Member, and Vice-President. Please list their names and job titles in executive_name and executive_title.

COMPANY NAME: {company_name}
DESCRIPTION: {description}

{format_instructions}
"""
# Task 2
AI_PRODUCTS = """Based on the AI startups’ names and business description, find out the products that the company provides.

COMPANY NAME: {company_name}
BUSINESS DESCRIPTION: {description}

{format_instructions}
"""

AI_TECHNOLOGY = """Based on the AI startups’ names and business descriptions, collect information on the AI related Technology that the company has.

COMPANY NAME: {company_name}
BUSINESS DESCRIPTION: {description}

{format_instructions}
"""

Tasks=[FOUNDERS_BACKGROUND, COMPANY_EXECUTIVES, AI_PRODUCTS, AI_TECHNOLOGY]

PATENT_INFORMATION = """Based on the AI startups’ names and business descriptions, collect information on the AI related product that the company provides and technology that the product is based on.

{content}

{format_instructions}
"""

"""You are a research analysis assistant for verifying AI startups. Based on the AI startups’ names and long company business descriptions, verify whether it is a genuine AI startup by verifying the startup’s AI capabilities through patents, products, technologies, partnerships and collaborations, etc.

For patent-related information, search the PatentsView database or the USPTO’s official website. Additionally, provide the yearly distribution of patents, including: the year of each patent filing (year), the patent name/title (PatentName), the patent class (PatentClass), and the number of patents filed per year (NumofPatent).

AIProduct equals 1 if the startup has AI products; otherwise, it is 0.
ProductName measures the names of the startup’s AI product.
TechName measures the key AI technologies in each startup.
AITech equals 1 if the startup has AI-related technologies; otherwise, it is 0.
Partner equals 1 if the startup has partnerships or collaborations with well-known AI firms; otherwise, it is 0.
AIPatent equals 1 if the startup has AI-related patenting activities; otherwise, it is 0.


Please output the variables AIProduct (0/1), ProductName, TechName, AITech (0/1), Partner (0/1), AIPatent (0/1), PatentYearlyDetails, PatentClass, PatentName, NumofPatent, Genuine (0/1), along with a short explanation supporting each variable."""