FIRE_IPO_MNA_PROMPT = """collect information on the success IPO and completed M&A of the company.
COMPANY NAME: {company_name}
DESCRIPTION: {description}

extract KEY information that if the company has success IPO to go publics, IPO date; if the company was completed M&A, M&A date.
"""
FIRE_EXECUTIVE_PROMPT = """
collect information on the EXECUTIVE team of the company.
COMPANY NAME: {company_name}
DESCRIPTION: {description}

extract KEY information CURRENT EXECUTIVE of the company, include executive name, and executive title.
"""
FIRE_FOUNDER_PROMPT = """
collect information on the FOUNDER team of the company.
COMPANY NAME: {company_name}
DESCRIPTION: {description}

extract KEY information EACH FOUNDER of the company, include founder name, educational background, graduated university, prior success IPO of each founder, prior success M&A of each founder.
"""


FIRE_PRODUCT_PROMPT ="""
collect information on the PRODUCT of the company.
COMPANY NAME: {company_name}
DESCRIPTION: {description}

extract KEY information EACH PRODUCT of the company, include product name, and if the product is AI related.
"""
FIRE_TECHNOLOGY_PROMPT = """
collect information on the TECHNOLOGY of the company.
COMPANY NAME: {company_name}
DESCRIPTION: {description}

extract KEY information EACH TECHNOLOGY of the company, include technology name, and if the technology is AI related.
"""
FIRE_PARTNER_PROMPT = """
collect information on the PARTNER AI firm of the company.
COMPANY NAME: {company_name}
DESCRIPTION: {description}

extract KEY information EACH PARTNER of the {company_name}, include partner name, collaboration_type.
"""