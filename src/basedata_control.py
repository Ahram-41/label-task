from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Literal

class Founder(BaseModel):
    founder_name: str = Field(
        description="name of the person in founding team members."
    )
    degree_name: str = Field(
        description="the name of the degree of the founder."
    )
    cs_degree: int = Field(
        description="if this founder holds a computer science-related degree (including bachelor’s, master’s, doctoral, or other qualifications), it is 1; otherwise 0. If no information of educational qualifications, assign 2."
    )
    graduated_university: Literal["Others", "Unknown", "MIT", "University of Cambridge", "University of Oxford", "Harvard University", "Stanford University", "Imperial College London", "ETH Zurich", "National University of Singapore (NUS)", "University College London (UCL)", "University of California, Berkeley (UCB)"] = Field(
        description="Founder's Graduated universities. Should be one of the provided options."
    )
    prior_success_ipo: int = Field(
        description="if this founder has previous experience in founding companies that successfully exited through an IPO in the past, it is 1; otherwise, it is 0."
    )
    prior_success_ma: int = Field(
        description="if this founder previously founded companies that achieved successful exits through mergers and acquisitions (M&A), it is 1; otherwise, it is 0."
    )

class Founders_Background_Response(BaseModel):
    founders: List[Founder] = Field(
        description="a list of the founding team members' information."
    )

class Executive(BaseModel):
    executive_name: str = Field(
        description="name of the person in executive team members."
    )
    executive_title: str = Field(
        description="a list of the job titles of the executive team members."
    )
    
class Company_Executives_Response(BaseModel):
    executives: List[Executive] = Field(
        description="a list of the executive team members' information."
    )

class Products(BaseModel):
    product_name: str = Field(
        description="the name of the startup’s product."
    )
    is_ai_product: int = Field(
        description="if the product is AI related, it is 1; otherwise, it is 0."
    )
    tech_name: str = Field(
        description="the key technologies in this product. keep the technology name concise"
    )
class AI_Products_Response(BaseModel):
    products: List[Products] = Field(
        description="a list of the product that the company provides."
    )

class Technology(BaseModel):
    tech_name: str = Field(
        description="Name of the core functional technology developed by the company or used as the foundation for its products. If not provided, assign 'Unknown'."
    )
    is_ai_tech: int = Field(
        description="if the technology is AI related, it is 1; otherwise, it is 0."
    )

class AI_Technology_Response(BaseModel):
    technologies: List[Technology] = Field(
        description="a list of the key functional technologies utilized or developed by the company, excluding the infrastructure providers."
    )

class Partner(BaseModel):
    partner_name: str = Field(
        description="name of the partner company that is well-known in the AI industry, excluding general infrastructure providers. if there is no information, assign 'Unknown'."
    )
    collaboration_type: str = Field(
        description="type of the AI related collaboration or partnership."
    )
class AI_Partner_Response(BaseModel):
    partners: List[Partner] = Field(
        description="a list of the well-known AI partner companies."
    )

class IPO_MnA_Response(BaseModel):
    ipo: int = Field(
        description="if the startup has asuccessful IPO, it is 1; otherwise, it is 0."
    )
    ipo_date: str = Field(
        description="the date of the IPO event of the AI startup."
    )
    ma: int = Field(
        description="if the startup successfully exited through mergers and acquisitions (M&A), it is 1; otherwise, it is 0."
    )
    ma_date: str = Field(
        description="the yyyy/mm/dd date of the mergers and acquisitions event of the AI startup."
    )

class Patent_Information_Response(BaseModel):
    ai_patent: int = Field(
        description="if the startup has AI-related patenting activities, it is 1; otherwise, it is 0."
    )
    patent_class: List[str] = Field(
        description="patent class CPC code for each startup’s AI patents"
    )
    number_of_patent: int = Field(
        description="the yearly number of AI-related patents."
    )
    patent_year_details: List[str] = Field(
        description="a table showing year, PatentName, PatentClass, NumofPatent"
    )
    
    # both info of other capabilities and patent information
    genuine: int = Field(
        description="if the startup shows AI capabilities through patents, products, technologies, partnerships and collaborations, etc., it is 1; otherwise, it is 0."
    )