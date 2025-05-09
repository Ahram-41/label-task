from langchain_core.pydantic_v1 import BaseModel, Field

class Core_or_General_AI_Application_Response(BaseModel):
    Core_or_General_AI_Application: int = Field(
        description="classify these AI startups' businesses as core AI technologies (0) or general AI applications (1)"
    )
    Core_or_General_AI_Application_explanation: str = Field(
        description="short explanation for the classification of core AI technologies or general AI applications"
    )
class Is_Data_Centric_Response(BaseModel):
    Is_data_centric: int = Field(
        description="classify these AI startups' businesses as data-centric (1) or non-data-centric (0)"
    )
    Is_data_centric_probability_of_Y_1: float = Field(
        description="probability of Y=1"
    )
    Is_data_centric_probability_of_Y_0: float = Field(
        description="probability of Y=0"
    )
    Is_data_centric_explanation: str = Field(
        description="short explanation for the classification of data-centric or non-data-centric"
    )
class Is_Niche_or_Broad_Market_A_Response(BaseModel):
    Is_niche_or_broad_market_A: int = Field(
        description="classify these AI startups' businesses as niche (1) or broad-market (0)"
    )
    Is_niche_or_broad_market_probability_of_Y_1_A: float = Field(
        description="probability of Y=1"
    )
    Is_niche_or_broad_market_probability_of_Y_0_A: float = Field(
        description="probability of Y=0"
    )
    Is_niche_or_broad_market_explanation_A: str = Field(
        description="short explanation for the classification of niche or broad-market"
    )
class Is_Niche_or_Broad_Market_B_Response(BaseModel):
    Is_niche_or_broad_market_B: int = Field(
        description="classify these AI startups' businesses as niche (1) or broad-market (0)"
    )
    Is_niche_or_broad_market_probability_of_Y_1_B: float = Field(
        description="probability of Y=1"
    )
    Is_niche_or_broad_market_probability_of_Y_0_B: float = Field(
        description="probability of Y=0"
    )
    Is_niche_or_broad_market_explanation_B: str = Field(
        description="short explanation for the classification of niche or broad-market"
    )
class Is_Product_or_Platform_Response(BaseModel):
    Is_product_or_platform: int = Field(
        description="classify these AI startups' businesses as product (0) or platform (1)"
    )
    Is_product_or_platform_probability_of_Y_1: float = Field(
        description="probability of Y=1"
    )
    Is_product_or_platform_probability_of_Y_0: float = Field(
        description="probability of Y=0"
    )
    Is_product_or_platform_explanation: str = Field(
        description="short explanation for the classification of product or platform"
    )
class AI_startup_type_Response(BaseModel):
    AI_startup_type: int = Field(
        description="classify these AI startups' businesses as hardware (1), foundation model (2), application AI (3), or all others (4)"
    )
    AI_startup_type_explanation: str = Field(
        description="short explanation for the classification of hardware, foundation model, application AI, or all others"
    )
class Is_Data_Centric_Response2(BaseModel):
    Is_data_centric: int = Field(
        description="classify these AI startups' businesses as data-centric (1) or non-data-centric (0)"
    )
    Is_data_centric_probability: float = Field(
        description="probability of Y=1"
    )
    Is_data_centric_explanation: str = Field(
        description="short explanation for the classification of data-centric or non-data-centric"
    )
class Is_Niche_or_Broad_Market_Response2(BaseModel):
    Is_niche_or_broad_market: int = Field(
        description="classify these AI startups' businesses as niche (1) or broad-market (0)"
    )
    Is_niche_or_broad_market_probability: float = Field(
        description="probability of Y=1"
    )
    Is_niche_or_broad_market_reasoning: str = Field(
        description="reasoning for the classification of niche or broad-market"
    )
class Is_Product_or_Platform_Response2(BaseModel):
    Is_product_or_platform: int = Field(
        description="classify these AI startups' businesses as product (0) or platform (1)"
    )
    Is_product_or_platform_probability: float = Field(
        description="probability of Y=1"
    )
    Is_product_or_platform_reasoning: str = Field(
        description="reasoning for the classification of product or platform"
    )
class AI_startup_type_Response3(BaseModel):
    vertical_ai_startup: float = Field(
        description="probability of the startup being a vertical AI startup"
    )
    horizontal_ai_startup: float = Field(
        description="probability of the startup being a horizontal AI startup"
    )
    vertical_or_horizontal_explanation: str = Field(
        description="explanation for the classification of vertical or horizontal AI startup"
    )
class Developer_or_Integrator_Response(BaseModel):
    model_developer: float = Field(
        description="probability of the startup being a model developer"
    )
    model_integrator: float = Field(
        description="probability of the startup being a model integrator"
    )
    model_developer_or_integrator_explanation: str = Field(
        description="explanation for the classification of model developer or model integrator"
    )
class AI_Native_or_Augmented_Response(BaseModel):
    ai_native: float = Field(
        description="probability of the startup being an AI-native product"
    )
    ai_augmented: float = Field(
        description="probability of the startup being an AI-augmented product"
    )
    ai_native_or_augmented_explanation: str = Field(
        description="explanation for the classification of AI-native or AI-augmented product"
    )
class Automation_Depth_Response(BaseModel):
    full_automation: float = Field(
        description="probability of the startup being a full automation product"
    )
    human_in_the_loop: float = Field(
        description="probability of the startup being a human-in-the-loop product"
    )
    recommendation_or_insight_only: float = Field(
        description="probability of the startup being a recommendation or insight only product"
    )
    automation_depth_explanation: str = Field(
        description="brief explanation for the classification of full automation, human-in-the-loop, or recommendation or insight only product"
    )
