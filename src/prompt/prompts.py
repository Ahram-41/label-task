"""
This module contains prompt templates used for various LLM tasks.
"""
# Version 1
CORE_OR_GENERAL_AI_APPLICATION = """You are a research analysis assistant for AI business analysis. Based on the AI startups’ company business descriptions, classify these AI startups’ businesses as core AI technologies or general AI applications. The definitions are as follows:
Being core AI technologies startups means that these AI startups focus on developing foundational AI technologies, tools, or platforms that enable the creation, training, deployment, or improvement of AI systems. Their primary goal is to provide general-purpose AI solutions that can be applied across various industries. These AI startups are often focused on advancing the infrastructure, algorithms, and core capabilities that support a wide range of downstream applications. The class for Core AI technologies is 0. 
Being a general AI application startup means applying existing AI technologies to solve problems in specific industries or functional domains, such as healthcare, finance, retail, logistics, or customer service. These AI startups offer tailored AI-driven solutions that are deeply integrated into the target industry, aiming to optimize operational performance and meet distinct contextual needs. Their innovation lies in domain integration, data usage, and user experience, not necessarily in developing new AI methods. You do not need to restrict general AI application startups to a single industry. An AI startup that applies AI technologies across multiple industries can still be classified as a general AI application startup, as long as the startup does not develop foundational AI technologies and its value proposition is primarily in the application of existing AI tools to solve domain-specific problems, even if these domains vary. The class for general AI application startups is 1.

{content}

{format_instructions}
"""

IS_DATA_CENTRIC = """
You are a research analysis assistant for AI business analysis. Based on the AI startups’ company business descriptions, classify these AI startups’ businesses as data-centric AI startups or others. The definitions are as follows. Being data-centric AI startups means that the core competitive advantages of these AI startups come from the acquisition, curation, or exclusive access to valuable data. These startups rely heavily on data intelligence and on proprietary, high-quality, or domain-specific datasets to train and improve their AI models. Their data assets are often difficult to replicate or access publicly. The class for data-centric AI startups is 1. All other startups are classified as 0. Notably, even when a startup utilizes public data sources, it can be considered data-centric if it has a distinctive ability to mine, clean, integrate, and transform these sources into a non-trivial, high-value dataset that is not easily replicable by others. For each classification, please output the predicted probability of Y = 1 (data-centric) or Y = 0 (non-data-centric), along with a short explanation supporting the classification decision.

{content}

{format_instructions}
"""

IS_NICHE_OR_BROAD_MARKET_A = """You are a research analysis assistant for AI business analysis. Based on the AI startups’ company business descriptions, classify these AI startups’ businesses as niche AI startups or others. The definitions are as follows. 
Being niche AI startups means that these AI startups develop customized AI solutions tailored to narrowly defined market segments or specialized application domains. These startups often serve customers with highly specific, complex, or previously unmet needs, and derive their competitive advantage from deep domain expertise, solution customization, and technological specialization. 
The class for niche AI startups is 1. All other startups are classified as 0. For each classification, please output the predicted probability of Y = 1 (niche) or Y = 0 (non-niche), along with a short explanation supporting.

{content}

{format_instructions}
"""
IS_NICHE_OR_BROAD_MARKET_B = """You are a research analysis assistant for AI business analysis. Based on the AI startups’ company business descriptions, classify these AI startups’ businesses as niche AI startups or others. The definitions are as follows. 
Being niche AI startups means that these AI startups develop customized AI solutions tailored to narrowly defined market segments or specialized application domains. These startups often serve customers with highly specific, complex, or previously unmet needs, and derive their competitive advantage from deep domain expertise, solution customization, and technological specialization. It is important that the classification does not depend on the number of industries, functions, or solutions served. Rather, it hinges on how the AI technology is applied within specialized market segments or specific application domains. 
The class for niche AI startups is 1. All other startups are classified as 0. For each classification, please output the predicted probability of Y = 1 (niche) or Y = 0 (non-niche), along with a short explanation supporting.

{content}

{format_instructions}
"""

IS_PRODUCT_OR_PLATFORM = """
You are a research analysis assistant for AI business analysis. Based on the AI startups’ company business descriptions, classify these AI startups’ businesses as AI platform startups or others. The definitions are as follows. 
Being an AI platform startup means providing computing infrastructure, tools, or integrated environments that enable others (developers, data scientists, enterprises) to build, deploy, or integrate AI applications. These startups often enable third-party developers and users to develop, deploy, and manage AI applications collaboratively. AI platforms often exhibit network effects, where increased usage and data improve the performance and value of the platform for all users. 
The class for AI platform startups is 1. All other startups are classified as 0. For each classification, please output the predicted probability of Y = 1 (platform) or Y = 0 (non-platform), along with a short explanation supporting.

{content}

{format_instructions}
"""
HARDWARE_OR_FOUNDATION_MODEL = """You are an expert in AI ecosystems and startup classification. Based on the following definitions adapted from academic literature, classify the core business of each AI startup into one of these three categories:
1. Hardware
Startups that design or manufacture physical infrastructure for AI workloads, such as GPUs, TPUs, custom AI chips, edge AI hardware, or specialized computing architectures. These firms enable the computational foundation for training and running AI models.
2. Foundation Model or Tool
Startups that build large-scale machine learning models trained on diverse and massive datasets. These foundation models (like GPT, BERT, or Stable Diffusion) can be adapted to a wide range of tasks and serve as general-purpose AI systems. These startups focus on developing foundational AI technologies, tools, or platforms that enable the creation, training, deployment, or improvement of AI products or systems.
3. Application AI
Startups that use existing AI models or services to deliver domain-specific solutions for industries such as healthcare, finance, education, marketing, etc. They do not build foundational AI models themselves but integrate or fine-tune them for real-world applications.
Hardware is classified as 1, foundation model is classified as 2, application AI is classified as 3, and all others as 4. Please provide the probability over 4 types. Also provide a brief explanation.


{content}

{format_instructions}
"""


# Version 3
VERTICAL_OR_HORIZONTAL = """You are a startup strategy expert. Classify the business focus of an AI startup as either Vertical or Horizontal based on the definitions below:

- Vertical AI Startup: Tailors AI to a specific industry or domain (e.g., healthcare, legal, retail).
- Horizontal AI Startup: Provides general-purpose AI tools usable across multiple sectors.

Examples:
Description: Provides AI models for automated medical imaging diagnostics.
→ Vertical AI Startup: 0.95
→ Horizontal AI Startup: 0.05
Explanation: Medical imaging is a specific domain (healthcare), indicating vertical specialization.

Description: Offers a general-purpose summarization API for documents.
→ Vertical AI Startup: 0.10
→ Horizontal AI Startup: 0.90
Explanation: Applicable across legal, finance, education — this is a general-use horizontal tool.

Now classify the startup below:

Startup Description:
{content}

{format_instructions}
"""

DEVELOPER_OR_INTEGRATOR = """
You are analyzing the AI value chain. Classify a startup based on whether it is a Model Developer (builds and owns AI models) or a Model Integrator (uses third-party models). Use the definitions below:

- Model Developer: Trains and owns proprietary models (e.g., builds its own LLMs or vision models).
- Model Integrator: Uses public APIs or open-source models to build applications or tools.

Examples:
Description: Trains custom transformer models for legal document analysis.
→ Model Developer: 0.90
→ Model Integrator: 0.10
Explanation: Indicates in-house model development for a specific task.

Description: Uses OpenAI’s API to provide AI customer support chatbots.
→ Model Developer: 0.05
→ Model Integrator: 0.95
Explanation: Relies on external models and builds value through workflow integration.

Now classify this startup:

Startup Description:
{content}

{format_instructions}
"""

AI_NATIVE_OR_AUGMENTED = """
You are a venture analyst assessing the strategic role of AI in startups. Based on the definitions below, classify the startup’s product:

- AI-Native Product: The product is AI — core value is from AI model output (e.g., code generation, content generation, predictions).
- AI-Augmented Product: AI is used to enhance a broader offering (e.g., CRM with AI-based recommendations).

Examples:
Description: Offers an AI tool that automatically writes marketing copy based on a brand’s tone.
→ AI-Native Startup: 0.90
→ AI-Augmented Product: 0.10
Explanation: The product itself is generative AI — not just enhanced.

Description: Provides an HR management platform with AI-powered candidate scoring.
→ AI-Native Startup: 0.20
→ AI-Augmented Product: 0.80
Explanation: AI supports an existing HR workflow, not the core product.

Now classify the following startup:

Startup Description:
{content}

{format_instructions}
"""

AUTOMATION_DEPTH = """You are analyzing the degree of automation in AI startups. Classify the startup's AI offering into one of the following categories based on its role in replacing or supporting human labor:

- Full Automation: The product performs the task entirely on its own, replacing human labor (e.g., AI that performs radiology diagnosis without human intervention).
- Human-in-the-loop: The product supports humans during the task, requiring human input or review (e.g., AI-assisted contract review).
- Recommendation/Insight Only: The product provides suggestions or insights but does not execute decisions or actions (e.g., dashboards or alerts based on AI predictions).

Examples:
Description: An AI that fully analyzes financial statements and generates audit reports without requiring human input.
→ Full Automation: 0.85
→ Human-in-the-loop: 0.10
→ Recommendation/Insight Only: 0.05
Explanation: This system replaces a complete human task, qualifying as full automation.

Description: Provides AI-generated contract summaries for lawyers to review.
→ Full Automation: 0.05
→ Human-in-the-loop: 0.85
→ Recommendation/Insight Only: 0.10
Explanation: AI assists humans who remain responsible for the task.

Description: Sends AI-generated alerts on changes in customer behavior but does not take action.
→ Full Automation: 0.05
→ Human-in-the-loop: 0.15
→ Recommendation/Insight Only: 0.80
Explanation: AI is advisory and does not make or act on decisions.

Now classify the following startup:

Startup Description:
{content}

{format_instructions}
"""

base_prompt = [CORE_OR_GENERAL_AI_APPLICATION, IS_DATA_CENTRIC, IS_NICHE_OR_BROAD_MARKET_A,IS_NICHE_OR_BROAD_MARKET_B, IS_PRODUCT_OR_PLATFORM, HARDWARE_OR_FOUNDATION_MODEL]
additional_prompt = [VERTICAL_OR_HORIZONTAL, DEVELOPER_OR_INTEGRATOR, AI_NATIVE_OR_AUGMENTED, AUTOMATION_DEPTH]