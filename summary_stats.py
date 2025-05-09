import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Load the three datasets
base_df = pd.read_csv('small_sample_label_base_prompt.csv')
modified_df = pd.read_csv('small_sample_label_modified_prompt.csv')
additional_df = pd.read_csv('small_sample_label_additional_prompt.csv')

print("Summary Statistics for AI Startup Classification\n")
print(f"Total samples in each dataset: {len(base_df)}")

# Define a function to print label distributions
def print_label_distribution(df, column_name, title):
    counts = df[column_name].value_counts(normalize=True) * 100
    print(f"\n{title}:")
    for label, percentage in counts.items():
        print(f"  {label}: {percentage:.1f}%")

# Define function to print binary classification based on probability threshold
def print_binary_distribution(df, prob_column, title, threshold=0.5):
    # Convert probability to binary classification
    binary_classes = (df[prob_column] > threshold).astype(int)
    counts = binary_classes.value_counts(normalize=True) * 100
    print(f"\n{title}:")
    for label, percentage in counts.items():
        print(f"  {label}: {percentage:.1f}%")
    # Return counts for later use
    return counts

# Base prompt label distributions
print("\n=== BASE PROMPT RESULTS ===")
print_label_distribution(base_df, 'Core_or_General_AI_Application', 'Core (0) vs General AI Application (1)')
print_label_distribution(base_df, 'Is_data_centric', 'Data-centric (1) vs Non-data-centric (0)')
print_label_distribution(base_df, 'Is_niche_or_broad_market', 'Niche (1) vs Broad Market (0)')
print_label_distribution(base_df, 'Is_product_or_platform', 'Product (0) vs Platform (1)')

# Modified prompt - binary classifications from probabilities
print("\n=== MODIFIED PROMPT RESULTS ===")
print_label_distribution(modified_df, 'AI_startup_type', 'AI startup type (1: Hardware, 2: Foundation Model, 3: Application AI, 4: Others)')

# Create binary classifications from probabilities
print("\nModified prompt binary classifications:")
data_centric_counts = print_binary_distribution(modified_df, 'Is_data_centric_probability', 'Data-centric (1) vs Non-data-centric (0)')
niche_market_counts = print_binary_distribution(modified_df, 'Is_niche_or_broad_market_probability', 'Niche (1) vs Broad Market (0)')
platform_counts = print_binary_distribution(modified_df, 'Is_product_or_platform_probability', 'Product (0) vs Platform (1)')

# Also include mean probabilities for reference
print("\nModified prompt mean probabilities (for reference):")
print(f"  Data-centric probability: {modified_df['Is_data_centric_probability'].mean():.2f}")
print(f"  Niche market probability: {modified_df['Is_niche_or_broad_market_probability'].mean():.2f}")
print(f"  Platform probability: {modified_df['Is_product_or_platform_probability'].mean():.2f}")

# Additional prompt - binary classifications
print("\n=== ADDITIONAL PROMPT RESULTS ===")
print("\nIndustry focus classifications:")
vertical_counts = print_binary_distribution(additional_df, 'vertical_ai_startup', 'Vertical AI (1) vs Horizontal AI (0)')
# Horizontal AI is the complement of Vertical AI
print("\nAutomation depth classifications:")
# For automation depth, we need to determine the dominant category for each row
automation_columns = ['full_automation', 'human_in_the_loop', 'recommendation_or_insight_only']
automation_dominant = additional_df[automation_columns].idxmax(axis=1)
automation_counts = automation_dominant.value_counts(normalize=True) * 100
print("\nDominant automation approach:")
for category, percentage in automation_counts.items():
    print(f"  {category}: {percentage:.1f}%")

print("\nProduct nature classifications:")
ai_native_counts = print_binary_distribution(additional_df, 'ai_native', 'AI-native (1) vs AI-augmented (0)')

print("\nModel development classifications:")
model_dev_counts = print_binary_distribution(additional_df, 'model_developer', 'Model developer (1) vs Model integrator (0)')

# Also include mean probabilities for reference
print("\nAdditional prompt mean probabilities (for reference):")
print(f"  Vertical AI probability: {additional_df['vertical_ai_startup'].mean():.2f}")
print(f"  Horizontal AI probability: {additional_df['horizontal_ai_startup'].mean():.2f}")
print(f"  Full automation: {additional_df['full_automation'].mean():.2f}")
print(f"  Human-in-the-loop: {additional_df['human_in_the_loop'].mean():.2f}")
print(f"  Recommendation/insight only: {additional_df['recommendation_or_insight_only'].mean():.2f}")
print(f"  AI-native: {additional_df['ai_native'].mean():.2f}")
print(f"  AI-augmented: {additional_df['ai_augmented'].mean():.2f}")
print(f"  Model developer: {additional_df['model_developer'].mean():.2f}")
print(f"  Model integrator: {additional_df['model_integrator'].mean():.2f}")

# Consistency analysis between prompts
print("\n=== CONSISTENCY ANALYSIS ===")

# Map binary labels for correlation analysis
base_df['data_centric_binary'] = base_df['Is_data_centric']
modified_df['data_centric_binary'] = (modified_df['Is_data_centric_probability'] > 0.5).astype(int)

base_df['niche_binary'] = base_df['Is_niche_or_broad_market']
modified_df['niche_binary'] = (modified_df['Is_niche_or_broad_market_probability'] > 0.5).astype(int)

base_df['platform_binary'] = base_df['Is_product_or_platform']
modified_df['platform_binary'] = (modified_df['Is_product_or_platform_probability'] > 0.5).astype(int)

# Merge datasets for comparison
merged_df = pd.merge(
    base_df[['investee_company_beid', 'data_centric_binary', 'niche_binary', 'platform_binary']], 
    modified_df[['investee_company_beid', 'data_centric_binary', 'niche_binary', 'platform_binary']],
    on='investee_company_beid',
    suffixes=('_base', '_modified')
)

# Calculate agreement percentages
print("\nAgreement between base and modified prompts:")
data_centric_agreement = (merged_df['data_centric_binary_base'] == merged_df['data_centric_binary_modified']).mean() * 100
niche_agreement = (merged_df['niche_binary_base'] == merged_df['niche_binary_modified']).mean() * 100
platform_agreement = (merged_df['platform_binary_base'] == merged_df['platform_binary_modified']).mean() * 100

print(f"  Data-centric agreement: {data_centric_agreement:.1f}%")
print(f"  Niche market agreement: {niche_agreement:.1f}%")
print(f"  Product/Platform agreement: {platform_agreement:.1f}%")

# Output file for charts
print("\nGenerating visualizations...")

# Create visualizations
plt.figure(figsize=(14, 10))

# 1. Base prompt distributions
plt.subplot(2, 2, 1)
base_props = [
    base_df['Core_or_General_AI_Application'].value_counts(normalize=True),
    base_df['Is_data_centric'].value_counts(normalize=True),
    base_df['Is_niche_or_broad_market'].value_counts(normalize=True),
    base_df['Is_product_or_platform'].value_counts(normalize=True)
]

labels = ['Core/General', 'Data-centric', 'Niche/Broad', 'Product/Platform']
x = np.arange(len(labels))
width = 0.35

plt.bar(x - width/2, [p.get(0, 0) for p in base_props], width, label='0')
plt.bar(x + width/2, [p.get(1, 0) for p in base_props], width, label='1')
plt.xlabel('Category')
plt.ylabel('Proportion')
plt.title('Base Prompt Label Distributions')
plt.xticks(x, labels, rotation=45)
plt.legend(title='Label')
plt.tight_layout()

# 2. Modified prompt binary classification distributions
plt.subplot(2, 2, 2)
mod_props = [
    data_centric_counts,
    niche_market_counts,
    platform_counts
]

mod_labels = ['Data-centric', 'Niche/Broad', 'Product/Platform']
x = np.arange(len(mod_labels))

plt.bar(x - width/2, [p.get(0, 0) for p in mod_props], width, label='0')
plt.bar(x + width/2, [p.get(1, 0) for p in mod_props], width, label='1')
plt.xlabel('Category')
plt.ylabel('Proportion')
plt.title('Modified Prompt Binary Classifications')
plt.xticks(x, mod_labels, rotation=45)
plt.legend(title='Label')
plt.tight_layout()

# 3. Additional prompt binary classification distributions
plt.subplot(2, 2, 3)
add_props = [
    vertical_counts,
    model_dev_counts,
    ai_native_counts
]

add_labels = ['Vertical/Horizontal', 'Model Dev/Integ', 'AI-native/Augmented']
x = np.arange(len(add_labels))

plt.bar(x - width/2, [p.get(0, 0) for p in add_props], width, label='0')
plt.bar(x + width/2, [p.get(1, 0) for p in add_props], width, label='1')
plt.xlabel('Category')
plt.ylabel('Proportion')
plt.title('Additional Prompt Binary Classifications')
plt.xticks(x, add_labels, rotation=45)
plt.legend(title='Label')
plt.tight_layout()

# 4. AI Startup Type distribution from modified prompt
plt.subplot(2, 2, 4)
ai_type_counts = modified_df['AI_startup_type'].value_counts(normalize=True) * 100
ai_type_labels = {1: 'Hardware', 2: 'Foundation Model', 3: 'Application AI', 4: 'Others'}
ai_type_dict = {ai_type_labels[k]: v for k, v in ai_type_counts.items()}

plt.bar(ai_type_dict.keys(), ai_type_dict.values())
plt.xlabel('AI Startup Type')
plt.ylabel('Percentage (%)')
plt.title('AI Startup Type Distribution (Modified Prompt)')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the visualization
plt.savefig('prompt_comparison_analysis.png', dpi=300, bbox_inches='tight')
print("Analysis complete. Visualizations saved to 'prompt_comparison_analysis.png'") 