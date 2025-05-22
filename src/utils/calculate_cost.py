import pandas as pd

# Define rates
COMPLETION_TOKEN_RATE = 20.0 / 1_000_000  # $20 per 1M tokens
PROMPT_TOKEN_RATE = 5.0 / 1_000_000      # $5 per 1M tokens

# Read the CSV file
df = pd.read_csv('outputs/merged_data.csv')

# Calculate sum of tokens
total_completion_tokens = df['total_completion_tokens'].sum()
total_prompt_tokens = df['total_prompt_tokens'].sum()

# Calculate cost
completion_cost = total_completion_tokens * COMPLETION_TOKEN_RATE
prompt_cost = total_prompt_tokens * PROMPT_TOKEN_RATE
total_cost = completion_cost + prompt_cost

# Print results
print(f"Total completion tokens: {total_completion_tokens:,}")
print(f"Total prompt tokens: {total_prompt_tokens:,}")
print(f"Cost for completion tokens: ${completion_cost:.2f}")
print(f"Cost for prompt tokens: ${prompt_cost:.2f}")
print(f"Total cost: ${total_cost:.2f}") 