#%%
import pandas as pd
import matplotlib.pyplot as plt
#%%
# Load your dataset (replace 'your_data_file.xlsx' with your actual file name and path)
file_path = '/Users/ajk/Library/CloudStorage/OneDrive-Personal/ajkdrive/codeLab/MLIM7000/ESGdatasets/MSCI_INDEX_ALL_ESG_SCORES_15JULY24_1240.xlsx'
excel_data = pd.ExcelFile(file_path)

# Display sheet names to understand the structure of the file
sheet_names = excel_data.sheet_names
sheet_names

#%%

data_df1 = pd.read_excel(file_path, sheet_name='Sheet1')
data_df1.head()

#%%

# Display descriptive statistics of the dataset
descriptive_stats = data_df1.describe()
descriptive_stats

#%%
import pandas as pd


# Generate descriptive statistics
descriptive_stats = data_df1.describe()

# Transpose the descriptive statistics for better readability
descriptive_stats_transposed = descriptive_stats.transpose()


descriptive_stats_transposed

#%%
import pandas as pd


# Select the specified columns for descriptive statistics
selected_columns = [
    'INDUSTRY_ADJUSTED_ESG_SCORE',
    'ENVIRONMENTAL_PILLAR_SCORE',
    'SOCIAL_PILLAR_SCORE',
    'GOVERNANCE_PILLAR_SCORE',
    'WATER_STRESS_SCORE',
    'CARBON_EMISSIONS_SCORE',
    'HLTH_SAFETY_SCORE',
    'HUMAN_CAPITAL_DEV_SCORE',
    'CONTROV_SRC_SCORE',
    'CORP_GOVERNANCE_SCORE'
]

# Generate descriptive statistics for the selected columns
descriptive_stats_selected = data_df1[selected_columns].describe()
descriptive_stats_selected = data_df1[selected_columns].describe().transpose()
# Display the descriptive statistics
descriptive_stats_selected
descriptive_stats_selected.to_excel('../../output/descriptive_statistics_1.xlsx')
#%%
import pandas as pd
import matplotlib.pyplot as plt

# Select the specified columns for descriptive statistics
selected_columns = [
    'INDUSTRY_ADJUSTED_ESG_SCORE',
    'ENVIRONMENTAL_PILLAR_SCORE',
    'SOCIAL_PILLAR_SCORE',
    'GOVERNANCE_PILLAR_SCORE',
    'WATER_STRESS_SCORE',
    'CARBON_EMISSIONS_SCORE',
    'HLTH_SAFETY_SCORE',
    'HUMAN_CAPITAL_DEV_SCORE',
    'CONTROV_SRC_SCORE',
    'CORP_GOVERNANCE_SCORE'
]

# Generate descriptive statistics for the selected columns
descriptive_stats_selected = data_df1[selected_columns].describe().transpose()

# Create a beautiful picture (plot) to display the descriptive statistics
fig, ax = plt.subplots(figsize=(15, 8))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=descriptive_stats_selected.values,
                 colLabels=descriptive_stats_selected.columns,
                 rowLabels=descriptive_stats_selected.index,
                 cellLoc='center',
                 loc='center')

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

plt.title('Descriptive Statistics for Selected Columns', fontsize=14)
plt.show()
descriptive_stats_selected.to_excel('../../output/descriptive_statistics_1.xlsx')
#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Select the specified columns for descriptive statistics
selected_columns = [
    'INDUSTRY_ADJUSTED_ESG_SCORE',
    'ENVIRONMENTAL_PILLAR_SCORE',
    'SOCIAL_PILLAR_SCORE',
    'GOVERNANCE_PILLAR_SCORE',
    'WATER_STRESS_SCORE',
    'CARBON_EMISSIONS_SCORE',
    'HLTH_SAFETY_SCORE',
    'HUMAN_CAPITAL_DEV_SCORE',
    'CONTROV_SRC_SCORE',
    'CORP_GOVERNANCE_SCORE'
]

# Generate descriptive statistics for the selected columns
descriptive_stats_selected = data_df1[selected_columns].describe().transpose()

# Create a beautiful picture (plot) to display the descriptive statistics
fig, ax = plt.subplots(figsize=(18, 10))
ax.axis('tight')
ax.axis('off')

# Create a table with the descriptive statistics
table = ax.table(cellText=descriptive_stats_selected.values,
                 colLabels=descriptive_stats_selected.columns,
                 rowLabels=descriptive_stats_selected.index,
                 cellLoc='center',
                 loc='center',
                 colColours=['#f2f2f2']*len(descriptive_stats_selected.columns),
                 rowColours=['#f2f2f2']*len(descriptive_stats_selected))

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

# Add colors to the table
colors = plt.cm.BuGn(np.linspace(0.3, 0.7, len(descriptive_stats_selected.index)))
for i, key in enumerate(descriptive_stats_selected.index):
    table[(i+1, -1)].set_facecolor(colors[i])

# Add title
plt.title('Descriptive Statistics for Selected Columns', fontsize=20, weight='bold')

# Save the figure
plt.savefig('../../output/descriptive_statistics_beautiful.png', bbox_inches='tight', dpi=300)

# Show the plot
plt.show()

#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



# Define the columns to plot
selected_columns = [
    'INDUSTRY_ADJUSTED_ESG_SCORE',
    'ENVIRONMENTAL_PILLAR_SCORE',
    'SOCIAL_PILLAR_SCORE',
    'GOVERNANCE_PILLAR_SCORE',
    'WATER_STRESS_SCORE',
    'CARBON_EMISSIONS_SCORE',
    'HLTH_SAFETY_SCORE',
    'HUMAN_CAPITAL_DEV_SCORE',
    'CONTROV_SRC_SCORE',
    'CORP_GOVERNANCE_SCORE'
]

import matplotlib.pyplot as plt
import numpy as np

# Sample data and selected_columns for demonstration purposes
# data_df1 = your_dataframe_here
# selected_columns = your_selected_columns_here

# Number of rows and columns based on the number of selected columns
n_cols = 3
n_rows = (len(selected_columns) + n_cols - 1) // n_cols  # Calculate the number of rows needed

# Set up the plotting environment
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(9, n_rows * 2))
fig.suptitle('Histogram Distribution of Selected ESG Scores', fontsize=14, weight='bold')

# Define a color palette
colors = plt.cm.viridis(np.linspace(0, 1, len(selected_columns)))

# Plot histograms
for i, column in enumerate(selected_columns):
    ax = axes[i // n_cols, i % n_cols]
    ax.hist(data_df1[column].dropna(), bins=20, color=colors[i], edgecolor='black')
    ax.set_title(column, fontsize=8)
    ax.set_xlabel('Score', fontsize=8)
    ax.set_ylabel('Frequency', fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=6)

# Hide any unused subplots
for j in range(len(selected_columns), n_rows * n_cols):
    fig.delaxes(axes.flatten()[j])

# Adjust layout to prevent overlapping
plt.subplots_adjust(wspace=0.4, hspace=0.8)

# Save the figure
plt.savefig('../../output/histogram_distributions_clear_and_small.png', bbox_inches='tight', dpi=300)

# Show the plot
plt.show()

#%%
# Detailed descriptions for each variable
descriptions = [
    'Index Name',
    'Issuer ID',
    'Issuer Name',
    'Issuer Ticker',
    'Issuer ISIN',
    'Issuer SEDOL',
    'Issuer CUSIP',
    'Issuer Country of Domicile',
    'Issuer Country of Domicile Name',
    'Industry',
    'Corruption Institutional Score',
    'Corruption Institutional Weight',
    'Financial System Institutional Score',
    'Financial System Institutional Weight',
    'Business Ethics and Fraud Score',
    'Business Ethics and Fraud Weight',
    'Anticompetitive Practices Score',
    'Anticompetitive Practices Weight',
    'Corporate Governance Score',
    'Corporate Governance Weight',
    'Board Structure Score',
    'Board Structure Weight',
    'Ownership Score',
    'Ownership Weight',
    'Accounting Score',
    'Accounting Weight',
    'Investor Protection Score',
    'Investor Protection Weight',
    'Stakeholder Governance Score',
    'Stakeholder Governance Weight',
    'Executive Compensation Score',
    'Executive Compensation Weight',
    'Audit Committee Score',
    'Audit Committee Weight',
    'Shareholder Rights Score',
    'Shareholder Rights Weight',
    'Controversies Score',
    'Controversies Weight',
    'Sector Score',
    'Sector Weight',
    'Country Score',
    'Country Weight',
    'Region Score',
    'Region Weight',
    'Overall ESG Score',
    'Overall ESG Weight',
    'Climate Change Score',
    'Climate Change Weight',
    'Environmental Risk Score',
    'Environmental Risk Weight',
    'Water Stress Score',
    'Water Stress Weight',
    'Carbon Emissions Score',
    'Carbon Emissions Weight',
    'Biodiversity Score',
    'Biodiversity Weight',
    'Raw Material Score',
    'Raw Material Weight',
    'Waste Management Score',
    'Waste Management Weight',
    'Energy Management Score',
    'Energy Management Weight',
    'Labor Rights Score',
    'Labor Rights Weight',
    'Health and Safety Score',
    'Health and Safety Weight',
    'Human Capital Development Score',
    'Human Capital Development Weight',
    'Community Relations Score',
    'Community Relations Weight',
    'Product Safety Score',
    'Product Safety Weight',
    'Privacy and Data Security Score',
    'Privacy and Data Security Weight',
    'Political Stability Score',
    'Political Stability Weight',
    'Regulatory Quality Score',
    'Regulatory Quality Weight',
    'Rule of Law Score',
    'Rule of Law Weight',
    'Government Effectiveness Score',
    'Government Effectiveness Weight',
    'Voice Accountability Score',
    'Voice Accountability Weight',
    'Foreign Exchange Score',
    'Foreign Exchange Weight',
    'Sovereign Domestic Credit Score',
    'Sovereign Domestic Credit Weight'
]

# Creating DataFrame for variables and descriptions
variables_descriptions = pd.DataFrame({
    'Variable': data_df1.columns,
    'Description': descriptions
})

# Display the DataFrame to the user
print(variables_descriptions)

variables_descriptions.to_excel('../../output/Variable_Descriptions.xlsx', index=False)


# Calculate the average INDUSTRY_ADJUSTED_ESG_SCORE by MSCI_INDEX group
# First, we need to check if the column 'INDUSTRY_ADJUSTED_ESG_SCORE' exists in the dataframe

if 'INDUSTRY_ADJUSTED_ESG_SCORE' in data_df1.columns:
    # Group by MSCI_INDEX and calculate the mean of INDUSTRY_ADJUSTED_ESG_SCORE
    avg_esg_score_by_index = data_df1.groupby('MSCI_INDEX')['INDUSTRY_ADJUSTED_ESG_SCORE'].mean().reset_index()
    avg_esg_score_by_index.columns = ['MSCI_INDEX', 'Average Industry Adjusted ESG Score']
else:
    avg_esg_score_by_index = pd.DataFrame(columns=['MSCI_INDEX', 'Average Industry Adjusted ESG Score'])


avg_esg_score_by_index

#%%
# Plotting the bar chart with green bars and value labels
plt.figure(figsize=(10, 6))
bars = plt.bar(avg_esg_score_by_index['MSCI_INDEX'], avg_esg_score_by_index['Average Industry Adjusted ESG Score'], color='green')

plt.xlabel('MSCI Index')
plt.ylabel('Average Industry Adjusted ESG Score')
plt.title('Average Industry Adjusted ESG Score by MSCI Index')
plt.xticks(rotation=45)
plt.grid(axis='y')

# Adding value labels on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom')  # va: vertical alignment

# Display the plot
plt.tight_layout()
plt.show()


#%%
# Taking a random sample of 100 from each MSCI_INDEX group and then calculating the average
sampled_df1 = data_df1.groupby('MSCI_INDEX').apply(lambda x: x.sample(min(100, len(x)), random_state=1)).reset_index(drop=True)

# Calculate the average INDUSTRY_ADJUSTED_ESG_SCORE by MSCI_INDEX group from the sampled data
sampled_avg_esg_score_by_index = sampled_df1.groupby('MSCI_INDEX')['INDUSTRY_ADJUSTED_ESG_SCORE'].mean().reset_index()
sampled_avg_esg_score_by_index.columns = ['MSCI_INDEX', 'Average Industry Adjusted ESG Score']

# Plotting the bar chart for the sampled data
plt.figure(figsize=(10, 6))
bars = plt.bar(sampled_avg_esg_score_by_index['MSCI_INDEX'], sampled_avg_esg_score_by_index['Average Industry Adjusted ESG Score'], color='green')

plt.xlabel('MSCI Index')
plt.ylabel('Average Industry Adjusted ESG Score')
plt.title('Average Industry Adjusted ESG Score by MSCI Index (Sampled Data)')
plt.xticks(rotation=45)
plt.grid(axis='y')

# Adding value labels on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom')  # va: vertical alignment

# Display the plot
plt.tight_layout()
plt.show()

# Displaying the sampled average ESG scores to the user

sampled_avg_esg_score_by_index

# %%

# Calculate the count of companies in each MSCI_INDEX group from the sampled data
company_count_by_index = sampled_df1['MSCI_INDEX'].value_counts().reset_index()
company_count_by_index.columns = ['MSCI_INDEX', 'Company Count']

# Merge the average ESG scores with the company count
merged_data = pd.merge(sampled_avg_esg_score_by_index, company_count_by_index, on='MSCI_INDEX')
merged_data
# %%
# Calculate the average INDUSTRY_ADJUSTED_ESG_SCORE by MSCI_INDEX group without sampling
avg_esg_score_by_index_full = data_df1.groupby('MSCI_INDEX')['INDUSTRY_ADJUSTED_ESG_SCORE'].mean().reset_index()
avg_esg_score_by_index_full.columns = ['MSCI_INDEX', 'Average Industry Adjusted ESG Score']

# Calculate the count of companies in each MSCI_INDEX group without sampling
company_count_by_index_full = data_df1['MSCI_INDEX'].value_counts().reset_index()
company_count_by_index_full.columns = ['MSCI_INDEX', 'Company Count']

# Merge the average ESG scores with the company count
merged_data_full = pd.merge(avg_esg_score_by_index_full, company_count_by_index_full, on='MSCI_INDEX')

# Displaying the merged data to the user

merged_data_full

# %%
# First, let's merge the data from both sampled and full datasets
merged_data_sampled = pd.merge(sampled_avg_esg_score_by_index, company_count_by_index, on='MSCI_INDEX')
merged_data_sampled['Type'] = 'Sampled'

merged_data_full = pd.merge(avg_esg_score_by_index_full, company_count_by_index_full, on='MSCI_INDEX')
merged_data_full['Type'] = 'Full'

# Concatenate the dataframes for easy plotting
combined_data = pd.concat([merged_data_sampled, merged_data_full])

# Plotting the bar charts side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

# Plotting the full data
color = 'tab:green'
ax1.set_xlabel('MSCI Index')
ax1.set_ylabel('Average Industry Adjusted ESG Score', color=color)
bars1 = ax1.bar(merged_data_full['MSCI_INDEX'], merged_data_full['Average Industry Adjusted ESG Score'], color=color, alpha=0.7, label='Average ESG Score')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_title('Full Data')

# Adding value labels on top of the bars for average ESG score
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom', color=color)

# Creating a secondary y-axis for the company count in full data
ax1_2 = ax1.twinx()  
color = 'tab:blue'
ax1_2.set_ylabel('Company Count', color=color)
bars2 = ax1_2.bar(merged_data_full['MSCI_INDEX'], merged_data_full['Company Count'], color=color, alpha=0.5, width=0.4, align='center', label='Company Count')
ax1_2.tick_params(axis='y', labelcolor=color)

# Adding value labels on top of the bars for company count
for bar in bars2:
    yval = bar.get_height()
    ax1_2.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom', color=color)

# Plotting the sampled data
color = 'tab:green'
ax2.set_xlabel('MSCI Index')
bars3 = ax2.bar(merged_data_sampled['MSCI_INDEX'], merged_data_sampled['Average Industry Adjusted ESG Score'], color=color, alpha=0.7, label='Average ESG Score')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_title('Sampled Data')

# Adding value labels on top of the bars for average ESG score
for bar in bars3:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom', color=color)

# Creating a secondary y-axis for the company count in sampled data
ax2_2 = ax2.twinx()  
color = 'tab:blue'
ax2_2.set_ylabel('Company Count', color=color)
bars4 = ax2_2.bar(merged_data_sampled['MSCI_INDEX'], merged_data_sampled['Company Count'], color=color, alpha=0.5, width=0.4, align='center', label='Company Count')
ax2_2.tick_params(axis='y', labelcolor=color)

# Adding value labels on top of the bars for company count
for bar in bars4:
    yval = bar.get_height()
    ax2_2.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom', color=color)

# Adding titles and legends
fig.tight_layout()  
fig.suptitle('Comparison of Average ESG Score and Company Count Before and After Sampling', fontsize=16)
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

# Display the plot
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np

# Set the positions and width for the bars
bar_width = 0.35
indices_full = np.arange(len(merged_data_full['MSCI_INDEX']))
indices_sampled = np.arange(len(merged_data_sampled['MSCI_INDEX']))

# Plotting the bar charts side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

# Plotting the full data
color_esg = 'tab:green'
color_count = 'tab:blue'

bars1 = ax1.bar(indices_full, merged_data_full['Average Industry Adjusted ESG Score'], bar_width, label='Average ESG Score', color=color_esg)
bars2 = ax1.bar(indices_full + bar_width, merged_data_full['Company Count'], bar_width, label='Company Count', color=color_count)

ax1.set_xlabel('MSCI Index')
ax1.set_ylabel('Values')
ax1.set_title('Full Data')
ax1.set_xticks(indices_full + bar_width / 2)
ax1.set_xticklabels(merged_data_full['MSCI_INDEX'])

# Adding value labels on top of the bars for full data
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom', ha='center', color=color_esg)

for bar in bars2:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom', ha='center', color=color_count)

ax1.legend()

# Plotting the sampled data
bars3 = ax2.bar(indices_sampled, merged_data_sampled['Average Industry Adjusted ESG Score'], bar_width, label='Average ESG Score', color=color_esg)
bars4 = ax2.bar(indices_sampled + bar_width, merged_data_sampled['Company Count'], bar_width, label='Company Count', color=color_count)

ax2.set_xlabel('MSCI Index')
ax2.set_title('Sampled Data')
ax2.set_xticks(indices_sampled + bar_width / 2)
ax2.set_xticklabels(merged_data_sampled['MSCI_INDEX'])

# Adding value labels on top of the bars for sampled data
for bar in bars3:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom', ha='center', color=color_esg)

for bar in bars4:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom', ha='center', color=color_count)

ax2.legend()

# Adding titles and legends
fig.suptitle('Comparison of Average ESG Score and Company Count Before and After Sampling', fontsize=16)
fig.tight_layout()

# Display the plot
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np

# Set the positions and width for the bars
bar_width = 0.35
indices_full = np.arange(len(merged_data_full['MSCI_INDEX']))
indices_sampled = np.arange(len(merged_data_sampled['MSCI_INDEX']))

# Plotting the bar charts side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# Plotting the full data
color_esg = 'tab:green'
color_count = 'tab:blue'

bars1 = ax1.bar(indices_full, merged_data_full['Average Industry Adjusted ESG Score'], bar_width, label='Average ESG Score', color=color_esg)
ax1_2 = ax1.twinx()
bars2 = ax1_2.bar(indices_full + bar_width, merged_data_full['Company Count'], bar_width, label='Company Count', color=color_count)

ax1.set_xlabel('MSCI Index')
ax1.set_ylabel('Average Industry Adjusted ESG Score', color=color_esg)
ax1.tick_params(axis='y', labelcolor=color_esg)
ax1.set_title('Full Data')
ax1.set_xticks(indices_full + bar_width / 2)
ax1.set_xticklabels(merged_data_full['MSCI_INDEX'], rotation=45, ha='right')

ax1_2.set_ylabel('Company Count', color=color_count)
ax1_2.tick_params(axis='y', labelcolor=color_count)

# Adding value labels on top of the bars for full data
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom', ha='center', color=color_esg)

for bar in bars2:
    yval = bar.get_height()
    ax1_2.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom', ha='center', color=color_count)

ax1.legend(loc='upper left')
ax1_2.legend(loc='upper right')

# Plotting the sampled data
bars3 = ax2.bar(indices_sampled, merged_data_sampled['Average Industry Adjusted ESG Score'], bar_width, label='Average ESG Score', color=color_esg)
ax2_2 = ax2.twinx()
bars4 = ax2_2.bar(indices_sampled + bar_width, merged_data_sampled['Company Count'], bar_width, label='Company Count', color=color_count)

ax2.set_xlabel('MSCI Index')
ax2.set_ylabel('Average Industry Adjusted ESG Score', color=color_esg)
ax2.tick_params(axis='y', labelcolor=color_esg)
ax2.set_title('Sampled Data')
ax2.set_xticks(indices_sampled + bar_width / 2)
ax2.set_xticklabels(merged_data_sampled['MSCI_INDEX'], rotation=45, ha='right')

ax2_2.set_ylabel('Company Count', color=color_count)
ax2_2.tick_params(axis='y', labelcolor=color_count)

# Adding value labels on top of the bars for sampled data
for bar in bars3:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom', ha='center', color=color_esg)

for bar in bars4:
    yval = bar.get_height()
    ax2_2.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom', ha='center', color=color_count)

ax2.legend(loc='upper left')
ax2_2.legend(loc='upper right')

# Adding titles and legends
fig.suptitle('Comparison of Average ESG Score and Company Count Before and After Sampling', fontsize=16)
fig.tight_layout()

# Display the plot
plt.show()

# %%
import pandas as pd

# Define the industry to sector mapping
industry_to_sector = {
    'Road & Rail Transport': 'Transport',
    'Oil & Gas Exploration & Production': 'Energy',
    'Auto Components': 'Automotive',
    'Food Products': 'Consumer Goods',
    'Utilities': 'Utilities',
    'Automobiles': 'Automotive',
    'Banks': 'Financials',
    'Beverages': 'Consumer Goods',
    'Biotechnology': 'Healthcare',
    'Building Products': 'Industrials',
    'Capital Markets': 'Financials',
    'Chemicals': 'Materials',
    'Commercial Services & Supplies': 'Industrials',
    'Communications Equipment': 'Information Technology',
    'Construction & Engineering': 'Industrials',
    'Construction Materials': 'Materials',
    'Consumer Finance': 'Financials',
    'Containers & Packaging': 'Materials',
    'Distributors': 'Consumer Discretionary',
    'Diversified Consumer Services': 'Consumer Discretionary',
    'Diversified Financial Services': 'Financials',
    'Electric Utilities': 'Utilities',
    'Electrical Equipment': 'Industrials',
    'Electronic Equipment Instruments & Components': 'Information Technology',
    'Energy Equipment & Services': 'Energy',
    'Entertainment': 'Communication Services',
    'Equity Real Estate Investment Trusts (REITs)': 'Real Estate',
    'Food & Staples Retailing': 'Consumer Staples',
    'Gas Utilities': 'Utilities',
    'Health Care Equipment & Supplies': 'Healthcare',
    'Health Care Providers & Services': 'Healthcare',
    'Health Care Technology': 'Healthcare',
    'Hotels Restaurants & Leisure': 'Consumer Discretionary',
    'Household Durables': 'Consumer Discretionary',
    'Household Products': 'Consumer Staples',
    'Independent Power and Renewable Electricity Producers': 'Utilities',
    'Industrial Conglomerates': 'Industrials',
    'Insurance': 'Financials',
    'Interactive Media & Services': 'Communication Services',
    'Internet & Direct Marketing Retail': 'Consumer Discretionary',
    'IT Services': 'Information Technology',
    'Leisure Products': 'Consumer Discretionary',
    'Life Sciences Tools & Services': 'Healthcare',
    'Machinery': 'Industrials',
    'Media': 'Communication Services',
    'Metals & Mining': 'Materials',
    'Multi-Utilities': 'Utilities',
    'Oil Gas & Consumable Fuels': 'Energy',
    'Paper & Forest Products': 'Materials',
    'Personal Products': 'Consumer Staples',
    'Pharmaceuticals': 'Healthcare',
    'Professional Services': 'Industrials',
    'Real Estate Management & Development': 'Real Estate',
    'Road & Rail': 'Industrials',
    'Semiconductors & Semiconductor Equipment': 'Information Technology',
    'Software': 'Information Technology',
    'Specialty Retail': 'Consumer Discretionary',
    'Technology Hardware Storage & Peripherals': 'Information Technology',
    'Textiles Apparel & Luxury Goods': 'Consumer Discretionary',
    'Tobacco': 'Consumer Staples',
    'Trading Companies & Distributors': 'Industrials',
    'Transportation Infrastructure': 'Industrials',
    'Water Utilities': 'Utilities',
    # Add additional mappings as needed
}

# Map industries to sectors
data_df1['Sector'] = data_df1['IVA_INDUSTRY'].map(industry_to_sector)

# Handle industries not in the mapping by setting them to 'Other'
data_df1['Sector'] = data_df1['Sector'].fillna('Other')

# Calculate the average INDUSTRY_ADJUSTED_ESG_SCORE by Sector
avg_esg_score_by_sector = data_df1.groupby('Sector')['INDUSTRY_ADJUSTED_ESG_SCORE'].mean().reset_index()
avg_esg_score_by_sector.columns = ['Sector', 'Average Industry Adjusted ESG Score']

# Display the results
print(avg_esg_score_by_sector)


# %%
import matplotlib.pyplot as plt

# Plotting the average ESG scores by sector
plt.figure(figsize=(14, 7))
bars = plt.bar(avg_esg_score_by_sector['Sector'], avg_esg_score_by_sector['Average Industry Adjusted ESG Score'], color='green')

# Adding value labels on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom', ha='center', color='black')

plt.xlabel('Sector')
plt.ylabel('Average Industry Adjusted ESG Score')
plt.title('Average ESG Scores by Sector')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


#%%
# Map industries to sectors for the sampled data
sampled_df1['Sector'] = sampled_df1['IVA_INDUSTRY'].map(industry_to_sector)

# Handle industries not in the mapping by setting them to 'Other'
sampled_df1['Sector'] = sampled_df1['Sector'].fillna('Other')

# Calculate the average INDUSTRY_ADJUSTED_ESG_SCORE by Sector for the sampled data
sampled_avg_esg_score_by_sector = sampled_df1.groupby('Sector')['INDUSTRY_ADJUSTED_ESG_SCORE'].mean().reset_index()
sampled_avg_esg_score_by_sector.columns = ['Sector', 'Average Industry Adjusted ESG Score']

# Plotting the average ESG scores by sector for the sampled data
plt.figure(figsize=(14, 7))
bars = plt.bar(sampled_avg_esg_score_by_sector['Sector'], sampled_avg_esg_score_by_sector['Average Industry Adjusted ESG Score'], color='green')

# Adding value labels on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom', ha='center', color='black')

plt.xlabel('Sector')
plt.ylabel('Average Industry Adjusted ESG Score')
plt.title('Average ESG Scores by Sector (Sampled Data)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %%
import pandas as pd

# Load your data
# data_df = pd.read_excel('MSCI_ALL_ESGRatings_2024_06_24 10_37_28.xlsx', sheet_name='Data')

# Define the industry to sector mapping
industry_to_sector = {
    'Road & Rail Transport': 'Transport',
    'Oil & Gas Exploration & Production': 'Energy',
    'Auto Components': 'Automotive',
    'Food Products': 'Consumer Goods',
    'Utilities': 'Utilities',
    'Automobiles': 'Automotive',
    'Banks': 'Financials',
    'Beverages': 'Consumer Goods',
    'Biotechnology': 'Healthcare',
    'Building Products': 'Industrials',
    'Capital Markets': 'Financials',
    'Chemicals': 'Materials',
    'Commercial Services & Supplies': 'Industrials',
    'Communications Equipment': 'Information Technology',
    'Construction & Engineering': 'Industrials',
    'Construction Materials': 'Materials',
    'Consumer Finance': 'Financials',
    'Containers & Packaging': 'Materials',
    'Distributors': 'Consumer Discretionary',
    'Diversified Consumer Services': 'Consumer Discretionary',
    'Diversified Financial Services': 'Financials',
    'Electric Utilities': 'Utilities',
    'Electrical Equipment': 'Industrials',
    'Electronic Equipment Instruments & Components': 'Information Technology',
    'Energy Equipment & Services': 'Energy',
    'Entertainment': 'Communication Services',
    'Equity Real Estate Investment Trusts (REITs)': 'Real Estate',
    'Food & Staples Retailing': 'Consumer Staples',
    'Gas Utilities': 'Utilities',
    'Health Care Equipment & Supplies': 'Healthcare',
    'Health Care Providers & Services': 'Healthcare',
    'Health Care Technology': 'Healthcare',
    'Hotels Restaurants & Leisure': 'Consumer Discretionary',
    'Household Durables': 'Consumer Discretionary',
    'Household Products': 'Consumer Staples',
    'Independent Power and Renewable Electricity Producers': 'Utilities',
    'Industrial Conglomerates': 'Industrials',
    'Insurance': 'Financials',
    'Interactive Media & Services': 'Communication Services',
    'Internet & Direct Marketing Retail': 'Consumer Discretionary',
    'IT Services': 'Information Technology',
    'Leisure Products': 'Consumer Discretionary',
    'Life Sciences Tools & Services': 'Healthcare',
    'Machinery': 'Industrials',
    'Media': 'Communication Services',
    'Metals & Mining': 'Materials',
    'Multi-Utilities': 'Utilities',
    'Oil Gas & Consumable Fuels': 'Energy',
    'Paper & Forest Products': 'Materials',
    'Personal Products': 'Consumer Staples',
    'Pharmaceuticals': 'Healthcare',
    'Professional Services': 'Industrials',
    'Real Estate Management & Development': 'Real Estate',
    'Road & Rail': 'Industrials',
    'Semiconductors & Semiconductor Equipment': 'Information Technology',
    'Software': 'Information Technology',
    'Specialty Retail': 'Consumer Discretionary',
    'Technology Hardware Storage & Peripherals': 'Information Technology',
    'Textiles Apparel & Luxury Goods': 'Consumer Discretionary',
    'Tobacco': 'Consumer Staples',
    'Trading Companies & Distributors': 'Industrials',
    'Transportation Infrastructure': 'Industrials',
    'Water Utilities': 'Utilities',
    # Add additional mappings as needed
}

# Map industries to sectors
data_df1['Sector'] = data_df1['IVA_INDUSTRY'].map(industry_to_sector)

# Handle industries not in the mapping by setting them to 'Other'
data_df1['Sector'] = data_df1['Sector'].fillna('Other')

# Group by Sector and Region and calculate the average ESG score
avg_esg_score_by_sector_region = data_df1.groupby(['Sector', 'MSCI_INDEX'])['INDUSTRY_ADJUSTED_ESG_SCORE'].mean().reset_index()
avg_esg_score_by_sector_region.columns = ['Sector', 'MSCI_INDEX', 'Average Industry Adjusted ESG Score']

# %%
# Display the results
print(avg_esg_score_by_sector_region)

# Create a pivot table for better readability
pivot_table = avg_esg_score_by_sector_region.pivot(index='Sector', columns='MSCI_INDEX', values='Average Industry Adjusted ESG Score')

# Display the pivot table
print(pivot_table)
#%%
import seaborn as sns
import matplotlib.pyplot as plt

# Create a heatmap
plt.figure(figsize=(16, 10))
sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', linewidths=.5)
plt.title('Average ESG Scores by Sector and Region')
plt.xlabel('Region')
plt.ylabel('Sector')
plt.show()


# %%
# Include INDUSTRY_ADJUSTED_ESG_SCORE in the analysis
analysis_columns_extended = ['MSCI_INDEX', 'ENVIRONMENTAL_PILLAR_SCORE', 'SOCIAL_PILLAR_SCORE', 'GOVERNANCE_PILLAR_SCORE', 'INDUSTRY_ADJUSTED_ESG_SCORE']
analysis_data_extended = data_df1[analysis_columns_extended]

# Convert relevant columns to numeric, forcing errors to NaN
analysis_data_extended['ENVIRONMENTAL_PILLAR_SCORE'] = pd.to_numeric(analysis_data_extended['ENVIRONMENTAL_PILLAR_SCORE'], errors='coerce')
analysis_data_extended['SOCIAL_PILLAR_SCORE'] = pd.to_numeric(analysis_data_extended['SOCIAL_PILLAR_SCORE'], errors='coerce')
analysis_data_extended['GOVERNANCE_PILLAR_SCORE'] = pd.to_numeric(analysis_data_extended['GOVERNANCE_PILLAR_SCORE'], errors='coerce')
analysis_data_extended['INDUSTRY_ADJUSTED_ESG_SCORE'] = pd.to_numeric(analysis_data_extended['INDUSTRY_ADJUSTED_ESG_SCORE'], errors='coerce')

# Calculate average scores by MSCI_INDEX
average_scores_extended = analysis_data_extended.groupby('MSCI_INDEX').mean().reset_index()

# Melt the dataframe to have a 'Score Type' column
melted_data_extended = pd.melt(average_scores_extended, id_vars=['MSCI_INDEX'], 
                      value_vars=['ENVIRONMENTAL_PILLAR_SCORE', 'SOCIAL_PILLAR_SCORE', 'GOVERNANCE_PILLAR_SCORE', 'INDUSTRY_ADJUSTED_ESG_SCORE'],
                      var_name='Score Type', value_name='Score')

# Plot the scores side by side
plt.figure(figsize=(14, 8))
sns.barplot(x='MSCI_INDEX', y='Score', hue='Score Type', data=melted_data_extended, palette="viridis")

plt.title('Average Pillar and Industry Adjusted ESG Scores by MSCI Index')
plt.xlabel('MSCI Index')
plt.ylabel('Average Score')
plt.legend(title='Score Type')

plt.tight_layout()
plt.show()

average_scores_extended

# %%
import seaborn as sns

# Prepare data for heatmap
heatmap_data = average_scores_extended.set_index('MSCI_INDEX').T

# Plot Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", cbar=True)
plt.title('Heatmap of ESG Scores by MSCI Index')
plt.ylabel('Pillar')
plt.xlabel('MSCI Index')
plt.show()

# %%
# Define the categories for the radar chart
categories = ['ENVIRONMENTAL_PILLAR_SCORE', 'SOCIAL_PILLAR_SCORE', 'GOVERNANCE_PILLAR_SCORE']

# Define the number of variables correctly
N = len(categories)

# Prepare data for Radar Chart again to ensure correctness
values_EM = average_scores_extended[average_scores_extended['MSCI_INDEX'] == 'MSCI_EM'][categories].values.flatten().tolist()
values_EUROPE = average_scores_extended[average_scores_extended['MSCI_INDEX'] == 'MSCI_EUROPE'][categories].values.flatten().tolist()
values_US = average_scores_extended[average_scores_extended['MSCI_INDEX'] == 'MSCI_US'][categories].values.flatten().tolist()
values_WORLD = average_scores_extended[average_scores_extended['MSCI_INDEX'] == 'MSCI_WORLD'][categories].values.flatten().tolist()

values_EM += values_EM[:1]
values_EUROPE += values_EUROPE[:1]
values_US += values_US[:1]
values_WORLD += values_WORLD[:1]

# Plot Radar Chart with distinct colors
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Radar Chart
ax1 = plt.subplot(121, polar=True)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

# Plot Radar Chart
ax1.set_theta_offset(np.pi / 2)
ax1.set_theta_direction(-1)

# Draw one axe per variable and add labels
plt.xticks(angles[:-1], categories, color='grey', size=10)

# Draw y-labels
ax1.set_rlabel_position(0)
plt.yticks([2, 4, 6, 8], ["2", "4", "6", "8"], color="grey", size=8)
plt.ylim(0, 8)

# Plot data with distinct colors
ax1.plot(angles, values_EM, linewidth=1, linestyle='solid', label='MSCI_EM', color=colors[0])
ax1.fill(angles, values_EM, colors[0], alpha=0.1)

ax1.plot(angles, values_EUROPE, linewidth=1, linestyle='solid', label='MSCI_EUROPE', color=colors[1])
ax1.fill(angles, values_EUROPE, colors[1], alpha=0.1)

ax1.plot(angles, values_US, linewidth=1, linestyle='solid', label='MSCI_US', color=colors[2])
ax1.fill(angles, values_US, colors[2], alpha=0.1)

ax1.plot(angles, values_WORLD, linewidth=1, linestyle='solid', label='MSCI_WORLD', color=colors[3])
ax1.fill(angles, values_WORLD, colors[3], alpha=0.1)

plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
ax1.set_title('Comparison of ESG Scores by MSCI Index')

# Heatmap
ax2 = plt.subplot(122)
sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", cbar=True, ax=ax2)
ax2.set_title('Heatmap of ESG Scores by MSCI Index')
ax2.set_ylabel('Pillar')
ax2.set_xlabel('MSCI Index')

plt.tight_layout()
plt.show()

# %%
