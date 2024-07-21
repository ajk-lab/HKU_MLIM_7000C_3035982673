#%%
import pandas as pd
import matplotlib.pyplot as plt
#%%
# Load your dataset (replace 'your_data_file.xlsx' with your actual file name and path)
file_path = '/Users/ajk/Library/CloudStorage/OneDrive-Personal/ajkdrive/codeLab/MLIM7000/ESGandFinCombinedDatasets/MSCI_INDEX_ALL_ESG_SCORES_2_Sampled_20240628_15_FINAL.csv'
df_sp_fin_data_1 = pd.read_csv(file_path)

#%%
df_sp_fin_data_1.head()

#%%
# Extracting columns that start with 'SP_' for descriptive statistics
sp_columns = [col for col in df_sp_fin_data_1.columns if col.startswith('SP_')]
sp_stats_df = df_sp_fin_data_1[sp_columns].describe()

sp_stats_df

# %%

import pandas as pd
import matplotlib.pyplot as plt

# Select the specified SP columns for descriptive statistics
selected_columns_1 = [
    'SP_FIN_MARKET_CAP',
    'SP_FIN_ENTERPRISE_VALUE',
    'SP_FIN_TOTAL_ASSETS',
    'SP_FIN_TOTAL_EQUITY',
    'SP_FIN_TOTAL_LIABILITIES',
    'SP_FIN_TOTAL_DEBT',
    'SP_FIN_CASH_AND_EQUIVALENTS',
    'SP_FIN_SIZE',
    'SP_FIN_LEVERAGE_RATIO_PERCENT',
    'SP_FIN_DEBT_TO_EQUITY_RATIO_PERCENT',
    'SP_FIN_RETURN_ON_ASSETS_ROA',
    'SP_FIN_RETURN_ON_ASSETS_ROA_PERCENT',
    'SP_FIN_RETURN_ON_EQUITY_PERCENT',
    'SP_FIN_PE_LTM',
    'SP_FIN_TOTAL_ASSETS_CAGR_5Y',
    'SP_FIN_NET_INCOME_IS',
    'SP_FIN_ENTERPRISE_VALUE_TO_ASSET'
]

# Generate descriptive statistics for the selected SP columns
descriptive_stats_selected_1 = df_sp_fin_data_1[selected_columns_1].describe().transpose()
# Save the descriptive statistics to an Excel file
descriptive_stats_selected_1.to_excel('../../output/descriptive_statistics_sp_columns_1.xlsx')
# %%
descriptive_stats_selected_1 
# %%
# Select the specified SP columns for descriptive statistics
selected_columns_2 = [
    'SP_FIN_LEVERAGE_RATIO_PERCENT',
    'SP_FIN_DEBT_TO_EQUITY_RATIO_PERCENT',
    'SP_FIN_RETURN_ON_ASSETS_ROA_PERCENT',
    'SP_FIN_RETURN_ON_EQUITY_PERCENT',
    'SP_FIN_PE_LTM',
    'SP_FIN_ENTERPRISE_VALUE_TO_ASSET'
]

# Generate descriptive statistics for the selected SP columns
descriptive_stats_selected_2 = df_sp_fin_data_1[selected_columns_2].describe().transpose()
# Save the descriptive statistics to an Excel file
descriptive_stats_selected_2.to_excel('../../output/descriptive_statistics_sp_columns_2.xlsx')
# %%
descriptive_stats_selected_2 

# %%
# Generating histograms for the selected SP columns per MSCI_INDEX
import seaborn as sns

# Extracting the necessary columns including MSCI_INDEX
df_histogram_data = df_sp_fin_data_1[selected_columns_2 + ['MSCI_INDEX']]

# Creating histograms for each selected column, grouped by MSCI_INDEX
fig, axes = plt.subplots(len(selected_columns_2), 1, figsize=(12, len(selected_columns_2) * 4), sharex=False)

for i, column in enumerate(selected_columns_2):
    sns.histplot(data=df_histogram_data, x=column, hue='MSCI_INDEX', multiple='stack', kde=True, ax=axes[i])
    axes[i].set_title(f'Histogram of {column} per MSCI_INDEX')
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# %%
# Creating histograms for each selected column without grouping by MSCI_INDEX
fig, axes = plt.subplots(len(selected_columns_2), 1, figsize=(12, len(selected_columns_2) * 4), sharex=False)

for i, column in enumerate(selected_columns_2):
    sns.histplot(data=df_sp_fin_data_1, x=column, kde=True, ax=axes[i])
    axes[i].set_title(f'Histogram of {column}')
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# %%
# Creating box plots for each selected column to show detailed distribution
fig, axes = plt.subplots(len(selected_columns_2), 1, figsize=(12, len(selected_columns_2) * 4), sharex=False)

for i, column in enumerate(selected_columns_2):
    sns.boxplot(data=df_sp_fin_data_1, x=column, ax=axes[i])
    axes[i].set_title(f'Box Plot of {column}')
    axes[i].set_xlabel(column)

plt.tight_layout()
plt.show()

# %%
# Creating both histograms and box plots side by side for each selected column
fig, axes = plt.subplots(len(selected_columns_2), 2, figsize=(18, len(selected_columns_2) * 4), sharex=False)

for i, column in enumerate(selected_columns_2):
    # Histogram
    sns.histplot(data=df_sp_fin_data_1, x=column, kde=True, ax=axes[i, 0])
    axes[i, 0].set_title(f'Histogram of {column}')
    axes[i, 0].set_xlabel(column)
    axes[i, 0].set_ylabel('Frequency')
    
    # Box Plot
    sns.boxplot(data=df_sp_fin_data_1, x=column, ax=axes[i, 1])
    axes[i, 1].set_title(f'Box Plot of {column}')
    axes[i, 1].set_xlabel(column)

plt.tight_layout()
plt.show()

# %%
# Creating both histograms and box plots side by side with contrasting colors for each selected column
fig, axes = plt.subplots(len(selected_columns_2), 2, figsize=(18, len(selected_columns_2) * 4), sharex=False)

for i, column in enumerate(selected_columns_2):
    # Histogram with contrasting color
    sns.histplot(data=df_sp_fin_data_1, x=column, kde=True, ax=axes[i, 0], color='blue')
    axes[i, 0].set_title(f'Histogram of {column}')
    axes[i, 0].set_xlabel(column)
    axes[i, 0].set_ylabel('Frequency')
    
    # Box Plot with contrasting color
    sns.boxplot(data=df_sp_fin_data_1, x=column, ax=axes[i, 1], color='orange')
    axes[i, 1].set_title(f'Box Plot of {column}')
    axes[i, 1].set_xlabel(column)

plt.tight_layout()
plt.show()


# %%
