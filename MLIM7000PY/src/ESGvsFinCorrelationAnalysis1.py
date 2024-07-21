#%%
#******************************************************************
# MSCI_INDUSTRY_ADJUSTED_ESG_SCORE vs SP_FIN_RETURN_ON_ASSETS_ROA
#******************************************************************


import pandas as pd
import matplotlib.pyplot as plt


# Load your dataset
dataset_directory_path='/Users/ajk/Library/CloudStorage/OneDrive-Personal/ajkdrive/codeLab/MLIM7000/ESGandFinCombinedDatasets'
file_path = f'{dataset_directory_path}/MSCI_INDEX_ALL_ESG_SCORES_2_Sampled_20240628_15_FINAL.csv'
file_path
DF_ESG_FIN_1 = pd.read_csv(file_path)
DF_ESG_FIN_1
# Filter datasets for each index
us_data = DF_ESG_FIN_1[DF_ESG_FIN_1['MSCI_INDEX'] == 'MSCI_US']
em_data = DF_ESG_FIN_1[DF_ESG_FIN_1['MSCI_INDEX'] == 'MSCI_EM']
eu_data = DF_ESG_FIN_1[DF_ESG_FIN_1['MSCI_INDEX'] == 'MSCI_EUROPE']
world_data = DF_ESG_FIN_1[DF_ESG_FIN_1['MSCI_INDEX'] == 'MSCI_WORLD']

# %%
# Calculate descriptive statistics for the specified variables
variables_of_interest = [
    'MSCI_INDUSTRY_ADJUSTED_ESG_SCORE',
    'SP_FIN_LEVERAGE_RATIO_PERCENT',
    'SP_FIN_DEBT_TO_EQUITY_RATIO_PERCENT',
    'SP_FIN_RETURN_ON_ASSETS_ROA_PERCENT',
    'SP_FIN_RETURN_ON_EQUITY_PERCENT',
    'SP_FIN_PE_LTM',
    'SP_FIN_ENTERPRISE_VALUE_TO_ASSET',
    'SP_TOBINQ_RATIO'
]

# Calculate the descriptive statistics
descriptive_stats = DF_ESG_FIN_1[variables_of_interest].describe()
descriptive_stats
#%%
# Transpose the descriptive statistics for better readability
descriptive_stats_transposed = descriptive_stats.transpose()
descriptive_stats_transposed

#%%
import seaborn as sns

# Define a function to plot histograms with specified colors
def plot_histogram(data, column, title, xlabel, color, bins=20, xlim=None):
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], bins=bins, kde=True, color=color, edgecolor='black', alpha=0.7)
    plt.title(title, fontsize=15, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
    if xlim:
        plt.xlim(xlim)
    plt.show()

# Colors for the histograms
colors = {
    'MSCI_INDUSTRY_ADJUSTED_ESG_SCORE': '#ff6f61',  # Coral
    'SP_FIN_RETURN_ON_ASSETS_ROA_PERCENT': '#6b5b95',  # Purple
    'SP_FIN_RETURN_ON_EQUITY_PERCENT': '#88b04b',  # Green
    'SP_FIN_PE_LTM': '#ffb347',  # Orange
    'SP_FIN_ENTERPRISE_VALUE_TO_ASSET': '#f7cac9',  # Pink
    'SP_TOBINQ_RATIO': '#92a8d1'  # Light Blue
}

# Plotting histograms for the selected variables with different colors

# Histogram for MSCI Industry Adjusted ESG Score
plot_histogram(DF_ESG_FIN_1, 'MSCI_INDUSTRY_ADJUSTED_ESG_SCORE', 
               'Distribution of MSCI Industry Adjusted ESG Score', 
               'MSCI Industry Adjusted ESG Score', colors['MSCI_INDUSTRY_ADJUSTED_ESG_SCORE'], bins=30)

# Histogram for SP Financial Return on Assets ROA Percent
plot_histogram(DF_ESG_FIN_1, 'SP_FIN_RETURN_ON_ASSETS_ROA_PERCENT', 
               'Distribution of SP Financial Return on Assets ROA Percent', 
               'SP Financial Return on Assets ROA Percent', colors['SP_FIN_RETURN_ON_ASSETS_ROA_PERCENT'], bins=30)

# Histogram for SP Financial Return on Equity Percent with a wider range and more bins
plot_histogram(DF_ESG_FIN_1, 'SP_FIN_RETURN_ON_EQUITY_PERCENT', 
               'Distribution of SP Financial Return on Equity Percent', 
               'SP Financial Return on Equity Percent', colors['SP_FIN_RETURN_ON_EQUITY_PERCENT'], bins=30)

# Histogram for SP Financial PE LTM
plot_histogram(DF_ESG_FIN_1, 'SP_FIN_PE_LTM', 
               'Distribution of SP Financial PE LTM', 
               'SP Financial PE LTM', colors['SP_FIN_PE_LTM'], bins=30)

# Histogram for SP Financial Enterprise Value to Asset
plot_histogram(DF_ESG_FIN_1, 'SP_FIN_ENTERPRISE_VALUE_TO_ASSET', 
               'Distribution of SP Financial Enterprise Value to Asset', 
               'SP Financial Enterprise Value to Asset', colors['SP_FIN_ENTERPRISE_VALUE_TO_ASSET'], bins=30)

# Histogram for SP Tobin's Q Ratio
plot_histogram(DF_ESG_FIN_1, 'SP_TOBINQ_RATIO', 
               'Distribution of SP Tobin\'s Q Ratio', 
               'SP Tobin\'s Q Ratio', colors['SP_TOBINQ_RATIO'], bins=30)


#%%
# Define a function to plot box plots with specified colors
def plot_boxplot(data, column, title, xlabel, color):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=data[column], color=color)
    plt.title(title, fontsize=15, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12, fontweight='bold')
    plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
    plt.show()

# Colors for the plots
colors = {
    'MSCI_INDUSTRY_ADJUSTED_ESG_SCORE': '#ff6f61',  # Coral
    'SP_FIN_RETURN_ON_ASSETS_ROA_PERCENT': '#6b5b95',  # Purple
    'SP_FIN_RETURN_ON_EQUITY_PERCENT': '#88b04b',  # Green
    'SP_FIN_PE_LTM': '#ffb347',  # Orange
    'SP_FIN_ENTERPRISE_VALUE_TO_ASSET': '#f7cac9',  # Pink
    'SP_TOBINQ_RATIO': '#92a8d1'  # Light Blue
}

# Plotting histograms and box plots for the selected variables with different colors

# Histogram and Box Plot for MSCI Industry Adjusted ESG Score
plot_histogram(DF_ESG_FIN_1, 'MSCI_INDUSTRY_ADJUSTED_ESG_SCORE', 
               'Distribution of MSCI Industry Adjusted ESG Score', 
               'MSCI Industry Adjusted ESG Score', colors['MSCI_INDUSTRY_ADJUSTED_ESG_SCORE'], bins=30)
plot_boxplot(DF_ESG_FIN_1, 'MSCI_INDUSTRY_ADJUSTED_ESG_SCORE', 
             'Box Plot of MSCI Industry Adjusted ESG Score', 
             'MSCI Industry Adjusted ESG Score', colors['MSCI_INDUSTRY_ADJUSTED_ESG_SCORE'])

# Histogram and Box Plot for SP Financial Return on Assets ROA Percent
plot_histogram(DF_ESG_FIN_1, 'SP_FIN_RETURN_ON_ASSETS_ROA_PERCENT', 
               'Distribution of SP Financial Return on Assets ROA Percent', 
               'SP Financial Return on Assets ROA Percent', colors['SP_FIN_RETURN_ON_ASSETS_ROA_PERCENT'], bins=30)
plot_boxplot(DF_ESG_FIN_1, 'SP_FIN_RETURN_ON_ASSETS_ROA_PERCENT', 
             'Box Plot of SP Financial Return on Assets ROA Percent', 
             'SP Financial Return on Assets ROA Percent', colors['SP_FIN_RETURN_ON_ASSETS_ROA_PERCENT'])

# Histogram and Box Plot for SP Financial Return on Equity Percent
plot_histogram(DF_ESG_FIN_1, 'SP_FIN_RETURN_ON_EQUITY_PERCENT', 
               'Distribution of SP Financial Return on Equity Percent', 
               'SP Financial Return on Equity Percent', colors['SP_FIN_RETURN_ON_EQUITY_PERCENT'], bins=30)
plot_boxplot(DF_ESG_FIN_1, 'SP_FIN_RETURN_ON_EQUITY_PERCENT', 
             'Box Plot of SP Financial Return on Equity Percent', 
             'SP Financial Return on Equity Percent', colors['SP_FIN_RETURN_ON_EQUITY_PERCENT'])

# Histogram and Box Plot for SP Financial PE LTM
plot_histogram(DF_ESG_FIN_1, 'SP_FIN_PE_LTM', 
               'Distribution of SP Financial PE LTM', 
               'SP Financial PE LTM', colors['SP_FIN_PE_LTM'], bins=30)
plot_boxplot(DF_ESG_FIN_1, 'SP_FIN_PE_LTM', 
             'Box Plot of SP Financial PE LTM', 
             'SP Financial PE LTM', colors['SP_FIN_PE_LTM'])

# Histogram and Box Plot for SP Financial Enterprise Value to Asset
plot_histogram(DF_ESG_FIN_1, 'SP_FIN_ENTERPRISE_VALUE_TO_ASSET', 
               'Distribution of SP Financial Enterprise Value to Asset', 
               'SP Financial Enterprise Value to Asset', colors['SP_FIN_ENTERPRISE_VALUE_TO_ASSET'], bins=30)
plot_boxplot(DF_ESG_FIN_1, 'SP_FIN_ENTERPRISE_VALUE_TO_ASSET', 
             'Box Plot of SP Financial Enterprise Value to Asset', 
             'SP Financial Enterprise Value to Asset', colors['SP_FIN_ENTERPRISE_VALUE_TO_ASSET'])

# Histogram and Box Plot for SP Tobin's Q Ratio
plot_histogram(DF_ESG_FIN_1, 'SP_TOBINQ_RATIO', 
               'Distribution of SP Tobin\'s Q Ratio', 
               'SP Tobin\'s Q Ratio', colors['SP_TOBINQ_RATIO'], bins=30)
plot_boxplot(DF_ESG_FIN_1, 'SP_TOBINQ_RATIO', 
             'Box Plot of SP Tobin\'s Q Ratio', 
             'SP Tobin\'s Q Ratio', colors['SP_TOBINQ_RATIO'])

#%%
#%%
import pandas as pd
from scipy.stats import shapiro

# Load the CSV file
#file_path = '/mnt/data/MSCI_INDEX_ALL_ESG_SCORES_2_Sampled_20240628_15_FINAL.csv'
#DF_ESG_FIN_1 = pd.read_csv(file_path)

# Variables of interest
variables_of_interest = [
    'MSCI_INDUSTRY_ADJUSTED_ESG_SCORE',
    'SP_FIN_LEVERAGE_RATIO_PERCENT',
    'SP_FIN_DEBT_TO_EQUITY_RATIO_PERCENT',
    'SP_FIN_RETURN_ON_ASSETS_ROA_PERCENT',
    'SP_FIN_RETURN_ON_EQUITY_PERCENT',
    'SP_FIN_PE_LTM',
    'SP_FIN_ENTERPRISE_VALUE_TO_ASSET',
    'SP_TOBINQ_RATIO'
]

# Perform the Shapiro-Wilk test and collect results
shapiro_results = {var: shapiro(DF_ESG_FIN_1[var].dropna()) for var in variables_of_interest}

# Create a DataFrame to display results
shapiro_table = pd.DataFrame({
    'Variable': list(shapiro_results.keys()),
    'W Statistic': [result.statistic for result in shapiro_results.values()],
    'p-value': [result.pvalue for result in shapiro_results.values()]
})

# Save the table to a CSV file
output_path = '../../output/shapiro_wilk_test_results.csv'
shapiro_table.to_csv(output_path, index=False)

output_path

#%%
import pandas as pd
from scipy.stats import shapiro

# Load the CSV file
#file_path = 'path_to_your_file/MSCI_INDEX_ALL_ESG_SCORES_2_Sampled_20240628_15_FINAL.csv'
#DF_ESG_FIN_1 = pd.read_csv(file_path)

# Variables of interest
variables_of_interest = [
    'MSCI_INDUSTRY_ADJUSTED_ESG_SCORE',
    'SP_FIN_RETURN_ON_ASSETS_ROA_PERCENT',
    'SP_FIN_RETURN_ON_EQUITY_PERCENT',
    'SP_FIN_PE_LTM',
    'SP_FIN_ENTERPRISE_VALUE_TO_ASSET',
    'SP_TOBINQ_RATIO'
]

# Perform the Shapiro-Wilk test by MSCI index
index_groups = DF_ESG_FIN_1.groupby('MSCI_INDEX')

# Initialize a dictionary to store the results for each index
shapiro_results_by_index = {}

# Perform the Shapiro-Wilk test for each index group and each variable of interest
for index, group in index_groups:
    shapiro_results_by_index[index] = {var: shapiro(group[var].dropna()) for var in variables_of_interest}

# Create a DataFrame to display results for each index
shapiro_results_combined = []

for index, results in shapiro_results_by_index.items():
    for var, result in results.items():
        shapiro_results_combined.append({
            'Index': index,
            'Variable': var,
            'W Statistic': result.statistic,
            'p-value': result.pvalue
        })

shapiro_results_df = pd.DataFrame(shapiro_results_combined)

# Save the table to an Excel file
output_excel_path = '../../output/shapiro_wilk_test_results_by_index.xlsx'
shapiro_results_df.to_excel(output_excel_path, index=False)

print(f"Shapiro-Wilk test results saved to {output_excel_path}")
shapiro_results_df

#%%
# Define the new variables of interest with short names for better visualization
short_name_mapping = {
    'MSCI_INDUSTRY_ADJUSTED_ESG_SCORE': 'ESG Score',
    'SP_FIN_RETURN_ON_ASSETS_ROA_PERCENT': 'ROA',
    'SP_FIN_RETURN_ON_EQUITY_PERCENT': 'ROE',
    'SP_FIN_PE_LTM': 'PE',
    'SP_FIN_ENTERPRISE_VALUE_TO_ASSET': 'EV',
    'SP_TOBINQ_RATIO': 'Tobin\'s Q Ratio'
}

# Rename the columns in the dataframe
DF_ESG_FIN_1_short = DF_ESG_FIN_1.rename(columns=short_name_mapping)

# Variables of interest with short names
new_variables_of_interest_short = list(short_name_mapping.values())

# Create a heatmap for each MSCI index in separate charts with a more contrasting color scheme and bigger font
index_groups = DF_ESG_FIN_1_short.groupby('MSCI_INDEX')

# Improved contrasting color scheme for heatmaps
contrast_colors = sns.diverging_palette(240, 10, as_cmap=True)

for index, group in index_groups:
    correlation_matrix = group[new_variables_of_interest_short].corr()
    
    # Set up the matplotlib figure
    plt.figure(figsize=(12, 10))
    
    # Draw the heatmap with the new correlation matrix
    sns.heatmap(correlation_matrix, annot=True, cmap=contrast_colors, vmin=-1, vmax=1, center=0, linewidths=.5, fmt=".2f", annot_kws={"size": 12})
    
    # Set title with bigger font
    plt.title(f'Correlation Matrix Heatmap for {index}', fontsize=20)
    
    # Show the plot
    plt.show()


#%%
    
    # Calculate the Spearman correlation matrix for the new variables of interest
    for index, group in index_groups:
        spearman_correlation_matrix = group[new_variables_of_interest_short].corr(method='spearman')
        
        # Set up the matplotlib figure
        plt.figure(figsize=(12, 10))
        
        # Draw the heatmap with the new correlation matrix using a more contrasting color palette
        sns.heatmap(spearman_correlation_matrix, annot=True, cmap=contrast_colors, vmin=-1, vmax=1, center=0, linewidths=.5, fmt=".2f", annot_kws={"size": 12})
        
        # Set title with bigger font
        plt.title(f'Spearman Correlation Matrix Heatmap for {index}', fontsize=20)
        
        # Show the plot
        plt.show()
    
#%%
from scipy.stats import spearmanr

# Perform the Spearman correlation test specifically between ESG score and ROA for each MSCI index
spearman_test_results = {}

for index, group in index_groups:
    corr, p_value = spearmanr(group['ESG Score'], group['ROA'])
    spearman_test_results[index] = {'Spearman Correlation': corr, 'p-value': p_value}

# Convert the results to a DataFrame for better readability
spearman_test_results_df = pd.DataFrame.from_dict(spearman_test_results, orient='index').reset_index()
spearman_test_results_df.columns = ['MSCI_INDEX', 'Spearman Correlation', 'p-value']


spearman_test_results_df
      
#%%
# Map the MSCI indices to their geographical regions
geo_labels = {
    'MSCI_EM': 'Emerging Markets',
    'MSCI_EUROPE': 'Europe',
    'MSCI_US': 'United States',
    'MSCI_WORLD': 'World'
}

# Update the MSCI_INDEX column in the dataframe with geographical labels
spearman_test_results_df['Geographical Region'] = spearman_test_results_df['MSCI_INDEX'].map(geo_labels)

# Correct approach to set bar colors for the Seaborn barplot with hue
fig, ax1 = plt.subplots(figsize=(12, 8))

# Create a new column for color labels
spearman_test_results_df['color_label'] = ['Statistically Significant' if p < 0.05 else 'Not Significant' for p in spearman_test_results_df['p-value']]

# Bar plot for Spearman Correlation with conditional colors using hue
bars = sns.barplot(x='Geographical Region', y='Spearman Correlation', hue='color_label', dodge=False, data=spearman_test_results_df, ax=ax1, palette={'Statistically Significant': 'green', 'Not Significant': 'red'})
ax1.axhline(0, color='gray', linewidth=0.8)
ax1.set_ylabel('Spearman Correlation', fontsize=14)
ax1.set_title('Spearman Correlation and p-values (ESG Score vs ROA) by Geographical Region', fontsize=16)

# Add labels to the bars
for bar in bars.patches:
    height = bar.get_height()
    ax1.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), 
                 textcoords='offset points', ha='center', va='bottom', fontsize=12, color='black')

# Create a twin axis for p-values
ax2 = ax1.twinx()
line = sns.lineplot(x='Geographical Region', y='p-value', data=spearman_test_results_df, color='blue', marker='o', linewidth=2, ax=ax2)
ax2.axhline(0.05, color='gray', linestyle='--', linewidth=0.8)
ax2.set_ylabel('p-value', fontsize=14)
ax2.set_ylim(0, 1)

# Add labels to the line plot
for i in range(len(spearman_test_results_df)):
    ax2.annotate(f"{spearman_test_results_df['p-value'][i]:.2f}", 
                 (spearman_test_results_df['Geographical Region'][i], spearman_test_results_df['p-value'][i]),
                 textcoords="offset points", xytext=(0, 10), ha='center', fontsize=12, color='black')

plt.tight_layout()
plt.show()


#%%
# Correct the approach to ensure the dataframe has the correct variables
# Update variables of interest for ROA analysis
variables_of_interest_roa = ['MSCI_INDUSTRY_ADJUSTED_ESG_SCORE', 'SP_FIN_RETURN_ON_ASSETS_ROA_PERCENT']

# Create short names for readability
short_name_mapping_roa = {
    'MSCI_INDUSTRY_ADJUSTED_ESG_SCORE': 'ESG Score',
    'SP_FIN_RETURN_ON_ASSETS_ROA_PERCENT': 'ROA'
}

# Rename the columns in the dataframe
DF_ESG_FIN_1_short_roa = DF_ESG_FIN_1.rename(columns=short_name_mapping_roa)

# Filter datasets for each index
us_data = DF_ESG_FIN_1_short_roa[DF_ESG_FIN_1_short_roa['MSCI_INDEX'] == 'MSCI_US']
em_data = DF_ESG_FIN_1_short_roa[DF_ESG_FIN_1_short_roa['MSCI_INDEX'] == 'MSCI_EM']
eu_data = DF_ESG_FIN_1_short_roa[DF_ESG_FIN_1_short_roa['MSCI_INDEX'] == 'MSCI_EUROPE']
world_data = DF_ESG_FIN_1_short_roa[DF_ESG_FIN_1_short_roa['MSCI_INDEX'] == 'MSCI_WORLD']

# Perform the Spearman correlation test specifically between ESG score and ROA for each MSCI index
spearman_test_results_roa = {}

for index, group in DF_ESG_FIN_1_short_roa.groupby('MSCI_INDEX'):
    corr, p_value = spearmanr(group['ESG Score'], group['ROA'])
    spearman_test_results_roa[index] = {'Spearman Correlation': corr, 'p-value': p_value}

# Convert the results to a DataFrame for better readability
spearman_test_results_roa_df = pd.DataFrame.from_dict(spearman_test_results_roa, orient='index').reset_index()
spearman_test_results_roa_df.columns = ['MSCI_INDEX', 'Spearman Correlation', 'p-value']

# Map the MSCI indices to their geographical regions
spearman_test_results_roa_df['Geographical Region'] = spearman_test_results_roa_df['MSCI_INDEX'].map(geo_labels)

# Create a new column for color labels
spearman_test_results_roa_df['Insight'] = ['Statistically Significant' if p < 0.05 else 'Not Significant' for p in spearman_test_results_roa_df['p-value']]


# Save the table to a CSV file
output_csv_path_roa = '../../output/spearman_test_results_esg_vs_roa.csv'
spearman_test_results_roa_df.to_csv(output_csv_path_roa, index=False)

output_csv_path_roa

spearman_test_results_roa_df


#%%
# Correct the dataset to include the necessary variables for ESG Score vs ROE analysis
variables_of_interest_roe = ['MSCI_INDUSTRY_ADJUSTED_ESG_SCORE', 'SP_FIN_RETURN_ON_EQUITY_PERCENT']

# Create short names for readability
short_name_mapping_roe = {
    'MSCI_INDUSTRY_ADJUSTED_ESG_SCORE': 'ESG Score',
    'SP_FIN_RETURN_ON_EQUITY_PERCENT': 'ROE'
}

# Rename the columns in the dataframe
DF_ESG_FIN_1_short_roe = DF_ESG_FIN_1.rename(columns=short_name_mapping_roe)

# Perform the Spearman correlation test specifically between ESG score and ROE for each MSCI index
spearman_test_results_roe = {}

for index, group in DF_ESG_FIN_1_short_roe.groupby('MSCI_INDEX'):
    corr, p_value = spearmanr(group['ESG Score'], group['ROE'])
    spearman_test_results_roe[index] = {'Spearman Correlation': corr, 'p-value': p_value}

# Convert the results to a DataFrame for better readability
spearman_test_results_roe_df = pd.DataFrame.from_dict(spearman_test_results_roe, orient='index').reset_index()
spearman_test_results_roe_df.columns = ['MSCI_INDEX', 'Spearman Correlation', 'p-value']

# Create the formatted table for the Spearman correlation test results between ESG Score and ROE
formatted_table_roe = pd.DataFrame({
    'MSCI_INDEX': spearman_test_results_roe_df['MSCI_INDEX'],
    'Spearman Correlation': spearman_test_results_roe_df['Spearman Correlation'],
    'p-value': spearman_test_results_roe_df['p-value'],
    'Significance': ['Significant' if p < 0.05 else 'Not significant' for p in spearman_test_results_roe_df['p-value']],
    'Insight': [
        'Weak but statistically significant positive correlation' if (corr > 0 and p < 0.05) else
        'Weak positive correlation, statistically significant' if (corr > 0 and p < 0.05) else
        'Very weak and not statistically significant correlation' for corr, p in zip(spearman_test_results_roe_df['Spearman Correlation'], spearman_test_results_roe_df['p-value'])
    ]
})

# Save the formatted table to an Excel file
output_excel_path_roe = '../../output/formatted_spearman_test_results_esg_vs_roe.xlsx'
formatted_table_roe.to_excel(output_excel_path_roe, index=False)

output_excel_path_roe

formatted_table_roe

#%%
# Create a bar plot for the Spearman correlation and p-values for ESG Score vs ROE by Geographical Region
fig, ax1 = plt.subplots(figsize=(12, 8))

# Bar plot for Spearman Correlation with conditional colors
bars = sns.barplot(x='MSCI_INDEX', y='Spearman Correlation', hue='Significance', dodge=False, data=formatted_table_roe, palette={'Significant': 'green', 'Not significant': 'red'}, ax=ax1)
ax1.axhline(0, color='gray', linewidth=0.8)
ax1.set_ylabel('Spearman Correlation', fontsize=14)
ax1.set_title('Spearman Correlation and p-values (ESG Score vs ROE) by Geographical Region', fontsize=16)

# Add labels to the bars
for bar in bars.patches:
    height = bar.get_height()
    ax1.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), 
                 textcoords='offset points', ha='center', va='bottom', fontsize=12, color='black')

# Create a twin axis for p-values
ax2 = ax1.twinx()
line = sns.lineplot(x='MSCI_INDEX', y='p-value', data=formatted_table_roe, color='blue', marker='o', linewidth=2, ax=ax2)
ax2.axhline(0.05, color='gray', linestyle='--', linewidth=0.8)
ax2.set_ylabel('p-value', fontsize=14)
ax2.set_ylim(0, 1)

# Add labels to the line plot
for i in range(len(formatted_table_roe)):
    ax2.annotate(f"{formatted_table_roe['p-value'][i]:.2f}", 
                 (formatted_table_roe['MSCI_INDEX'][i], formatted_table_roe['p-value'][i]),
                 textcoords="offset points", xytext=(0, 10), ha='center', fontsize=12, color='black')

plt.tight_layout()
plt.show()

#%%


import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import seaborn as sns

# Load the data
#file_path = '/mnt/data/MSCI_INDEX_ALL_ESG_SCORES_2_Sampled_20240628_15_FINAL.csv'
#DF_ESG_FIN_1 = pd.read_csv(file_path)

# Variables of interest for ROA analysis
variables_of_interest_roa = ['MSCI_INDUSTRY_ADJUSTED_ESG_SCORE', 'SP_FIN_RETURN_ON_ASSETS_ROA_PERCENT']

# Create short names for readability
short_name_mapping_roa = {
    'MSCI_INDUSTRY_ADJUSTED_ESG_SCORE': 'ESG Score',
    'SP_FIN_RETURN_ON_ASSETS_ROA_PERCENT': 'ROA'
}

# Rename the columns in the dataframe
DF_ESG_FIN_1_short_roa = DF_ESG_FIN_1.rename(columns=short_name_mapping_roa)

# Define colors for each MSCI index
colors = {
    'MSCI_EM': 'blue',
    'MSCI_EUROPE': 'orange',
    'MSCI_US': 'green',
    'MSCI_WORLD': 'purple'
}

# Define a function to create scatter plots and histograms in one row with same color scheme for each row, and add labels
def plot_scatter_hist_row_color_labels(data, index_name, var1, var2, short_var1, short_var2, color):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Scatter plot
    scatter = axs[0].scatter(data[var1], data[var2], alpha=0.6, c=color, edgecolors='w', linewidth=0.5)
    axs[0].set_title(f'Scatter Plot of {short_var1} vs {short_var2} for {index_name}', fontsize=14)
    axs[0].set_xlabel(short_var1, fontsize=12)
    axs[0].set_ylabel(short_var2, fontsize=12)
    axs[0].grid(True)

    # Histogram for var1
    axs[1].hist(data[var1], bins=30, alpha=0.7, edgecolor='black', color=color)
    axs[1].set_title(f'Histogram of {short_var1} for {index_name}', fontsize=14)
    axs[1].set_xlabel(short_var1, fontsize=12)
    axs[1].set_ylabel('Frequency', fontsize=12)
    axs[1].grid(True)

    # Histogram for var2
    axs[2].hist(data[var2], bins=30, alpha=0.7, edgecolor='black', color=color)
    axs[2].set_title(f'Histogram of {short_var2} for {index_name}', fontsize=14)
    axs[2].set_xlabel(short_var2, fontsize=12)
    axs[2].set_ylabel('Frequency', fontsize=12)
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

# Perform the plots for each MSCI index
for index, group in DF_ESG_FIN_1_short_roa.groupby('MSCI_INDEX'):
    plot_scatter_hist_row_color_labels(group, index, 'ESG Score', 'ROA', 'ESG Score', 'ROA', colors[index])

#%%

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import seaborn as sns

# Load the data
#file_path = '/mnt/data/MSCI_INDEX_ALL_ESG_SCORES_2_Sampled_20240628_15_FINAL.csv'
#DF_ESG_FIN_1 = pd.read_csv(file_path)

# Variables of interest for ROE analysis
variables_of_interest_roe = ['MSCI_INDUSTRY_ADJUSTED_ESG_SCORE', 'SP_FIN_RETURN_ON_EQUITY_PERCENT']

# Create short names for readability
short_name_mapping_roe = {
    'MSCI_INDUSTRY_ADJUSTED_ESG_SCORE': 'ESG Score',
    'SP_FIN_RETURN_ON_EQUITY_PERCENT': 'ROE'
}

# Rename the columns in the dataframe
DF_ESG_FIN_1_short_roe = DF_ESG_FIN_1.rename(columns=short_name_mapping_roe)

# Define colors for each MSCI index
colors = {
    'MSCI_EM': 'blue',
    'MSCI_EUROPE': 'orange',
    'MSCI_US': 'green',
    'MSCI_WORLD': 'purple'
}

# Define a function to create scatter plots and histograms in one row with same color scheme for each row, and add labels
def plot_scatter_hist_row_color_labels(data, index_name, var1, var2, short_var1, short_var2, color):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Scatter plot
    scatter = axs[0].scatter(data[var1], data[var2], alpha=0.6, c=color, edgecolors='w', linewidth=0.5)
    axs[0].set_title(f'Scatter Plot of {short_var1} vs {short_var2} for {index_name}', fontsize=14)
    axs[0].set_xlabel(short_var1, fontsize=12)
    axs[0].set_ylabel(short_var2, fontsize=12)
    axs[0].grid(True)

    # Histogram for var1
    axs[1].hist(data[var1], bins=30, alpha=0.7, edgecolor='black', color=color)
    axs[1].set_title(f'Histogram of {short_var1} for {index_name}', fontsize=14)
    axs[1].set_xlabel(short_var1, fontsize=12)
    axs[1].set_ylabel('Frequency', fontsize=12)
    axs[1].grid(True)

    # Histogram for var2
    axs[2].hist(data[var2], bins=30, alpha=0.7, edgecolor='black', color=color)
    axs[2].set_title(f'Histogram of {short_var2} for {index_name}', fontsize=14)
    axs[2].set_xlabel(short_var2, fontsize=12)
    axs[2].set_ylabel('Frequency', fontsize=12)
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

# Perform the plots for each MSCI index
for index, group in DF_ESG_FIN_1_short_roe.groupby('MSCI_INDEX'):
    plot_scatter_hist_row_color_labels(group, index, 'ESG Score', 'ROE', 'ESG Score', 'ROE', colors[index])


#%%
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from sklearn.metrics import r2_score
    from statsmodels.robust.robust_linear_model import RLM
    
    # Load the data
    #file_path = '/mnt/data/MSCI_INDEX_ALL_ESG_SCORES_2_Sampled_20240628_15_FINAL.csv'
    #DF_ESG_FIN_1 = pd.read_csv(file_path)
    
    # Rename the necessary columns for clarity
    DF_ESG_FIN_1_short_roa = DF_ESG_FIN_1.rename(columns={
        'MSCI_INDUSTRY_ADJUSTED_ESG_SCORE': 'ESG_Score',
        'SP_FIN_RETURN_ON_ASSETS_ROA_PERCENT': 'ROA'
    })
    
    # Function to fit polynomial regression
    def fit_polynomial_regression(data, degree):
        poly_features = np.vander(data['ESG_Score'], N=degree + 1, increasing=True)
        model = sm.OLS(data['ROA'], poly_features).fit()
        predictions = model.predict(poly_features)
        r2 = r2_score(data['ROA'], predictions)
        return model, predictions, r2
    
    # Function to fit logarithmic regression
    def fit_logarithmic_regression(data):
        data['log_ESG_Score'] = np.log(data['ESG_Score'])
        model = sm.OLS(data['ROA'], sm.add_constant(data['log_ESG_Score'])).fit()
        predictions = model.predict(sm.add_constant(data['log_ESG_Score']))
        r2 = r2_score(data['ROA'], predictions)
        return model, predictions, r2
    
    # Function to fit exponential regression
    def fit_exponential_regression(data):
        data['log_ROA'] = np.log(data['ROA'])
        model = sm.OLS(data['log_ROA'], sm.add_constant(data['ESG_Score'])).fit()
        predictions = np.exp(model.predict(sm.add_constant(data['ESG_Score'])))
        r2 = r2_score(data['ROA'], predictions)
        return model, predictions, r2
    
    # Function to fit robust regression
    def fit_robust_regression(data):
        model = RLM(data['ROA'], sm.add_constant(data['ESG_Score'])).fit()
        predictions = model.predict(sm.add_constant(data['ESG_Score']))
        r2 = r2_score(data['ROA'], predictions)
        return model, predictions, r2
    
    # Function to perform and compare non-linear models for each MSCI index
    def compare_non_linear_models(data, index_name):
        results = {}
    
        # Fit polynomial regression
        poly_model, poly_predictions, poly_r2 = fit_polynomial_regression(data, degree=2)
        results['Polynomial Regression'] = poly_r2
    
        # Fit logarithmic regression
        log_model, log_predictions, log_r2 = fit_logarithmic_regression(data[data['ESG_Score'] > 0].copy())
        results['Logarithmic Regression'] = log_r2
    
        # Fit exponential regression
        exp_model, exp_predictions, exp_r2 = fit_exponential_regression(data[data['ROA'] > 0].copy())
        results['Exponential Regression'] = exp_r2
    
        # Fit robust regression
        robust_model, robust_predictions, robust_r2 = fit_robust_regression(data)
        results['Robust Regression'] = robust_r2
    
        # Quantile regression results
        quantile_model = smf.quantreg('ROA ~ ESG_Score', data).fit(q=0.5)
        quantile_predictions = quantile_model.predict(data['ESG_Score'])
        quantile_r2 = r2_score(data['ROA'], quantile_predictions)
        results['Quantile Regression'] = quantile_r2
    
        print(f"R-squared values for {index_name}:")
        for model, r2 in results.items():
            print(f"{model}: {r2}")
    
        # Plot the models for visual comparison
        plt.figure(figsize=(14, 10))
        plt.scatter(data['ESG_Score'], data['ROA'], alpha=0.6, label='Data Points')
    
        # Plot predictions
        plt.plot(data['ESG_Score'], poly_predictions, color='blue', label='Polynomial Regression')
        plt.plot(data[data['ESG_Score'] > 0]['ESG_Score'], log_predictions, color='green', label='Logarithmic Regression')
        plt.plot(data[data['ROA'] > 0]['ESG_Score'], exp_predictions, color='red', label='Exponential Regression')
        plt.plot(data['ESG_Score'], robust_predictions, color='purple', label='Robust Regression')
        plt.plot(data['ESG_Score'], quantile_predictions, color='orange', label='Quantile Regression')
    
        plt.xlabel('ESG Score', fontsize=12)
        plt.ylabel('ROA', fontsize=12)
        plt.title(f'Comparison of Non-Linear Models for ESG Score vs ROA ({index_name})', fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()
    
        return results
    
    # Dictionary to hold R-squared values for each MSCI index
    r2_dict = {
        'MSCI_INDEX': [],
        'Polynomial Regression': [],
        'Logarithmic Regression': [],
        'Exponential Regression': [],
        'Robust Regression': [],
        'Quantile Regression': []
    }
    
    # Perform the comparison for each MSCI index and collect R-squared values
    for index, group in DF_ESG_FIN_1_short_roa.groupby('MSCI_INDEX'):
        r2_values = compare_non_linear_models(group, index)
        r2_dict['MSCI_INDEX'].append(index)
        for model, r2 in r2_values.items():
            r2_dict[model].append(r2)
    
    # Create a DataFrame to hold the R-squared values for each model and MSCI index
    r2_df = pd.DataFrame(r2_dict)
    
    # Display the table to the user
    
    # Save the table to an Excel file for better visualization
    output_excel_path = '../../output/r2_values_non_linear_models_by_msci_index.xlsx'
    r2_df.to_excel(output_excel_path, index=False)
    
    output_excel_path
    r2_df

    #%%

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    from scipy import stats
    
    # Load the data
   # file_path = '/mnt/data/MSCI_INDEX_ALL_ESG_SCORES_2_Sampled_20240628_15_FINAL.csv'
    #DF_ESG_FIN_1 = pd.read_csv(file_path)
    
    # Rename the necessary columns for clarity
    DF_ESG_FIN_1_short_roa = DF_ESG_FIN_1.rename(columns={
        'MSCI_INDUSTRY_ADJUSTED_ESG_SCORE': 'ESG_Score',
        'SP_FIN_RETURN_ON_ASSETS_ROA_PERCENT': 'ROA'
    })
    
    # Function to perform log transformation and exclude outliers
    def transform_and_exclude_outliers(data):
        # Ensure positive values for log transformation
        data = data[(data['ESG_Score'] > 0) & (data['ROA'] > 0)]
        data['log_ESG_Score'] = np.log(data['ESG_Score'])
        data['log_ROA'] = np.log(data['ROA'])
    
        # Remove outliers using Z-score
        z_scores = np.abs(stats.zscore(data[['log_ESG_Score', 'log_ROA']]))
        data_cleaned = data[(z_scores < 3).all(axis=1)]
    
        return data_cleaned
    
    # Function to perform logarithmic regression on transformed and cleaned data with different high contrast colors
    def perform_logarithmic_regression_cleaned_colored(data, index_name, color):
        data_cleaned = transform_and_exclude_outliers(data)
        model = sm.OLS(data_cleaned['log_ROA'], sm.add_constant(data_cleaned['log_ESG_Score'])).fit()
        predictions = model.predict(sm.add_constant(data_cleaned['log_ESG_Score']))
    
        # Print the summary of the regression model
        print(f"Logarithmic Regression Results for {index_name} (Transformed and Cleaned)")
        print(model.summary())
    
        # Plot the regression results
        plt.figure(figsize=(10, 6))
        plt.scatter(data_cleaned['log_ESG_Score'], data_cleaned['log_ROA'], alpha=0.6, edgecolors='w', linewidth=0.5, label='Data Points', color=color)
        plt.plot(data_cleaned['log_ESG_Score'], predictions, color='red', label='Logarithmic Regression Line')
        plt.title(f'Logarithmic Regression: ESG Score vs ROA for {index_name} (Transformed and Cleaned)', fontsize=14)
        plt.xlabel('Log(ESG Score)', fontsize=12)
        plt.ylabel('Log(ROA)', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.show()
    
        return model
    
    # Colors for each index
    colors = {
        'MSCI_EM': 'blue',
        'MSCI_EUROPE': 'green',
        'MSCI_US': 'purple',
        'MSCI_WORLD': 'orange'
    }
    
    # Perform the logarithmic regression for each MSCI index on transformed and cleaned data with different colors
    for index, group in DF_ESG_FIN_1_short_roa.groupby('MSCI_INDEX'):
        perform_logarithmic_regression_cleaned_colored(group, index, colors[index])
    
#%%    
    # Function to perform log transformation and exclude outliers for ROE
    def transform_and_exclude_outliers_roe(data):
        # Ensure positive values for log transformation
        data = data[(data['ESG_Score'] > 0) & (data['ROE'] > 0)]
        data['log_ESG_Score'] = np.log(data['ESG_Score'])
        data['log_ROE'] = np.log(data['ROE'])
    
        # Remove outliers using Z-score
        z_scores = np.abs(stats.zscore(data[['log_ESG_Score', 'log_ROE']]))
        data_cleaned = data[(z_scores < 3).all(axis=1)]
    
        return data_cleaned
    
    # Function to perform logarithmic regression after log transformation and outlier exclusion for ESG Score vs ROE
    def perform_logarithmic_regression_cleaned(data, index_name):
        data_cleaned = transform_and_exclude_outliers_roe(data)
    
        model = sm.OLS(data_cleaned['log_ROE'], sm.add_constant(data_cleaned['log_ESG_Score'])).fit()
        predictions = model.predict(sm.add_constant(data_cleaned['log_ESG_Score']))
    
        # Print the summary of the regression model
        print(f"Logarithmic Regression Results for {index_name} (Transformed and Cleaned) (ROE)")
        print(model.summary())
    
        # Plot the regression results
        plt.figure(figsize=(10, 6))
        plt.scatter(data_cleaned['log_ESG_Score'], data_cleaned['log_ROE'], alpha=0.6, edgecolors='w', linewidth=0.5, label='Data Points')
        plt.plot(data_cleaned['log_ESG_Score'], predictions, color='red', label='Logarithmic Regression Line')
        plt.title(f'Logarithmic Regression: ESG Score vs ROE for {index_name} (Transformed and Cleaned)', fontsize=14)
        plt.xlabel('Log(ESG Score)', fontsize=12)
        plt.ylabel('Log(ROE)', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.show()
    
        return model
    
    # Perform the logarithmic regression for each MSCI index on transformed and cleaned data for ESG Score vs ROE
    results = {}
    for index, group in DF_ESG_FIN_1_short_roe.groupby('MSCI_INDEX'):
        results[index] = perform_logarithmic_regression_cleaned(group, index)
        

##########################################################









#%%
# Function to plot scatter plot and histogram for a given dataset and index name
def plot_esg_vs_roa(data, index_name, scatter_color, hist_color):
    plt.rcParams.update({'font.size': 8})
    
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # Scatter plot
    axs[0].scatter(data['MSCI_INDUSTRY_ADJUSTED_ESG_SCORE'], data['SP_FIN_RETURN_ON_ASSETS_ROA_PERCENT'], alpha=0.7, color=scatter_color)
    axs[0].set_title(f'Scatter Plot for {index_name}')
    axs[0].set_xlabel('ESG Score')
    axs[0].set_ylabel('ROA Percent')
    axs[0].grid(True)

    # Histogram of ESG Scores
    axs[1].hist(data['MSCI_INDUSTRY_ADJUSTED_ESG_SCORE'], bins=20, edgecolor='k', alpha=0.7, color=hist_color)
    axs[1].set_title(f'ESG Scores for {index_name}')
    axs[1].set_xlabel('ESG Score')
    axs[1].set_ylabel('Frequency')
    axs[1].grid(True)

    # Histogram of ROA Percent
    axs[2].hist(data['SP_FIN_RETURN_ON_ASSETS_ROA_PERCENT'], bins=20, edgecolor='k', alpha=0.7, color=hist_color)
    axs[2].set_title(f'ROA Percent for {index_name}')
    axs[2].set_xlabel('ROA Percent')
    axs[2].set_ylabel('Frequency')
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig(f'plots_{index_name}.png')
    plt.show()

# Plot for MSCI_US with a beautiful color mix
plot_esg_vs_roa(us_data, 'MSCI_US', '#3498db', '#85c1e9')  # Blue shades

# Plot for MSCI_EM with a beautiful color mix
plot_esg_vs_roa(em_data, 'MSCI_EM', '#2ecc71', '#82e0aa')  # Green shades

# Plot for MSCI_EU with a beautiful color mix
plot_esg_vs_roa(eu_data, 'MSCI_EU', '#e74c3c', '#f5b7b1')  # Red shades

# Plot for MSCI_EU with a beautiful color mix
plot_esg_vs_roa(world_data, 'MSCI_WORLD', '#678c3c', '#85b6b1')  # Red shades
#%%

#%%
# Perform correlation analysis and plot results for all data including individual and aggregated with correlation coefficients and p-values on the chart
def correlation_analysis(data, index_name, scatter_color, hist_color):
    # Calculate Spearman correlation coefficient and p-value
    corr, p_value = spearmanr(data['MSCI_INDUSTRY_ADJUSTED_ESG_SCORE'], data['SP_FIN_RETURN_ON_ASSETS_ROA'])
    
    # Print correlation results
    print(f"{index_name} - Spearman Correlation Coefficient: {corr}, P-value: {p_value}")
    
    # Plotting
    plt.rcParams.update({'font.size': 8})
    
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # Scatter plot
    axs[0].scatter(data['MSCI_INDUSTRY_ADJUSTED_ESG_SCORE'], data['SP_FIN_RETURN_ON_ASSETS_ROA'], alpha=0.7, color=scatter_color)
    axs[0].set_title(f'Scatter Plot for {index_name}')
    axs[0].set_xlabel('ESG Score')
    axs[0].set_ylabel('ROA Percent')
    axs[0].grid(True)
    axs[0].text(0.95, 0.01, f'Spearman Corr: {corr:.2f}\nP-value: {p_value:.4f}', 
                verticalalignment='bottom', horizontalalignment='right', 
                transform=axs[0].transAxes, color='black', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    # Histogram of ESG Scores
    axs[1].hist(data['MSCI_INDUSTRY_ADJUSTED_ESG_SCORE'], bins=20, edgecolor='k', alpha=0.7, color=hist_color)
    axs[1].set_title(f'ESG Scores for {index_name}')
    axs[1].set_xlabel('ESG Score')
    axs[1].set_ylabel('Frequency')
    axs[1].grid(True)

    # Histogram of ROA Percent
    axs[2].hist(data['SP_FIN_RETURN_ON_ASSETS_ROA'], bins=20, edgecolor='k', alpha=0.7, color=hist_color)
    axs[2].set_title(f'ROA Percent for {index_name}')
    axs[2].set_xlabel('ROA Percent')
    axs[2].set_ylabel('Frequency')
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig(f'../../output/correlation_analysis_{index_name}.png')
    plt.show()

# Perform correlation analysis and plot results for each MSCI index
correlation_analysis(us_data, 'MSCI_US', '#3498db', '#85c1e9')  # Blue shades
correlation_analysis(em_data, 'MSCI_EM', '#2ecc71', '#82e0aa')  # Green shades
correlation_analysis(eu_data, 'MSCI_EUROPE', '#e74c3c', '#f5b7b1')  # Red shades
correlation_analysis(world_data, 'MSCI_WORLD', '#8e44ad', '#d2b4de')  # Purple shades


# %%
import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
indexes = ['MSCI_US', 'MSCI_EM', 'MSCI_EUROPE', 'MSCI_WORLD']
correlation_coefficients = [0.221, 0.148, 0.039, -0.020]
p_values = [0.027, 0.142, 0.698, 0.846]
esg_scores = [6.61, 4.19, 7.65, 6.02]  # Average ESG scores for each index

# Create figure and axis objects with specified size for a slim and tall chart
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plotting the correlation coefficients
color = 'tab:blue'
ax1.set_xlabel('MSCI Index')
ax1.set_ylabel('Spearman Correlation Coefficient', color=color)
ax1.bar(indexes, correlation_coefficients, color=color, alpha=0.7, width=0.3, label='Correlation Coefficient')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(-0.1, 0.3)  # setting the limit for better visualization

# Adding a secondary y-axis to plot p-values
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('P-value', color=color)
ax2.plot(indexes, p_values, color=color, marker='o', label='P-value')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, 1.0)  # setting the limit for better visualization

# Adding a secondary y-axis to plot ESG scores
ax3 = ax1.twinx()
color = 'tab:green'
ax3.spines['right'].set_position(('outward', 60))  # move the third axis to the right
ax3.set_ylabel('ESG Score', color=color)
ax3.bar(indexes, esg_scores, color=color, alpha=0.5, width=0.2, label='ESG Score')
ax3.tick_params(axis='y', labelcolor=color)
ax3.set_ylim(0, 10)  # setting the limit for better visualization

# Adding titles and grid
plt.title('Spearman Correlation Coefficient, P-value, and ESG Score for MSCI Indexes')
fig.tight_layout()
plt.grid(True)

# Adding legend
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
ax3.legend(loc='upper center')

# Save and show the plot
plt.savefig('../../output/correlation_pvalue_esg_chart_slim.png')
plt.show()
# %%
import statsmodels.formula.api as smf
import statsmodels.api as sm
# Redefine the function to perform and compare regression analyses
def compare_regressions(data, index_name):
    # OLS Regression
    X = sm.add_constant(data['MSCI_INDUSTRY_ADJUSTED_ESG_SCORE'])
    Y = data['SP_FIN_RETURN_ON_ASSETS_ROA_PERCENT']
    ols_model = sm.OLS(Y, X).fit()

    # Robust Regression (RLM)
    rlm_model = sm.RLM(Y, X, M=sm.robust.norms.HuberT()).fit()

    # Quantile Regression (Median, 50th percentile)
    quant_model = smf.quantreg('SP_FIN_RETURN_ON_ASSETS_ROA_PERCENT ~ MSCI_INDUSTRY_ADJUSTED_ESG_SCORE', data).fit(q=0.5)
    
    # Compare models
    print(f"\nComparison of Regression Models for {index_name}")
    print("\nOLS Regression:")
    print(ols_model.summary())
    print("\nRobust Regression (RLM):")
    print(rlm_model.summary())
    print("\nQuantile Regression (50th percentile):")
    print(quant_model.summary())

# Perform and compare regression analyses for each MSCI index
compare_regressions(us_data, 'MSCI_US')
compare_regressions(em_data, 'MSCI_EM')
compare_regressions(eu_data, 'MSCI_EUROPE')
compare_regressions(world_data, 'MSCI_WORLD')


#%%
# Create a DataFrame to summarize the results
summary_data = {
    "Model": [],
    "Index": [],
    "R-squared": [],
    "Adj. R-squared": [],
    "F-statistic": [],
    "P-value (F-statistic)": [],
    "Coef (Intercept)": [],
    "P-value (Intercept)": [],
    "Coef (ESG Score)": [],
    "P-value (ESG Score)": [],
}

def append_results(model, index_name, model_name):
    summary_data["Model"].append(model_name)
    summary_data["Index"].append(index_name)
    summary_data["R-squared"].append(model.rsquared if hasattr(model, 'rsquared') else 'N/A')
    summary_data["Adj. R-squared"].append(model.rsquared_adj if hasattr(model, 'rsquared_adj') else 'N/A')
    summary_data["F-statistic"].append(model.fvalue if hasattr(model, 'fvalue') else 'N/A')
    summary_data["P-value (F-statistic)"].append(model.f_pvalue if hasattr(model, 'f_pvalue') else 'N/A')
    summary_data["Coef (Intercept)"].append(model.params[0])
    summary_data["P-value (Intercept)"].append(model.pvalues[0])
    summary_data["Coef (ESG Score)"].append(model.params[1])
    summary_data["P-value (ESG Score)"].append(model.pvalues[1])

# Append results for each model and index
# OLS
append_results(sm.OLS(us_data['SP_FIN_RETURN_ON_ASSETS_ROA_PERCENT'], sm.add_constant(us_data['MSCI_INDUSTRY_ADJUSTED_ESG_SCORE'])).fit(), 'MSCI_US', 'OLS')
append_results(sm.OLS(em_data['SP_FIN_RETURN_ON_ASSETS_ROA_PERCENT'], sm.add_constant(em_data['MSCI_INDUSTRY_ADJUSTED_ESG_SCORE'])).fit(), 'MSCI_EM', 'OLS')
append_results(sm.OLS(eu_data['SP_FIN_RETURN_ON_ASSETS_ROA_PERCENT'], sm.add_constant(eu_data['MSCI_INDUSTRY_ADJUSTED_ESG_SCORE'])).fit(), 'MSCI_EUROPE', 'OLS')
append_results(sm.OLS(world_data['SP_FIN_RETURN_ON_ASSETS_ROA_PERCENT'], sm.add_constant(world_data['MSCI_INDUSTRY_ADJUSTED_ESG_SCORE'])).fit(), 'MSCI_WORLD', 'OLS')

# RLM
append_results(sm.RLM(us_data['SP_FIN_RETURN_ON_ASSETS_ROA_PERCENT'], sm.add_constant(us_data['MSCI_INDUSTRY_ADJUSTED_ESG_SCORE']), M=sm.robust.norms.HuberT()).fit(), 'MSCI_US', 'RLM')
append_results(sm.RLM(em_data['SP_FIN_RETURN_ON_ASSETS_ROA_PERCENT'], sm.add_constant(em_data['MSCI_INDUSTRY_ADJUSTED_ESG_SCORE']), M=sm.robust.norms.HuberT()).fit(), 'MSCI_EM', 'RLM')
append_results(sm.RLM(eu_data['SP_FIN_RETURN_ON_ASSETS_ROA_PERCENT'], sm.add_constant(eu_data['MSCI_INDUSTRY_ADJUSTED_ESG_SCORE']), M=sm.robust.norms.HuberT()).fit(), 'MSCI_EUROPE', 'RLM')
append_results(sm.RLM(world_data['SP_FIN_RETURN_ON_ASSETS_ROA_PERCENT'], sm.add_constant(world_data['MSCI_INDUSTRY_ADJUSTED_ESG_SCORE']), M=sm.robust.norms.HuberT()).fit(), 'MSCI_WORLD', 'RLM')

# Quantile Regression (50th percentile)
append_results(smf.quantreg('SP_FIN_RETURN_ON_ASSETS_ROA_PERCENT ~ MSCI_INDUSTRY_ADJUSTED_ESG_SCORE', us_data).fit(q=0.5), 'MSCI_US', 'Quantile Regression')
append_results(smf.quantreg('SP_FIN_RETURN_ON_ASSETS_ROA_PERCENT ~ MSCI_INDUSTRY_ADJUSTED_ESG_SCORE', em_data).fit(q=0.5), 'MSCI_EM', 'Quantile Regression')
append_results(smf.quantreg('SP_FIN_RETURN_ON_ASSETS_ROA_PERCENT ~ MSCI_INDUSTRY_ADJUSTED_ESG_SCORE', eu_data).fit(q=0.5), 'MSCI_EUROPE', 'Quantile Regression')
append_results(smf.quantreg('SP_FIN_RETURN_ON_ASSETS_ROA_PERCENT ~ MSCI_INDUSTRY_ADJUSTED_ESG_SCORE', world_data).fit(q=0.5), 'MSCI_WORLD', 'Quantile Regression')

# Convert to DataFrame
summary_df = pd.DataFrame(summary_data)


summary_df

# %%
# Export the summary DataFrame to an Excel file
output_file_path = '../../output/dataset/Regression_Summary_Table.xlsx'
summary_df.to_excel(output_file_path, index=False)

output_file_path


# %%
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Function to plot RLM results for each MSCI index in a single row
def plot_rlm_results_in_row(indices):
    fig, axes = plt.subplots(1, len(indices), figsize=(20, 6))
    
    for ax, (index_name, data) in zip(axes, indices.items()):
        X = sm.add_constant(data['MSCI_INDUSTRY_ADJUSTED_ESG_SCORE'])
        Y = data['SP_FIN_RETURN_ON_ASSETS_ROA_PERCENT']
        rlm_model = sm.RLM(Y, X, M=sm.robust.norms.HuberT()).fit()
        
        ax.scatter(data['MSCI_INDUSTRY_ADJUSTED_ESG_SCORE'], Y, color='blue', label='Actual Data')
        ax.plot(data['MSCI_INDUSTRY_ADJUSTED_ESG_SCORE'], rlm_model.predict(X), color='red', label='RLM Fit')
        ax.set_xlabel('MSCI ESG Score')
        ax.set_ylabel('ROA Percent')
        ax.set_title(f'RLM Model Fit for {index_name}')
        ax.legend()
    
    plt.tight_layout()
    plt.show()

# Create a dictionary with the data for each MSCI index
indices = {
    'MSCI_US': us_data,
    'MSCI_EM': em_data,
    'MSCI_EUROPE': eu_data,
    'MSCI_WORLD': world_data
}

# Plot RLM results for each MSCI index in a single row
plot_rlm_results_in_row(indices)


# %%
# Function to plot RLM model results in a 2x2 grid
def plot_rlm_results_in_grid(indices):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for ax, (index_name, data) in zip(axes.flatten(), indices.items()):
        X = sm.add_constant(data['MSCI_INDUSTRY_ADJUSTED_ESG_SCORE'])
        Y = data['SP_FIN_RETURN_ON_ASSETS_ROA_PERCENT']
        rlm_model = sm.RLM(Y, X, M=sm.robust.norms.HuberT()).fit()
        
        ax.scatter(data['MSCI_INDUSTRY_ADJUSTED_ESG_SCORE'], Y, color='blue', label='Actual Data')
        ax.plot(data['MSCI_INDUSTRY_ADJUSTED_ESG_SCORE'], rlm_model.predict(X), color='red', label='RLM Fit')
        ax.set_xlabel('MSCI ESG Score')
        ax.set_ylabel('ROA Percent')
        ax.set_title(f'RLM Model Fit for {index_name}')
        ax.legend()
    
    plt.tight_layout()
    plt.show()

# Plot RLM results for each MSCI index in a 2x2 grid
plot_rlm_results_in_grid(indices)

# %%
#******************************************************************
# MSCI_INDUSTRY_ADJUSTED_ESG_SCORE vs ????
#******************************************************************

import pandas as pd
import matplotlib.pyplot as plt


# Load your dataset
dataset_directory_path='/Users/ajk/Library/CloudStorage/OneDrive-Personal/ajkdrive/HKU/Class_3035982673/MLIM 7000C/Datasets/MSCI_ESG_SP_KOYFIN/'
file_path = f'{dataset_directory_path}MSCI_INDEX_ALL_ESG_SCORES_2_Sampled_20240628_15_FINAL.csv'
file_path
DF_ESG_FIN_1 = pd.read_csv(file_path)
DF_ESG_FIN_1
# Filter datasets for each index
us_data = DF_ESG_FIN_1[DF_ESG_FIN_1['MSCI_INDEX'] == 'MSCI_US']
em_data = DF_ESG_FIN_1[DF_ESG_FIN_1['MSCI_INDEX'] == 'MSCI_EM']
eu_data = DF_ESG_FIN_1[DF_ESG_FIN_1['MSCI_INDEX'] == 'MSCI_EUROPE']
world_data = DF_ESG_FIN_1[DF_ESG_FIN_1['MSCI_INDEX'] == 'MSCI_WORLD']


#%%
import matplotlib.pyplot as plt

# Function to plot scatter plots for MSCI_ENVIRONMENTAL_PILLAR_SCORE vs. all SP_FIN_ variables
def plot_env_vs_sp_vars(data, index_name):
    sp_columns = [col for col in data.columns if col.startswith('SP_FIN_')]
    
    num_plots = len(sp_columns)
    num_cols = 3  # Number of columns in the plot grid
    num_rows = (num_plots // num_cols) + (num_plots % num_cols > 0)  # Calculate the number of rows needed
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axes = axes.flatten()  # Flatten the array of axes for easy iteration

    for ax, col in zip(axes, sp_columns):
        ax.scatter(data['MSCI_ENVIRONMENTAL_PILLAR_SCORE'], data[col], alpha=0.7)
        ax.set_xlabel('MSCI Environmental Pillar Score')
        ax.set_ylabel(col)
        ax.set_title(f'{index_name}: {col}')
        ax.grid(True)

    # Remove any empty subplots
    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

# Create a dictionary with the data for each MSCI index
indices = {
    'MSCI_US': us_data,
    # 'MSCI_EM': em_data,
    # 'MSCI_EUROPE': eu_data,
    # 'MSCI_WORLD': world_data
}

# Plot for each MSCI index
for index_name, data in indices.items():
    plot_env_vs_sp_vars(data, index_name)

# %%
import pandas as pd

# Function to find correlation between MSCI_ENVIRONMENTAL_PILLAR_SCORE and SP_FIN_ variables
def find_correlations(data, index_name):
    sp_columns = [col for col in data.columns if col.startswith('SP_FIN_')]
    
    correlations = {}
    for col in sp_columns:
        correlation = data['MSCI_ENVIRONMENTAL_PILLAR_SCORE'].corr(data[col])
        correlations[col] = correlation
    
    # Convert the correlations dictionary to a DataFrame for better readability
    correlations_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['Correlation'])
    correlations_df.index.name = 'SP Variable'
    correlations_df.reset_index(inplace=True)
    
    print(f"Correlations between MSCI_ENVIRONMENTAL_PILLAR_SCORE and SP_FIN_ variables for {index_name}:")
    print(correlations_df)
    print()

# Create a dictionary with the data for each MSCI index
indices = {
    'MSCI_US': us_data,
    'MSCI_EM': em_data,
    'MSCI_EUROPE': eu_data,
    'MSCI_WORLD': world_data
}

# Find and display correlations for each MSCI index
for index_name, data in indices.items():
    find_correlations(data, index_name)

# %%

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV file
#file_path = 'path_to_your_file.csv'  # Replace with the actual file path
data = pd.read_csv(file_path)

# Function to create a correlation heatmap for all MSCI_ and SP_ variables
def plot_correlation_heatmap(data, index_name):
    # Select the relevant columns and ensure they are numeric
    relevant_columns = [col for col in data.columns if (col.startswith('MSCI_') or col.startswith('SP_')) and data[col].dtype in [float, int]]
    relevant_data = data[relevant_columns]
    
    # Calculate the correlation matrix
    corr_matrix = relevant_data.corr()
    
    # Plot the heatmap
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
    plt.title(f'Correlation Heatmap for {index_name}')
    plt.show()

# Get the unique MSCI_INDEX values
unique_indices = data['MSCI_INDEX'].unique()

# Generate and display correlation heatmaps for each MSCI index
for index_name in unique_indices:
    index_data = data[data['MSCI_INDEX'] == index_name]
    plot_correlation_heatmap(index_data, index_name)

# %%
import pandas as pd
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import seaborn as sns


# Load the uploaded CSV file
data = pd.read_csv(file_path)

# List of relevant variables
relevant_columns = [
    'MSCI_INDUSTRY_ADJUSTED_ESG_SCORE',
    'MSCI_ENVIRONMENTAL_PILLAR_SCORE',
    'MSCI_SOCIAL_PILLAR_SCORE',
    'MSCI_GOVERNANCE_PILLAR_SCORE',
    'MSCI_CARBON_EMISSIONS_SCORE',
    'SP_FIN_DEBT_TO_EQUITY_RATIO_PERCENT',
    'SP_FIN_RETURN_ON_ASSETS_ROA',
    'SP_FIN_RETURN_ON_ASSETS_ROA_PERCENT',
    'SP_FIN_RETURN_ON_EQUITY_PERCENT',
    'SP_FIN_PE_LTM',
    'TOBINQ RATIO',
    'SP_FIN_ENTERPRISE_VALUE_TO_ASSET'
]

# Function to perform Shapiro-Wilk test for normality on each relevant variable for each MSCI group
def normality_test_by_group(data):
    unique_indices = data['MSCI_INDEX'].unique()
    group_normality_results = {}

    for index_name in unique_indices:
        index_data = data[data['MSCI_INDEX'] == index_name]
        normality_results = {}
        for column in index_data.columns:
            if pd.api.types.is_numeric_dtype(index_data[column]):
                stat, p_value = shapiro(index_data[column].dropna())
                normality_results[column] = {'Statistic': stat, 'p-value': p_value}
        group_normality_results[index_name] = normality_results
    
    return group_normality_results

# Perform normality test on each MSCI group
group_normality_results = normality_test_by_group(data)

# Display the normality test results for each MSCI group
group_normality_results_df = pd.concat({k: pd.DataFrame(v).T for k, v in group_normality_results.items()}, axis=0)

group_normality_results_df.index = group_normality_results_df.index.set_names(['MSCI_INDEX', 'Variable'])
group_normality_results_df.reset_index(inplace=True)

# Adding a column to indicate whether the variable follows a normal distribution based on p-value
group_normality_results_df['Normal'] = group_normality_results_df['p-value'] > 0.05
group_normality_results_df['Normal'] = group_normality_results_df['Normal'].apply(lambda x: 'Normal' if x else 'Not Normal')

# Export the results to an Excel file
output_file_path = '../../output/dataset/Full_Group_Normality_Test_Results.xlsx'
group_normality_results_df.to_excel(output_file_path, index=False)

# %%

# Load the uploaded CSV file
data = pd.read_csv(file_path)


# Extract the necessary variables for normality test
variables_for_normality_test = [
    'MSCI_INDUSTRY_ADJUSTED_ESG_SCORE',
    'SP_FIN_DEBT_TO_EQUITY_RATIO_PERCENT',
    'SP_FIN_RETURN_ON_ASSETS_ROA_PERCENT',
    'SP_FIN_RETURN_ON_EQUITY_PERCENT',
    'SP_FIN_PE_LTM',
    'TOBINQ RATIO',
    'SP_FIN_ENTERPRISE_VALUE_TO_ASSET'
]

# Perform Shapiro-Wilk normality test for specified variables within each MSCI group
def specific_normality_test_by_group(data):
    unique_indices = data['MSCI_INDEX'].unique()
    group_normality_results = {}

    for index_name in unique_indices:
        index_data = data[data['MSCI_INDEX'] == index_name]
        normality_results = {}
        for column in variables_for_normality_test:
            if pd.api.types.is_numeric_dtype(index_data[column]):
                stat, p_value = shapiro(index_data[column].dropna())
                normality_results[column] = {'Statistic': stat, 'p-value': p_value}
        group_normality_results[index_name] = normality_results
    
    return group_normality_results

# Perform the normality test
specific_group_normality_results = specific_normality_test_by_group(data)

# Create a DataFrame for the results
specific_group_normality_results_df = pd.concat({k: pd.DataFrame(v).T for k, v in specific_group_normality_results.items()}, axis=0)
specific_group_normality_results_df.index = specific_group_normality_results_df.index.set_names(['MSCI_INDEX', 'Variable'])
specific_group_normality_results_df.reset_index(inplace=True)

# Add a column to indicate whether the variable follows a normal distribution
specific_group_normality_results_df['Normal'] = specific_group_normality_results_df['p-value'] > 0.05
specific_group_normality_results_df['Normal'] = specific_group_normality_results_df['Normal'].apply(lambda x: 'Normal' if x else 'Not Normal')

# Export the results to an Excel file
output_specific_file_path = '../../output/dataset/Specific_Group_Normality_Test_Results.xlsx'
specific_group_normality_results_df.to_excel(output_specific_file_path, index=False)

output_specific_file_path


# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import matplotlib.image as mpimg

# Load your data
data = pd.read_csv(file_path)  # Replace with the actual file path

# Define the variables
variables_for_normality_test = [
    'MSCI_INDUSTRY_ADJUSTED_ESG_SCORE',
    'SP_FIN_DEBT_TO_EQUITY_RATIO_PERCENT',
    'SP_FIN_RETURN_ON_ASSETS_ROA_PERCENT',
    'SP_FIN_RETURN_ON_EQUITY_PERCENT',
    'SP_FIN_PE_LTM',
    'TOBINQ RATIO',
    'SP_FIN_ENTERPRISE_VALUE_TO_ASSET'
]

# Function to create and save individual heatmaps
def save_individual_heatmaps(data, variables):
    unique_indices = data['MSCI_INDEX'].unique()
    
    for index_name in unique_indices:
        index_data = data[data['MSCI_INDEX'] == index_name]
        relevant_data = index_data[variables]
        
        # Calculate the Spearman correlation matrix
        spearman_corr_matrix = relevant_data.corr(method='spearman')
        
        # Plot the heatmap with the 'viridis' color palette
        plt.figure(figsize=(10, 8))
        sns.heatmap(spearman_corr_matrix, annot=True, cmap='viridis', fmt='.2f', vmin=-1, vmax=1)
        plt.title(f'Spearman Correlation Heatmap for {index_name}', fontsize=16)
        plt.tight_layout()
        
        # Save the plot to a file
        file_path = f'{index_name}_heatmap.png'
        plt.savefig(file_path)
        plt.close()

# Save individual heatmaps for the specified variables within each MSCI index group
save_individual_heatmaps(data, variables_for_normality_test)

# Display combined heatmaps in a 2x2 layout
def display_combined_heatmaps():
    unique_indices = data['MSCI_INDEX'].unique()[:4]  # Limit to 4 groups for 2x2 layout
    
    fig, axs = plt.subplots(2, 2, figsize=(30, 30))
    fig.suptitle('Spearman Correlation Heatmaps for MSCI Index Groups', fontsize=30)
    
    for i, index_name in enumerate(unique_indices):
        img = mpimg.imread(f'{index_name}_heatmap.png')
        axs[i//2, i%2].imshow(img)
        axs[i//2, i%2].axis('off')
        axs[i//2, i%2].set_title(f'{index_name}', fontsize=16)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

display_combined_heatmaps()

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Load your data
data = pd.read_csv(file_path)  # Replace with the actual file path

# Define the variables
variables_for_normality_test = [
    'MSCI_INDUSTRY_ADJUSTED_ESG_SCORE',
    'SP_FIN_DEBT_TO_EQUITY_RATIO_PERCENT',
    'SP_FIN_RETURN_ON_ASSETS_ROA_PERCENT',
    'SP_FIN_RETURN_ON_EQUITY_PERCENT',
    'SP_FIN_PE_LTM',
    'TOBINQ RATIO',
    'SP_FIN_ENTERPRISE_VALUE_TO_ASSET'
]

# Function to create and save individual heatmaps
def save_individual_heatmaps(data, variables):
    unique_indices = data['MSCI_INDEX'].unique()
    
    for index_name in unique_indices:
        index_data = data[data['MSCI_INDEX'] == index_name]
        relevant_data = index_data[variables]
        
        # Calculate the Spearman correlation matrix
        spearman_corr_matrix = relevant_data.corr(method='spearman')
        
        # Plot the heatmap with the 'viridis' color palette
        plt.figure(figsize=(10, 8))
        sns.heatmap(spearman_corr_matrix, annot=True, cmap='viridis', fmt='.2f', vmin=-1, vmax=1)
        plt.title(f'Spearman Correlation Heatmap for {index_name}', fontsize=16)
        plt.tight_layout()
        
        # Save the plot to a file
        file_path = f'{index_name}_heatmap.png'
        plt.savefig(file_path)
        plt.close()

# Save individual heatmaps for the specified variables within each MSCI index group
save_individual_heatmaps(data, variables_for_normality_test)

# %%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Display combined heatmaps for the first row
def display_first_row_heatmaps():
    unique_indices = data['MSCI_INDEX'].unique()[:2]  # Limit to the first 2 groups for the first row
    
    fig, axs = plt.subplots(1, 2, figsize=(30, 15))
    fig.suptitle('Spearman Correlation Heatmaps for MSCI Index Groups (First Row)', fontsize=30)
    
    for i, index_name in enumerate(unique_indices):
        img = mpimg.imread(f'{index_name}_heatmap.png')
        axs[i].imshow(img)
        axs[i].axis('off')
        axs[i].set_title(f'{index_name}', fontsize=16)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

display_first_row_heatmaps()

# %%
# Display combined heatmaps for the second row
def display_second_row_heatmaps():
    unique_indices = data['MSCI_INDEX'].unique()[2:4]  # Limit to the next 2 groups for the second row
    
    fig, axs = plt.subplots(1, 2, figsize=(30, 15))
    fig.suptitle('Spearman Correlation Heatmaps for MSCI Index Groups (Second Row)', fontsize=30)
    
    for i, index_name in enumerate(unique_indices):
        img = mpimg.imread(f'{index_name}_heatmap.png')
        axs[i].imshow(img)
        axs[i].axis('off')
        axs[i].set_title(f'{index_name}', fontsize=16)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

display_second_row_heatmaps()

# %%