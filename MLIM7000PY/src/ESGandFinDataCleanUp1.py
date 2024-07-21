#%%
import pandas as pd

#%%
# Load the Excel file
dataset_directory_path = '/Users/ajk/Library/CloudStorage/OneDrive-Personal/ajkdrive/codeLab/MLIM7000/ESGdatasets'
dataset_file_path = f'{dataset_directory_path}/MSCI_INDEX_ALL_ESG_SCORES_15JULY24_1240.xlsx'
df_MSCI_INDEX_ALL_ESG_SCORES_1 = pd.read_excel(dataset_file_path, sheet_name='Sheet1')
df_MSCI_INDEX_ALL_ESG_SCORES_1 
#%%
# Check which columns have missing values
missing_values =  df_MSCI_INDEX_ALL_ESG_SCORES_1.isnull().sum()
missing_values
missing_columns = missing_values[missing_values > 0]

# Display the columns with missing values and their counts
print("Columns with missing values and their counts:")
print(missing_columns)


#%%
import pandas as pd
from datetime import datetime

feature_counts_before_drop = len(df_MSCI_INDEX_ALL_ESG_SCORES_1.columns)
print("Number of columns in the DataFrame:", feature_counts_before_drop)

#%%
# Identify numerical columns
numerical_cols = df_MSCI_INDEX_ALL_ESG_SCORES_1.select_dtypes(include=['number']).columns

# Drop numerical columns with all NaN values
df_MSCI_INDEX_ALL_ESG_SCORES_1_DropNumericalNA = df_MSCI_INDEX_ALL_ESG_SCORES_1.drop(columns=[col for col in numerical_cols if df_MSCI_INDEX_ALL_ESG_SCORES_1[col].isna().any()])

# Check which columns have missing values
missing_values =  df_MSCI_INDEX_ALL_ESG_SCORES_1_DropNumericalNA .isnull().sum()
missing_values
missing_columns = missing_values[missing_values > 0]

# Display the columns with missing values and their counts
print("Columns with missing values and their counts:")
print(missing_columns)

df_MSCI_INDEX_ALL_ESG_SCORES_1_DropNumericalNA.head()

#%%

feature_counts_after_drop = len(df_MSCI_INDEX_ALL_ESG_SCORES_1_DropNumericalNA.columns)
print("Number of columns in the DataFrame:", feature_counts_after_drop)

dropped_feature_counts = feature_counts_before_drop - feature_counts_after_drop
print("Number of columns dropped from DataFrame:", dropped_feature_counts )

# Get current datetime and format it
current_datetime = datetime.now().strftime("%Y%m%d_%H")
current_datetime
# Create a new filename with the datetime suffix
output_file_path_drop_columns = f'{dataset_directory_path}/MSCI_INDEX_ALL_ESG_SCORES_2_{current_datetime}.xlsx'

# Save the cleaned data to a new Excel file
df_MSCI_INDEX_ALL_ESG_SCORES_1_DropNumericalNA.to_excel(output_file_path_drop_columns, index=False)

print(f"Cleaned data saved to: {output_file_path_drop_columns}")

df_MSCI_INDEX_ALL_ESG_SCORES_1_NoMissingValue = pd.read_excel(output_file_path_drop_columns,sheet_name='Sheet1')
df_MSCI_INDEX_ALL_ESG_SCORES_1_NoMissingValue

#%%
#File path

df_MSCI_INDEX_ALL_ESG_SCORES_2 = df_MSCI_INDEX_ALL_ESG_SCORES_1_NoMissingValue
# Perform random sampling to get 100 records from each MSCI_INDEX
df_sampled_data = df_MSCI_INDEX_ALL_ESG_SCORES_2.groupby('MSCI_INDEX').apply(lambda x: x.sample(n=100, random_state=1)).reset_index(drop=True)
type(df_sampled_data)


# %%
output_file_path = f'{dataset_directory_path}/MSCI_INDEX_ALL_ESG_SCORES_2_Sampled_{current_datetime}.xlsx'
output_file_path
# Save the cleaned data to a new Excel file
df_sampled_data.to_excel(output_file_path,index=False)

# %%

import pandas as pd
from datetime import datetime

dataset_file_path = output_file_path

# Load the dataset
df_MSCI_INDEX_ALL_ESG_SCORES_2_Sampled = pd.read_excel(dataset_file_path, sheet_name='Sheet1')
df_MSCI_INDEX_ALL_ESG_SCORES_2_Sampled 
# %%


