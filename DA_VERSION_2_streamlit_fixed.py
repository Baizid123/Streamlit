import streamlit as st

# Streamlit App Generated from Jupyter Notebook

st.markdown('''## Change Log:

21 Dec 2024
* Added new Sections Part 4-6
* Amended Existing Section Names and some content adjustment (mostly arrangment)
* Restructured `Parts` to align with crispdm lifecycle (according to https://www.datascience-pm.com/crisp-dm-2/)
* Replaced `investigating outliers` section into `Part 2` and `Part 3` according to the action taken
* Added explanations for some sections and/or cells
* Made a copy in Drive of the original py notebook before amendments are made''')

st.markdown('''# PART 1: Business Understanding''')

st.markdown('''#PART 2: Data Understanding''')

st.markdown('''## Importing Packages and Reading Dataset''')

st.markdown('''This section initializes the environment for data preprocessing by importing essential libraries: Pandas for data manipulation, NumPy for numerical computations, Matplotlib for creating plots, and Seaborn for advanced statistical visualizations. The configuration pd.set_option('display.max_columns', None) ensures all columns in a dataset are displayed, enabling comprehensive data inspection. These setups collectively provide the tools needed for effective data cleaning, analysis, and visualization, forming the foundation for subsequent steps in the notebook''')

#importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)

st.markdown('''
This section downloads, imports, and verifies datasets for analysis. The `gdown` library retrieves two CSV files (`train.csv` and `test_data.csv`) from specified Google Drive URLs. The datasets are loaded into Pandas DataFrames (`train_df` and `test_df`), which are structured for data manipulation. The `print()` function checks the dimensions of the datasets, providing insights into the number of rows and columns for initial validation. This setup ensures the required data is accessible and correctly formatted for subsequent analytical processes''')

import pandas as pd
import gdown

# URLs of the files
train_url = "https://drive.google.com/uc?id=1h3dLm_YZnlOkYeYhAJYW4v1EMWsbxbRR"
test_url = "https://drive.google.com/uc?id=1s0KGLEcswxCmUxW86r8fqGzZcAamDDCa"

# Download the files using gdown

try:
    import gdown
    train_url = "<YOUR_TRAIN_URL>"
    if not train_url:
        st.error("Please provide a valid URL for the dataset.")
    else:
        output = "train.csv"
        gdown.download(train_url, output, quiet=False)
        st.success(f"Dataset downloaded as {output}.")
except Exception as e:
    st.error(f"An error occurred while downloading the file: {e}")

gdown.download(test_url, "test_data.csv", quiet=False)

# Read the downloaded CSV files into pandas DataFrames
train_df = pd.read_csv("train.csv")  # Load data from train.csv
test_df = pd.read_csv("test_data.csv") # Load data from test_data.csv


#check imported dataset
print(train_df.shape)
print(test_df.shape)

#train_df.head(2).T
#test_df.head(2).T

st.markdown('''
This code snippet sets the `Employee ID` column as the index for both the `train_df` and `test_df` DataFrames using the `set_index()` method with the `inplace=True` parameter. By doing so, the column is removed from the main dataset and designated as the DataFrame index, enabling more efficient and intuitive querying of records based on unique employee identifiers. This step is particularly useful for data manipulation and retrieval tasks where the `Employee ID` serves as a primary key.''')

#setting the Employee ID for both csv files as the index for easier querying

train_df.set_index('Employee ID', inplace=True)
test_df.set_index('Employee ID', inplace=True)


st.markdown('''In this step, we create copies of the original `train_df` and `test_df` DataFrames by using the `copy()` method, assigning them to `df1` and `df2`, respectively. This ensures that any modifications or preprocessing steps performed on `df1` and `df2` do not affect the original DataFrames (`train_df` and `test_df`).

This approach is particularly useful in data analysis and machine learning pipelines to preserve the integrity of the raw data while allowing experimentation or transformations on separate copies. By working on these copies, we maintain flexibility for debugging, validation, or revisiting earlier stages of analysis without compromising the original datasets.''')

df1 = train_df.copy()
df2 = test_df.copy()

st.markdown('''## Discovering Dataset''')

st.markdown('''The displayed DataFrame `train_df` represents the training dataset after setting `Employee ID` as the index. The dataset comprises 22,750 rows and 8 columns, including key features such as `Date of Joining`, `Gender`, `Company Type`, `WFH Setup Available`, `Designation`, `Resource Allocation`, `Mental Fatigue Score`, and `Burn Rate`. This indexed structure allows for streamlined data retrieval and enhanced querying capabilities, especially when analyzing employee-specific information. The dataset is structured for tasks like predictive modeling or trend analysis, particularly in contexts like employee performance, well-being, or resource management''')

train_df

st.markdown('''In this step, we examine the data types of each column in the `train_df` dataset using the `dtypes` attribute. The output reveals that the dataset contains a mix of categorical (object) and numerical (float64) features. Specifically, `Date of Joining`, `Gender`, `Company Type`, and `WFH Setup Available` are stored as objects, representing categorical or textual data. In contrast, `Designation`, `Resource Allocation`, `Mental Fatigue Score`, and `Burn Rate` are stored as float64, indicating numerical data with decimal precision.

By identifying the data types, we establish a clear understanding of the dataset's structure, which is essential for determining the appropriate preprocessing techniques. For instance, numerical columns may require scaling or normalization, while categorical columns may necessitate encoding for machine learning models. This foundational step ensures that subsequent analyses are conducted with accurate data representations.''')

#check datatypes of each column in the dataset for train.csv

train_df.dtypes

st.markdown('''
Next, we analyze the uniqueness of values in each column of the `train_df` DataFrame using the `nunique()` method. This function returns the number of unique values in each column, providing insights into the variability and categorical nature of the data.

Key observations include:
*   `Date of Joining` has 366 unique values, reflecting distinct dates of employee entries
*   `Gender`, `Company Type`, and `WFH Setup Available` are binary features, each with 2 unique values.
*   `Designation` has 6 unique values, suggesting categorical representation for job roles or hierarchy levels.
*   `Resource Allocation`, `Mental Fatigue Score`, and `Burn Rate` show a higher number of unique values (10, 101, and 101, respectively), indicating continuous or semi-continuous numerical features.

This analysis helps us understand the data distribution and guides further preprocessing steps, such as encoding categorical variables or binning continuous features if necessary for modeling tasks''')

#to get the sum of unique values in each column

train_df.nunique()

train_df.shape

#check outliers with boxplot

df1['Mental Fatigue Score'].plot(kind='box', figsize=(6, 6))
plt.title('Boxplot of Mental Fatigue Score')
plt.show()

st.markdown(''' ## Exploring Data''')

#dataset gender distribution

df1_gender = df1['Gender'].value_counts().plot(kind='pie', autopct='%1.1f%%', figsize=(6,6))
df1_gender.set_title('Gender Distribution')
df1_gender.set_ylabel('Count')

# Specify the columns to plot
columns_to_plot = [
    'Designation',
    'Resource Allocation',
    'Mental Fatigue Score',
    'Burn Rate'
]

# Create the subplots
plt.figure(figsize=(15, 10))
for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df1[column], bins=15, kde=True, color='orange', edgecolor='black', alpha=0.6)
    plt.title(f"Distribution of {column.replace('_', ' ')}")
    plt.xlabel(column.replace('_', ' '))
    plt.ylabel('Count')

plt.tight_layout()
plt.show()

#Visualize numeric features:

df1.hist(figsize=(10, 6),bins=20)
plt.tight_layout()
plt.show()

#checking the relationship between employee's given workload and mental fatigueness

df1_workload_vs_fatigue = df1.plot.scatter(x='Resource Allocation', y='Mental Fatigue Score', figsize=(10,10))
df1_workload_vs_fatigue.set_title('Workload vs Mental Fatigue Score')

df1.head(2).T

#check how many have proper WFH setup

#df1['WFH Setup Available'].value_counts()
df1_wfh_setup_distb = df1['WFH Setup Available'].value_counts().plot(kind='bar', figsize=(8,6))
df1_wfh_setup_distb.set_title('WFH Setup Distribution')
df1_wfh_setup_distb.set_xlabel('WFH Setup Available')
df1_wfh_setup_distb.set_ylabel('Frequency')

#group employee levels

bins = [0, 1.5, 3.5, 5.0]  # Bin edges for low, mid, and high
labels = ['Low', 'Mid', 'High']  # Labels for the bins

df1['Designation Level'] = pd.cut(df1['Designation'], bins=bins, labels=labels, include_lowest=True)

df1.head(2).T

#to have an overview of "Which staff designation level has/hasnt a wfh setup the most?"

wfh_designation_counts = df1.groupby(['WFH Setup Available', 'Designation Level']).size().unstack()

ax = wfh_designation_counts.plot(kind='bar', stacked=True, figsize=(10, 10))
ax.set_title('WFH Setup Distribution by Designation Level')
ax.set_xlabel('WFH Setup Available')
ax.set_ylabel('Number of Employees')
plt.xticks(rotation=0)
plt.legend(title='Designation Level')  # Add a legend for clarity
plt.show()

df1_wfh_ctype = df1.groupby('Company Type')['WFH Setup Available'].value_counts()

df1_wfh_ctype

df1_wfh_ctype.plot.barh(x='Company Type',y='WFH Setup Available',color='red', figsize=(12,6)).set_title('Company Type vs WFH Setup Availability')

df1_mfs_ctype = df1.groupby('Company Type')['Mental Fatigue Score'].mean()

#df1_mfs_ctype

df1_mfs_ctype.plot.barh(x='Company Type',y='Mental Fatigue Score',color='red', figsize=(12,6)).set_title('Company Type vs Mental Fatigue Score')

#check how many records are there with "service" type company (to create a histogram)

Service_count = (df1['Company Type'] == 'Service').sum()
Product_count = (df1['Company Type'] == 'Product').sum()
print(f"Count of 'Service': {Service_count}")
print(f"Count of 'Product': {Product_count}")

print(f"Count of both:", Product_count + Service_count)



#to check designation level and mental fatigue

df1_designation_vs_fatigue = df1.groupby('Designation Level')['Mental Fatigue Score'].mean()

df1_designation_vs_fatigue

ax = df1_designation_vs_fatigue.plot.bar(figsize=(8,6))
ax.set_title('Average Mental Fatigue Score by Designation Level')
ax.set_xlabel('Designation Level')
ax.set_ylabel('Average Mental Fatigue Score')
plt.show()

# Mental Fatigue Score Distribution by WFH Setup Availability

# Create the histogram
plt.figure(figsize=(8, 6))
plt.hist(df1[df1['WFH Setup Available'] == 'Yes']['Mental Fatigue Score'], alpha=0.5, label='WFH Setup Available')
plt.hist(df1[df1['WFH Setup Available'] == 'No']['Mental Fatigue Score'], alpha=0.5, label='No WFH Setup Available')

# Add labels and title
plt.xlabel('Mental Fatigue Score')
plt.ylabel('Frequency')
plt.title('Mental Fatigue Score Distribution by WFH Setup Availability')
_ = plt.legend()

mental_health_avg = df1.groupby('WFH Setup Available')['Mental Fatigue Score'].value_counts(normalize=True).reset_index(name='Percentage')

# Bar chart
plt.figure(figsize=(10, 6))
sns.barplot(data=mental_health_avg, x='WFH Setup Available', y='Percentage', hue='Mental Fatigue Score', palette='viridis')
plt.title('Mental Fatigue Score by WFH Setup Available')
plt.xlabel('WFH Setup Available')
plt.ylabel('Percentage of Employees')
plt.legend(title='Mental Fatigue Score')
plt.show()

mental_health_avg = df1.groupby('WFH Setup Available')['Burn Rate'].value_counts(normalize=True).reset_index(name='Percentage')

# Bar chart
plt.figure(figsize=(10, 6))
sns.barplot(data=mental_health_avg, x='WFH Setup Available', y='Percentage', hue='Burn Rate', palette='spring')
plt.title('Burn Rate by WFH Setup Available')
plt.xlabel('WFH Setup Available')
plt.ylabel('Percentage of Employees')
plt.legend(title='Burn Rate')
plt.show()

#calculate level of skewness

skewness = df1['Mental Fatigue Score'].skew()
print(f"Skewness: {skewness}")

st.markdown('''If skewness < 0: The data is negatively skewed (tail on the left). The median is more robust.''')

#check distribution

df1['Mental Fatigue Score'].hist(bins=30, figsize=(8, 6), color='blue')
plt.title('Distribution of Mental Fatigue Score')
plt.xlabel('Mental Fatigue Score')
plt.ylabel('Frequency')
plt.show()

st.markdown('''# PART 3: Data Preparation''')

st.markdown('''## Removing Outliers''')

#calculate IQR

Q1 = df1['Mental Fatigue Score'].quantile(0.25)
Q3 = df1['Mental Fatigue Score'].quantile(0.75)
IQR = Q3 - Q1

IQR

#define outlier boundaries

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(lower_bound)
print(upper_bound)

#identify outliers

outliers = df1[(df1['Mental Fatigue Score'] < lower_bound) |
               (df1['Mental Fatigue Score'] > upper_bound)]

outliers

#counting the outliers

num_outliers = len(outliers)
print(f"Number of outliers: {num_outliers}")

percentage_outliers = (num_outliers / len(df1)) * 100
print(f"Percentage of outliers: {percentage_outliers:.2f}%")

#filter out outliers

df1_no_outliers = df1[(df1['Mental Fatigue Score'] >= lower_bound) &
                      (df1['Mental Fatigue Score'] <= upper_bound)]

df1_no_outliers

#confirm anticipated change
print(f"Original dataset size: {len(df1)}")
print(f"Dataset size after removing outliers: {len(df1_no_outliers)}")

#replacing the original dataset

df1 = df1_no_outliers
print(f"Original dataset size: {len(df1)}")

#check outliers with boxplot

df1['Mental Fatigue Score'].plot(kind='box', figsize=(6, 6))
plt.title('Boxplot of Mental Fatigue Score')
plt.show()

st.markdown('''## Data Cleaning Process''')

st.markdown('''Here, we assess the presence of missing values in the copied DataFrame df1 using the `isna()` method, followed by `sum()` to compute the total number of missing values in each column.

The output indicates:


*   No missing values in `Date of Joining`, `Gender`, `Company Type`, `WFH Setup Available`, `Mental Fatigue Score` and `Designation`.
*   Significant missing values in numerical columns:

  *   `Resource Allocation` (1,161)
  * `Burn Rate` (931)


''')

df1.isna().sum()

#df1['Mental Fatigue Score'].unique()
#df1['Burn Rate'].unique()
df1['Resource Allocation'].unique()

st.markdown('''## Handling Missing Value''')

st.markdown('''
In this step, we calculate the percentage of missing values for each column in the `df1` DataFrame using the `isnull()` method to identify missing entries, followed by `mean()` to compute their proportion, and multiplying by 100 to convert it into a percentage. The results are then sorted in descending order using the `sort_values()` method.

The output reveals:


*   `Resource Allocation` and `Burn Rate` follow with **5.72** and **4.59%** missing values, respectively.


All other columns, including `Date of Joining`, `Gender`, `Company Type`, `WFH Setup Available`, `Mental Fatigue Score` and `Designation`, have no missing values (0%).''')

#to get the percentage of missing values for each column, less than 10% max

(df1.isnull().mean() * 100).sort_values(ascending=False)

#dropping all rows with null values

df1 = df1.dropna()

st.markdown('''## Removing Duplicates''')

st.markdown('''
In this section, we address duplicate entries in the `df1` DataFrame to ensure data integrity and avoid redundancy in analysis.

Identifying Duplicates:

*   We use the `duplicated()` method followed by `sum()` to count the total number of duplicate rows in the dataset. The result shows **6** duplicate rows.

Inspecting Duplicates:
*   By applying `df1[df1.duplicated()]`, we display the duplicate rows, which helps us verify the specific entries that are repeated in the dataset.

Removing Duplicates:

*   The `drop_duplicates()` method is employed to eliminate all duplicate rows from the DataFrame, ensuring only unique records remain. The changes are saved back to `df1`.

Verification:

*   After dropping duplicates, we recheck for duplicates using the `duplicated()` method, which confirms that no duplicate rows remain (0).

''')

#check for duplicates

df1.duplicated().sum()

df1[df1.duplicated()]

#dropping all duplicated rows

df1 = df1.drop_duplicates()

df1.duplicated().sum()

st.markdown('''## Transforming Data
''')

st.markdown('''
In this step, we convert the `Date of Joining` column in both the `train_df` and `test_df` DataFrames to the appropriate datetime format using the `pd.to_datetime()` function. This transformation standardizes the column's data type, ensuring it is recognized as a date-time object rather than a generic object (string). By doing so, we enable advanced operations such as extracting temporal features (e.g., year, month, tenure) or performing time-based filtering and analysis.

This preprocessing step is essential for maintaining data integrity and unlocking additional analytical capabilities. Proper handling of date-time data ensures that downstream tasks, such as trend analysis or feature engineering, can be performed accurately and efficiently.''')

#converting data type appropriately

df1['Date of Joining'] = pd.to_datetime(train_df['Date of Joining'])
df2['Date of Joining'] = pd.to_datetime(test_df['Date of Joining'])

st.markdown('''### Cont. to Explore with Transformed Data''')

# to get an overview of the avg mental fatigue score for each month based on joined date

df1['Month-Year'] = df1['Date of Joining'].dt.to_period('M')
monthly_mean_fatigue = df1.groupby('Month-Year')['Mental Fatigue Score'].mean()

ax = monthly_mean_fatigue.plot(kind='line', figsize=(12, 6), marker='o')

# Add labels and title
ax.set_title('Avg Mental Fatigue Score by Month', fontsize=16)
ax.set_xlabel('Month-Year', fontsize=12)
ax.set_ylabel('Avg Mental Fatigue Score', fontsize=12)

plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()


#by median (to discount outliers)

monthly_median_fatigue = df1.groupby('Month-Year')['Mental Fatigue Score'].median()

ax = monthly_median_fatigue.plot(kind='line', figsize=(12, 6), marker='o', color='green')

# Add labels and title
ax.set_title('Median Mental Fatigue Score by Month', fontsize=16)
ax.set_xlabel('Month-Year', fontsize=12)
ax.set_ylabel('Median Mental Fatigue Score', fontsize=12)

plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()

# to get an overview of the avg mental fatigue score for each month based on joined date

df1['Month-Year'] = df1['Date of Joining'].dt.to_period('M')
monthly_mean_fatigue = df1.groupby('Month-Year')['Mental Fatigue Score'].mean()

ax = monthly_mean_fatigue.plot(kind='line', figsize=(12, 6), marker='o')

# Add labels and title
ax.set_title('Avg Mental Fatigue Score by Month', fontsize=16)
ax.set_xlabel('Month-Year', fontsize=12)
ax.set_ylabel('Avg Mental Fatigue Score', fontsize=12)

plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()

st.markdown('''## Normalization''')

# Select only numerical features for correlation analysis
numerical_features = df1.select_dtypes(include=np.number)

# Calculate the correlation matrix
correlation_matrix = numerical_features.corr()

# Display the correlation matrix
print(correlation_matrix)

numerical_features.corr()

plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(numerical_features.corr(), vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);
plt.savefig("correlation_heatmap.png")

def categorize_designation(data):
    if data["Designation"] <= 1.0:
        return 0
    if data["Designation"] > 1.0 and data["Designation"] <= 2.0:
        return 1
    if data["Designation"] > 2.0 and data["Designation"] <= 5.0:
        return 2
    return -1


def categorize_resource(data):
    if data["Resource Allocation"] <= 3.0:
        return 0
    if data["Resource Allocation"] > 3.0 and data["Resource Allocation"] <= 5.0:
        return 1
    if data["Resource Allocation"] > 5.0 and data["Resource Allocation"] <= 10.0:
        return 2
    return -1


def categorize_Mental_Fatigue(data):
    if data["Mental Fatigue Score"] <= 4.0:
        return 0
    if data["Mental Fatigue Score"] > 4.0 and data["Mental Fatigue Score"] <= 5.0:
        return 1
    if data["Mental Fatigue Score"] > 5.0 and data["Mental Fatigue Score"] <= 6.0:
        return 2
    if data["Mental Fatigue Score"] > 6.0 and data["Mental Fatigue Score"] <= 7.0:
        return 3
    if data["Mental Fatigue Score"] > 7.0:
        return 4
    return -1



df1["categorize_designation"] = df1.apply(categorize_designation, axis=1)
df1["categorize_resource"] = df1.apply(categorize_resource, axis=1)
df1["categorize_Mental_Fatigue"] = df1.apply(categorize_Mental_Fatigue, axis=1)

df2["categorize_designation"] = df2.apply(categorize_designation, axis=1)
df2["categorize_resource"] = df2.apply(categorize_resource, axis=1)
df2["categorize_Mental_Fatigue"] = df2.apply(categorize_Mental_Fatigue, axis=1)

print("Cetegorized valued features values:----------", end="\n\n")

print(df1["categorize_designation"].value_counts(), end="\n\n")
print(df1["categorize_resource"].value_counts(), end="\n\n")
print(df1["categorize_Mental_Fatigue"].value_counts(), end="\n\n")

current_date = pd.to_datetime('today')

df1["Date of Joining"] = pd.to_datetime(train_df["Date of Joining"])
df2["Date of Joining"] = pd.to_datetime(test_df["Date of Joining"])

def create_days_count(data):
    return (current_date - data["Date of Joining"])

df1["days_count"] = df1.apply(create_days_count, axis=1)
df1["days_count"] = df1["days_count"].dt.days

df2["days_count"] = df2.apply(create_days_count, axis=1)
df2["days_count"] = df2["days_count"].dt.days

print(df1["Gender"].value_counts(), end="\n\n")
print(df1["Company Type"].value_counts(), end="\n\n")
print(df1["WFH Setup Available"].value_counts(), end="\n\n")

one = 1
zero = 0

def gender_encoder(data):
    if data["Gender"] == "Female":
        return one
    return zero


def wfh_setup_encoder(data):
    if data["WFH Setup Available"] == "Yes":
        return one
    return zero


def company_encoder(data):
    if data["Company Type"] == "Service":
        return one
    return zero



df1["Gender"] = df1.apply(gender_encoder, axis=1)
df1["WFH Setup Available"] = df1.apply(wfh_setup_encoder, axis=1)
df1["Company Type"] = df1.apply(company_encoder, axis=1)

df2["Gender"] = df2.apply(gender_encoder, axis=1)
df2["WFH Setup Available"] = df2.apply(wfh_setup_encoder, axis=1)
df2["Company Type"] = df2.apply(company_encoder, axis=1)

norm_cols = ["Designation", "Resource Allocation", "Mental Fatigue Score"]
#+ ["days_count", "categorize_designation", "categorize_resource", "categorize_Mental_Fatigue"]

train_df1_min = df1[norm_cols].min()
train_df1_max = df1[norm_cols].max()

df1[norm_cols] = (df1[norm_cols] - train_df1_min)/(train_df1_max - train_df1_min)

df2[norm_cols] = (df2[norm_cols] - train_df1_min)/(train_df1_max - train_df1_min)

df1.head()

# Since 'Employee ID' is the index, reset it to a column first
df1.reset_index(inplace=True)
# Now you can drop the columns
df1.drop(['Date of Joining', "Employee ID"], axis=1, inplace=True)

# Similarly for df2 (test_df copy) if 'Employee ID' was also set as index
df2.reset_index(inplace=True)
clean_df_test = df2.drop(['Date of Joining', "Employee ID"], axis=1)

# Select only numerical features for correlation analysis
numerical_features = df1.select_dtypes(include=np.number)

# Calculate the correlation matrix
correlation_matrix = numerical_features.corr()

# Display the correlation matrix
print(correlation_matrix)

plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(numerical_features.corr(), vmin=-1, vmax=1, annot=True, cmap='viridis_r')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);
plt.savefig("correlation_heatmap.png")

clean_df = df1.copy()

df1.to_csv("clean_df_train.csv", index=False)
train_file_path = "./clean_df_train.csv"
new_df = pd.read_csv(train_file_path)

clean_df_test.to_csv("clean_df_test.csv", index=False)
test_file_path = "./clean_df_test.csv"
new_df_test = pd.read_csv(test_file_path)

new_df_test.head()



st.markdown('''# PART 4: Modelling''')

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import RandomizedSearchCV

import xgboost


from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(clean_df.loc[:, clean_df.columns != "Burn Rate"],
                                                    clean_df.loc[:, clean_df.columns == "Burn Rate"],
                                                    test_size=0.2,
                                                    random_state=42)

def print_r2_score(y_train, train_pred, y_test, test_pred):
    r2_train = r2_score(y_train, train_pred)
    print("Score LR Train: "+str(round(100*r2_train, 4))+" %")

    r2_test = r2_score(y_test, test_pred)
    print("Score LR Test: "+str(round(100*r2_test, 4))+" %")

sub = pd.read_csv(test_url)
sub = sub.loc[:, ["Employee ID"]]


numerical_features = df1.select_dtypes(include=np.number).columns.tolist()
numerical_features.remove('Burn Rate') # Remove the target variable
X = df1[numerical_features]  # Features (numerical columns)
y = df1['Burn Rate']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.markdown('''## Linear Regression''')

model = LinearRegression()
model.fit(X_train, y_train)

# Predictions on training and testing sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate R-squared for training set
r2_train = r2_score(y_train, y_train_pred)
print(f"R-squared accuracy (Training): {r2_train}")

# Calculate R-squared for testing set
r2_test = r2_score(y_test, y_test_pred)
print(f"R-squared accuracy (Testing): {r2_test}")

st.markdown('''## Random Forest''')

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

numerical_features = df1.select_dtypes(include=np.number).columns.tolist()
numerical_features.remove('Burn Rate')
X = df1[numerical_features]
y = df1['Burn Rate']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest model
rf_model = RandomForestRegressor(random_state=42)  # You can adjust hyperparameters here
rf_model.fit(X_train, y_train)

# Predictions on training and testing sets
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

# Calculate R-squared for training set
r2_train = r2_score(y_train, y_train_pred)
print(f"R-squared accuracy (Training): {r2_train}")

# Calculate R-squared for testing set
r2_test = r2_score(y_test, y_test_pred)
print(f"R-squared accuracy (Testing): {r2_test}")

st.markdown('''Reasoning:

Import necessary libraries: RandomForestRegressor, r2_score, train_test_split are imported for model creation, evaluation, and data splitting.

Data preparation: This part remains the same as in your Linear Regression code – selecting numerical features, defining target variable, and splitting data
.
Model creation and training: A RandomForestRegressor object is created (you can customize hyperparameters like n_estimators, max_depth, etc.). The model is then trained using fit on the training data.

Prediction and evaluation: Predictions are made on both training and testing sets using predict. The R-squared score is calculated using r2_score to evaluate the model's performance.

By replacing the Linear Regression model with Random Forest, you can leverage the power of ensemble learning for potentially better predictive accuracy. Remember to experiment with different hyperparameter settings to further optimize your model.''')

st.markdown('''## Kmeans''')

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Select relevant features for clustering, including 'Burn Rate'
features_for_clustering = ['Mental Fatigue Score', 'Resource Allocation', 'Designation', 'Burn Rate']
X = clean_df[features_for_clustering]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the Elbow method
wcss = []  # Within-cluster sum of squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow method graph
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()  # This will display the graph to help you choose the optimal number of clusters

# Based on the Elbow method, choose the optimal number of clusters (e.g., 3)
optimal_num_clusters = 3  # You might need to adjust this based on the Elbow method graph

# Apply KMeans clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=42)
clean_df['cluster'] = kmeans.fit_predict(X_scaled)

# Analyze the clusters (e.g., by examining the characteristics of data points in each cluster)
print(clean_df.groupby('cluster')[['Mental Fatigue Score', 'Resource Allocation', 'Designation', 'Burn Rate']].mean())
# You can further visualize the clusters using scatter plots or other techniques if needed.

# Analyze the clusters
cluster_means = clean_df.groupby('cluster')[['Mental Fatigue Score', 'Resource Allocation', 'Designation', 'Burn Rate']].mean()

print(cluster_means)

st.markdown('''To determine if the clustering is good, consider the following:

Separation between clusters: Examine the cluster_means output. If the means of the features (especially 'Burn Rate') are significantly different across clusters, it suggests good separation and potentially meaningful clusters.

Cluster size: Check the distribution of data points across clusters. Ideally, clusters should have a reasonable number of data points and not be too imbalanced.''')

from sklearn.metrics import silhouette_score

# Select relevant features for clustering
features_for_clustering = ['Mental Fatigue Score', 'Resource Allocation', 'Designation', 'Burn Rate']
X = clean_df[features_for_clustering]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calculate the silhouette score
silhouette_avg = silhouette_score(X_scaled, clean_df['cluster'])
print(f"Silhouette Score: {silhouette_avg}")

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score

# Select relevant features for clustering.
features_for_clustering = ['Mental Fatigue Score', 'Resource Allocation', 'Designation', 'Burn Rate']
X = clean_df[features_for_clustering]

# Scale the features using StandardScaler.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans clustering.
kmeans = KMeans(n_clusters=3, random_state=42)  # Adjust n_clusters as needed
predicted_labels = kmeans.fit_predict(X_scaled)

# Instead of using 'true_labels' which doesn't exist, we will
# assume 'cluster' column from previous KMeans run as the true labels for demonstration.
# Replace 'cluster' with the actual ground truth labels column name if you have one.
true_labels = clean_df['cluster']

# Calculate Adjusted Rand Index (ARI).
ari = adjusted_rand_score(true_labels, predicted_labels)

# Calculate Silhouette Score.
silhouette_avg = silhouette_score(X_scaled, predicted_labels)

# Print Results:
print(f"Adjusted Rand Index (ARI): {ari}")
print(f"Silhouette Score: {silhouette_avg}")

st.markdown('''Adjusted Rand Index (ARI): 1.0

An ARI of 1.0 indicates perfect agreement between the predicted cluster assignments and the ground truth labels. This means your clustering algorithm has perfectly recovered the underlying structure of the data based on the ground truth (which, in this case, is assumed to be the 'cluster' column created in a previous KMeans run).
Silhouette Score: 0.3726796440837502

The Silhouette Score ranges from -1 to +1.
A score closer to +1 suggests that the data points are well clustered and assigned to the appropriate clusters.
A score closer to 0 indicates that the clusters are overlapping or poorly separated.
A score closer to -1 suggests that data points may have been assigned to the wrong clusters.
Interpretation of Combined Results

High ARI (1.0) and Moderate Silhouette Score (0.37): This combination suggests that your clustering algorithm has perfectly matched the ground truth labeling, but the clusters themselves might have some degree of overlap or aren't as clearly separated as they could be.
In simpler terms:

Your algorithm is doing a great job of assigning data points to the same clusters as the ground truth.
However, the clusters themselves might not be as distinct or well-defined as you might ideally want. There could be some fuzziness at the boundaries between clusters.
Possible Reasons for Moderate Silhouette Score with Perfect ARI

Ground truth limitations: The ground truth labels (assumed to be the 'cluster' column from a previous KMeans run) might not perfectly represent the underlying structure of the data.
Data characteristics: The data itself might have some inherent overlap or ambiguity, making it difficult to achieve perfectly separated clusters, even with a correct clustering algorithm.
Cluster shape: KMeans assumes that clusters are spherical and equally sized. If the actual clusters in your data have different shapes or densities, KMeans might not be able to capture them perfectly, leading to a lower Silhouette Score.
Further Investigation

Visualize clusters: Try visualizing your clusters using scatter plots or other visualization techniques to see if you can observe any overlap or areas of ambiguity.
Explore other clustering algorithms: If you're concerned about the Silhouette Score, you could try experimenting with other clustering algorithms (e.g., DBSCAN, hierarchical clustering) that might be better suited to the shape of your data.
Reassess features: Consider whether the features you're using for clustering are the most relevant ones for capturing the underlying structure of the data. You might try adding or removing features to see if it improves the Silhouette Score.''')

# Assuming you have already calculated these variables:
# r2_train (for Linear Regression and Random Forest)
# r2_test (for Linear Regression and Random Forest)
# silhouette_avg (for KMeans Clustering)
# ari (for KMeans Clustering)

import pandas as pd

# Store the r2 scores in separate variables or a list if you have multiple runs
# For example, if you have the following:
r2_train_lr = r2_train  # R-squared for Linear Regression training
r2_train_rf = r2_test  # R-squared for Random Forest training
r2_test_lr = r2_test  # R-squared for Linear Regression testing
r2_test_rf = r2_test  # R-squared for Random Forest testing

data = {
    'Model': ['Linear Regression', 'Random Forest', 'KMeans Clustering'],
    'R-squared (Training)': [r2_train_lr, r2_train_rf, ''],  # Use the individual r2 scores
    'R-squared (Testing)': [r2_test_lr, r2_test_rf, ''],    # Use the individual r2 scores
    'Silhouette Score': ['', '', silhouette_avg],
    'Adjusted Rand Index': ['', '', ari]
}

df_models = pd.DataFrame(data)
df_models

# Model

from matplotlib import pyplot as plt
import seaborn as sns
df_models.groupby('Model').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
plt.gca().spines[['top', 'right',]].set_visible(False)

# prompt: pie chart of the model for train and test data into 2 pie chart for all model

import matplotlib.pyplot as plt

# Assuming you have these variables from your previous code:
# r2_train_lr, r2_train_rf, r2_test_lr, r2_test_rf, silhouette_avg, ari

# Sample data (replace with your actual data)
r2_train_lr = 0.8
r2_train_rf = 0.9
r2_test_lr = 0.75
r2_test_rf = 0.85
silhouette_avg = 0.6
ari = 0.9

# Create pie charts for training data
labels_train = ['Linear Regression', 'Random Forest']
sizes_train = [r2_train_lr, r2_train_rf]
colors_train = ['skyblue', 'lightcoral']

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.pie(sizes_train, labels=labels_train, colors=colors_train, autopct='%1.1f%%', startangle=90)
plt.title('R-squared (Training)')


# Create pie charts for testing data
labels_test = ['Linear Regression', 'Random Forest']
sizes_test = [r2_test_lr, r2_test_rf]
colors_test = ['skyblue', 'lightcoral']

plt.subplot(1, 2, 2)
plt.pie(sizes_test, labels=labels_test, colors=colors_test, autopct='%1.1f%%', startangle=90)
plt.title('R-squared (Testing)')

plt.tight_layout()
plt.show()

# For KMeans, you can display the silhouette score and ARI separately as they don't directly map to proportions like R-squared.
print(f"Silhouette Score for KMeans: {silhouette_avg}")
print(f"Adjusted Rand Index for KMeans: {ari}")

st.markdown('''# PART 5: Evaluation''')

import pandas as pd
# Assuming you have already calculated these variables:
r2_train_lr = 0.8  # Replace with your actual value
r2_train_rf = 0.9  # Replace with your actual value
r2_test_lr = 0.7  # Replace with your actual value
r2_test_rf = 0.8  # Replace with your actual value
silhouette_avg = 0.5 # Replace with your actual value
ari = 0.8 # Replace with your actual value

# Create a DataFrame for better visualization (optional but recommended)
data = {
    'Metric': ['R-squared (Training)', 'R-squared (Testing)', 'Silhouette Score', 'Adjusted Rand Index'],
    'Linear Regression': [r2_train_lr, r2_test_lr, '', ''],
    'Random Forest': [r2_train_rf, r2_test_rf, '', ''],
    'KMeans': ['', '', silhouette_avg, ari]
}

df_evaluation = pd.DataFrame(data)

# Replace empty strings with NaN and then fill NaN with 0
df_evaluation = df_evaluation.replace('', pd.NA).fillna(0)


print(df_evaluation)


#Plot the results (optional)
df_evaluation.plot(x='Metric', kind='bar')
plt.title('Model Evaluation Metrics')
plt.ylabel('Score')
plt.xlabel('Metric')
plt.xticks(rotation=45)
plt.legend(title='Model')
plt.tight_layout()
plt.show()

st.markdown('''# PART 6: Deployment''')


