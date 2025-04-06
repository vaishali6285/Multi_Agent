import pandas as pd

# Load datasets
customer_data = pd.read_csv('data/customer_data_collection.csv')
product_data = pd.read_csv('data/product_recommendation_data.csv')

# Print column names for debugging
print("Column names in Customer Data:")
print(customer_data.columns)

print("Column names in Product Data:")
print(product_data.columns)

# Fill missing values based on column type for Customer Data
for column in customer_data.columns:
    if customer_data[column].dtype == 'float64' or customer_data[column].dtype == 'int64':  # Numeric columns
        customer_data[column] = customer_data[column].fillna(0)  # Replace NaNs with 0 for numeric columns
    else:  # Non-numeric columns
        customer_data[column] = customer_data[column].fillna("Unknown")  # Replace NaNs with "Unknown" for non-numeric columns

# Fill missing values based on column type for Product Data
for column in product_data.columns:
    if product_data[column].dtype == 'float64' or product_data[column].dtype == 'int64':  # Numeric columns
        product_data[column] = product_data[column].fillna(0)
    else:  # Non-numeric columns
        product_data[column] = product_data[column].fillna("Unknown")

# Normalize Gender column in Customer Data
if 'Gender' in customer_data.columns:  # Using the correct column name
    customer_data['Gender'] = customer_data['Gender'].map({'Male': 'M', 'Female': 'F', None: 'Unknown'})
else:
    print("The 'Gender' column is missing from the dataset. Please verify your data.")

# Preview the cleaned datasets for verification
print("\nCleaned Customer Data:")
print(customer_data.head())

print("\nCleaned Product Data:")
print(product_data.head())

# Save the cleaned datasets to new files (optional)
customer_data.to_csv('./data/cleaned_customer_data.csv', index=False)
product_data.to_csv('./data/cleaned_product_data.csv', index=False)
print("\nCleaned data has been saved as 'cleaned_customer_data.csv' and 'cleaned_product_data.csv'.")