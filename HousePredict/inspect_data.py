import pandas as pd
file_path = r"c:/Users/Teslas Laptops/OneDrive/Desktop/DataCrumbs/DataCrumbs_Projects/HousePredict/data/data.csv"
df = pd.read_csv(file_path)
print('Rows:', df.shape[0])
print('Columns:', df.shape[1])
print('Column list:', list(df.columns))
print('Dtypes:\n', df.dtypes)
print('Missing values per column:\n', df.isnull().sum())
print('Duplicate rows:', df.duplicated().sum())
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
num_cols = df.select_dtypes(exclude=['object']).columns.tolist()
print('Categorical columns:', cat_cols)
print('Numeric columns:', num_cols)
print('Target column presumed "price" stats:\n', df['price'].describe())
