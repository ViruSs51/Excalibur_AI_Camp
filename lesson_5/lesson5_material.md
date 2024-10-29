# **intro to pandas**

## **Quick Overview**
- **Pandas** is a powerful data manipulation library for Python. It is built on top of NumPy and provides easy-to-use data structures and data analysis tools for Python programming language.

### **What is Pandas?**
- **Definition:**
  - Pandas is an open-source Python library primarily used for data analysis and manipulation. It provides high-level data structures and functions designed to make working with structured data fast, easy, and intuitive.

### **Key Features of Pandas:**
1. **Data Structures:**
   - **Series**: A 1-dimensional labeled array capable of holding data of any type.
   - **DataFrame**: A 2-dimensional labeled data structure with columns of potentially different types, akin to a spreadsheet or SQL table.
2. **Data Handling:**
   - Efficiently handles large datasets, allowing for easy data manipulation.
   - Allows for time-series analysis and manipulation.
3. **Data Input/Output:**
   - Support for reading and writing data to various formats, including CSV, Excel, SQL databases, JSON, and more.
4. **Data Analysis Tools:**
   - Offers advanced data aggregation and transformation options akin to groups and joins in SQL.
5. **Indexing and Selecting Data:**
   - Powerful data selection methods.
6. **Handling Missing Data:**
   - Built-in methods to detect, remove, or replace missing data.



```python
import pandas as pd
import numpy as np
```

### **Definition:**
- A **Pandas Series** is a one-dimensional labeled array that can hold any data type (integers, floats, strings, etc.). It is similar to a Python list or dictionary but provides more functionality.

### **Characteristics of a Pandas Series:**
1. **Index:** Each element in a Series has an associated label (index), which can be custom-defined.
2. **Homogeneous Data:** All elements are of the same data type.
3. **Data Type:** Events, financial data, or any other structured data types can be stored.


```python
# Creating a Series from a list
data = [10, 20, 30, 40]
series = pd.Series(data)
print(series)
```

    0    10
    1    20
    2    30
    3    40
    dtype: int64



```python
# Creating a Series from a dictionary
data_dict = {'a': 10, 'b': 20, 'c': 30}
series_dict = pd.Series(data_dict)
print(series_dict)
```

    a    10
    b    20
    c    30
    dtype: int64



```python
# Creating a Series with Custom Index
data = [10, 20, 30, 40]
index = ['a', 'b', 'c', 'd']
series_custom = pd.Series(data, index=index)
print(series_custom)
```

    a    10
    b    20
    c    30
    d    40
    dtype: int64


### **Attributes of a Pandas Series:**


```python
# Accessing the index, values, and dtype of a Series
print(series.index)
print(series.values)
print(series.dtype)
```

    RangeIndex(start=0, stop=4, step=1)
    [100  20  30  40]
    int64


### **Accessing Elements in a Series**
- Use **integer-based indexing** or **label-based indexing** to access elements.


```python
# Elements by index in a Series
print(series_custom[0]) # Output: 10 (using integer location) will be deprecated
print(series_custom['b']) # Output: 20 (using custom index)
```

    10
    20


    /var/folders/dt/tgmydmjd1g354lgxy3yn76g40000gn/T/ipykernel_22447/2371884906.py:2: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      print(series_custom[0]) # Output: 10 (using integer location)


### **Slicing a Series:**


```python
print(series[1:3])  # Output: Series with elements at index 1 and 2
```

    1    20
    2    30
    dtype: int64


### **Custom Indexing:**


```python
# Creating a Series with custom index
series3 = pd.Series([100, 200, 300], index=['x', 'y', 'z'])
print(series3)

# Accessing elements using custom index labels
print(series3['y'])   
print(series3[['x', 'z']])  # Selecting multiple elements
```

    x    100
    y    200
    z    300
    dtype: int64
    200
    x    100
    z    300
    dtype: int64


#### **2. DataFrame**

**Creation from various data sources (dicts, lists, NumPy arrays):**


```python
import numpy as np

# Creating a DataFrame from a dictionary
data_dict = {'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 30, 35]}
df1 = pd.DataFrame(data_dict)
print(df1)
print("---------------------")

# Creating a DataFrame from a list of lists
data_list = [['Alice', 25], ['Bob', 30], ['Charlie', 35]]
df2 = pd.DataFrame(data_list, columns=['Name', 'Age'])
print(df2)
print("---------------------")

# Creating a DataFrame from a NumPy array
data_array = np.array([['Alice', 25], ['Bob', 30], ['Charlie', 35]])
df3 = pd.DataFrame(data_array, columns=['Name', 'Age'])
print(df3)
print("---------------------")
```

          Name  Age
    0    Alice   25
    1      Bob   30
    2  Charlie   35
    ---------------------
          Name  Age
    0    Alice   25
    1      Bob   30
    2  Charlie   35
    ---------------------
          Name Age
    0    Alice  25
    1      Bob  30
    2  Charlie  35
    ---------------------


### **Attributes (columns, index, values, shape, dtypes):**


```python
# Accessing DataFrame attributes
print(df1.columns)
print("---------------------")
print(df1.index)
print("---------------------")
print(df1.values)
print("---------------------")
print(df1.shape)
print("---------------------")
print(df1.dtypes)
print("---------------------")
```

    Index(['Name', 'Age'], dtype='object')
    ---------------------
    RangeIndex(start=0, stop=3, step=1)
    ---------------------
    [['Alice' 25]
     ['Bob' 30]
     ['Charlie' 35]]
    ---------------------
    (3, 2)
    ---------------------
    Name    object
    Age      int64
    dtype: object
    ---------------------


### **Basic Operations (head, tail, info, describe):**


```python
# Displaying the first few rows
print(df1.head())
print("---------------------")

# Displaying the last few rows
print(df1.tail())
print("---------------------")

# Getting DataFrame info
print(df1.info())
print("---------------------")

# Descriptive statistics of DataFrame
print(df1.describe())
print("---------------------")
```

          Name  Age
    0    Alice   25
    1      Bob   30
    2  Charlie   35
    ---------------------
          Name  Age
    0    Alice   25
    1      Bob   30
    2  Charlie   35
    ---------------------
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3 entries, 0 to 2
    Data columns (total 2 columns):
     #   Column  Non-Null Count  Dtype 
    ---  ------  --------------  ----- 
     0   Name    3 non-null      object
     1   Age     3 non-null      int64 
    dtypes: int64(1), object(1)
    memory usage: 180.0+ bytes
    None
    ---------------------
            Age
    count   3.0
    mean   30.0
    std     5.0
    min    25.0
    25%    27.5
    50%    30.0
    75%    32.5
    max    35.0
    ---------------------


### **Data Input/Output**

#### **Writing Data:**


```python
# Writing DataFrame to a CSV file
df1.to_csv('output.csv', index=False)

# Writing DataFrame to an Excel file
df1.to_excel('output.xlsx', index=False) # you might need to install openpyxl with `pip install openpyxl`

# Writing DataFrame to a SQL database
import sqlite3

# Creating a connection to a SQLite database
# conn = sqlite3.connect('example.db')

# df1.to_sql('table_name', conn, if_exists='replace', index=False)

# Writing DataFrame to a JSON file
df1.to_json('output.json', orient='records')

```

#### **Reading Data:**


```python
# Reading data from a CSV file
df_csv = pd.read_csv('output.csv')
print(df_csv.head())

# Reading data from an Excel file
df_excel = pd.read_excel('output.xlsx')
print(df_excel.head())

# Reading data from a SQL database
# df_sql = pd.read_sql('SELECT * FROM table_name', conn)
# print(df_sql.head())

# Reading data from a JSON file
df_json = pd.read_json('output.json')
print(df_json.head())
```

          Name  Age
    0    Alice   25
    1      Bob   30
    2  Charlie   35
          Name  Age
    0    Alice   25
    1      Bob   30
    2  Charlie   35
          Name  Age
    0    Alice   25
    1      Bob   30
    2  Charlie   35


### **Handling Missing Data**

#### **Detection:**


```python
# isnull():
# Detecting missing values
data_with_nan = {'A': [1, 2, None], 'B': [None, 2, 3]}
df_with_nan = pd.DataFrame(data_with_nan)
print(df_with_nan.isnull())

# notnull():
# Detecting non-missing values
print(df_with_nan.notnull())
```

           A      B
    0  False   True
    1  False  False
    2   True  False
           A      B
    0   True  False
    1   True   True
    2  False   True


#### **Handling:**


```python
# dropna():
# Dropping rows with missing values
print(df_with_nan.dropna())
print("---------------------")

# Dropping columns with missing values
print(df_with_nan.dropna(axis=1))
print("---------------------")

# fillna():
# Filling missing values with a constant
print(df_with_nan.fillna(0))
print("---------------------")

# Filling missing values with a method (forward fill)
# print(df_with_nan.fillna(method='ffill')) # willl be deprecated
print(df_with_nan.ffill())
print("---------------------")

# Filling missing values with a method (backward fill)
# print(df_with_nan.fillna(method='bfill')) # willl be deprecated
print(df_with_nan.bfill())
print("---------------------")
```

         A    B
    1  2.0  2.0
    ---------------------
    Empty DataFrame
    Columns: []
    Index: [0, 1, 2]
    ---------------------
         A    B
    0  1.0  0.0
    1  2.0  2.0
    2  0.0  3.0
    ---------------------
         A    B
    0  1.0  NaN
    1  2.0  2.0
    2  2.0  3.0
    ---------------------
         A    B
    0  1.0  2.0
    1  2.0  2.0
    2  NaN  3.0
    ---------------------


### **Indexing and Selecting Data**

#### **Indexing:**



```python
# Setting a new index
df1.set_index('Name', inplace=True)
print(df1)

# Resetting index back to default
df1.reset_index(inplace=True)
print(df1)
```

             Age
    Name        
    Alice     25
    Bob       30
    Charlie   35
          Name  Age
    0    Alice   25
    1      Bob   30
    2  Charlie   35



```python
# Hierarchical Indexing:
# Creating a DataFrame
df_multi = pd.DataFrame({
    'Animal': ['Cat', 'Dog', 'Cat', 'Dog'],
    'Color': ['White', 'Brown', 'Black', 'White'],
    'Age': [2, 3, 4, 5]
})

# Setting hierarchical index
df_multi.set_index(['Animal', 'Color'], inplace=True)
print(df_multi)
```

                  Age
    Animal Color     
    Cat    White    2
    Dog    Brown    3
    Cat    Black    4
    Dog    White    5



```python
# Boolean Indexing:
# Selecting rows based on a condition
print(df1[df1['Age'] > 30])
```

          Name  Age
    2  Charlie   35


#### **Selecting Data:**

##### **Selection by Label (`loc`)** Detailed Use Cases


```python
# Selecting Specific Rows and Columns
# Sample DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [24, 27, 22, 32],
    'Score': [85, 90, 88, 76]
}
df = pd.DataFrame(data)

# Selecting a specific row by label
print(df.loc[1])
print("---------------------")

# Selecting a specific cell by row and column labels
print(df.loc[1, 'Age'])
print("---------------------")

# Selecting multiple rows and columns
print(df.loc[1:2, ['Name', 'Score']])
print("---------------------")
```

    Name     Bob
    Age       27
    Score     90
    Name: 1, dtype: object
    ---------------------
    27
    ---------------------
          Name  Score
    1      Bob     90
    2  Charlie     88
    ---------------------



```python
# Selecting Based on Boolean Conditions
# Selecting rows where Age > 25
print(df.loc[df['Age'] > 25])
print("---------------------")

# Selecting rows where Age > 25 and selecting specific columns
print(df.loc[df['Age'] > 25, ['Name', 'Age']])
print("---------------------")

```

        Name  Age  Score
    1    Bob   27     95
    3  David   32     76
    ---------------------
        Name  Age
    1    Bob   27
    3  David   32
    ---------------------
          Name  Age  Score
    0    Alice   24    100
    1      Bob   27     95
    2  Charlie   22    100
    3    David   32     76
    ---------------------
          Name  Age  Score
    0    Alice   24    100
    1      Bob   27     95
    2  Charlie   22    100
    3    David   32     76
    ---------------------



```python
# Modifying Data Using `loc`
# Changing the score for 'Bob'
df.loc[df['Name'] == 'Bob', 'Score'] = 95
print(df)
print("---------------------")

# Setting the score to 100 for all rows where Age < 25
df.loc[df['Age'] < 25, 'Score'] = 100
print(df)
print("---------------------")
```

          Name  Age  Score
    0    Alice   24     85
    1      Bob   27     95
    2  Charlie   22     88
    3    David   32     76
    ---------------------
          Name  Age  Score
    0    Alice   24    100
    1      Bob   27     95
    2  Charlie   22    100
    3    David   32     76
    ---------------------



```python
#Slicing with `loc`
# Selecting all rows and specific columns
print(df.loc[:, ['Name', 'Score']])
print("---------------------")

# Selecting a range of rows and specific columns
print(df.loc[1:3, 'Name':'Score'])
print("---------------------")
```

          Name  Score
    0    Alice    100
    1      Bob     95
    2  Charlie    100
    3    David     76
    ---------------------
          Name  Age  Score
    1      Bob   27     95
    2  Charlie   22    100
    3    David   32     76
    ---------------------


### **Selection by Position (`iloc`) Detailed Use Cases**


```python
# Selecting Specific Rows and Columns by Position
# Selecting a specific row by position
print(df.iloc[1])
print("---------------------")

# Selecting a specific cell by row and column positions
print(df.iloc[1, 1])
print("---------------------")

# Selecting multiple rows and columns
print(df.iloc[1:3, [0, 2]])
print("---------------------")
```

    Name     Bob
    Age       27
    Score     95
    Name: 1, dtype: object
    ---------------------
    27
    ---------------------
          Name  Score
    1      Bob     95
    2  Charlie    100
    ---------------------



```python
# Slicing with `iloc`
# Selecting all rows and specific columns by position
print(df.iloc[:, [0, 2]])
print("---------------------")

# Selecting a range of rows and columns by position
print(df.iloc[1:3, 0:2])
print("---------------------")
```

          Name  Score
    0    Alice    100
    1      Bob     95
    2  Charlie    100
    3    David     76
    ---------------------
          Name  Age
    1      Bob   27
    2  Charlie   22
    ---------------------



```python
# Boolean Indexing with `iloc`
# Using Boolean indexing to select data based on a condition (not common with iloc)
indices = df[df['Age'] > 25].index
print(df.iloc[indices])
```

        Name  Age  Score
    1    Bob   27     95
    3  David   32     76



```python
# Modifying Data Using `iloc`
# Changing the score at the second row
df.iloc[1, 2] = 90
print(df)
print("---------------------")

# Setting the scores to 100 for the first two rows
df.iloc[0:2, 2] = 100
print(df)
print("---------------------")
```

          Name  Age  Score
    0    Alice   24    100
    1      Bob   27     90
    2  Charlie   22    100
    3    David   32     76
    ---------------------
          Name  Age  Score
    0    Alice   24    100
    1      Bob   27    100
    2  Charlie   22    100
    3    David   32     76
    ---------------------


### **Complex Use Cases Combining `loc` and `iloc`**


```python
# Combining Conditions and Slicing
# Selecting rows where Age > 25 and then slicing columns
print(df.loc[df['Age'] > 25, 'Name':'Score'])
print("---------------------")

# Modifying specific cells based on conditions
df.loc[df['Age'] > 25, 'Score'] = 90
print(df)
print("---------------------")

# Selecting specific rows and then using iloc to further slice columns
subset = df.loc[df['Age'] > 25]
print(subset.iloc[:, 0:2])
print("---------------------")
```

        Name  Age  Score
    1    Bob   27    100
    3  David   32     76
          Name  Age  Score
    0    Alice   24    100
    1      Bob   27     90
    2  Charlie   22    100
    3    David   32     90
        Name  Age
    1    Bob   27
    3  David   32


#### Using MultiIndex with `loc` and `iloc`


```python
# Sample DataFrame with MultiIndex
arrays = [
    ['A', 'A', 'B', 'B'],
    ['one', 'two', 'one', 'two']
]
index = pd.MultiIndex.from_arrays(arrays, names=('first', 'second'))
df_multi = pd.DataFrame({'value': [10, 20, 30, 40]}, index=index)

# Using loc on MultiIndex
print(df_multi.loc['A'])
print("---------------------")
print(df_multi.loc['A', 'one'])
print("---------------------")

# Using iloc on MultiIndex
print(df_multi.iloc[0])
print("---------------------")
print(df_multi.iloc[0:2])
print("---------------------")
```

            value
    second       
    one        10
    two        20
    ---------------------
    value    10
    Name: (A, one), dtype: int64
    ---------------------
    value    10
    Name: (A, one), dtype: int64
    ---------------------
                  value
    first second       
    A     one        10
          two        20
    ---------------------

