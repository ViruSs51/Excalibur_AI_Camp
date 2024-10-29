# **Exploratory Data Analysis Wine Quality dataset**


Disclaimer: The majority of the content in this notebook was from https://github.com/PacktPublishing/Hands-on-Exploratory-Data-Analysis-with-Python/tree/master?tab=readme-ov-file so make sure to check them out, they have great content!


**Definition of EDA:** Understanding EDA as a process for examining datasets, identifying patterns, detecting anomalies, testing hypotheses, and understanding the dataset's underlying structure.

**Why?:** How EDA provides insights into data and lays the foundation for further statistical or machine learning models.


```python
import pandas as pd
```


```python
df_red = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", delimiter=";")
df_white = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", delimiter=";")
```


```python
df_red.columns
```




    Index(['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
           'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
           'pH', 'sulphates', 'alcohol', 'quality'],
          dtype='object')




```python
df_white.columns
```




    Index(['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
           'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
           'pH', 'sulphates', 'alcohol', 'quality'],
          dtype='object')



As shown in this output, the dataset contains the following columns:
- ***Fixed acidity***: It indicates the amount of tartaric acid in wine and is measured in g/dm3.
- ***Volatile acidity***: It indicates the amount of acetic acid in the wine. It is measured in g/dm3.
- ***Citric acid***: It indicates the amount of citric acid in the wine. It is also measured in g/dm3.
- ***Residual sugar***: It indicates the amount of sugar left in the wine after the fermentation process is done. It is also measured in g/dm3.
- ***Free sulfur dioxide***: It measures the amount of sulfur dioxide (SO2) in free form. It is also measured in g/dm3.
- ***Total sulfur dioxide***: It measures the total amount of SO2 in the wine. This chemical works as an antioxidant and antimicrobial agent.
- ***Density***: It indicates the density of the wine and is measured in g/dm3.
- ***pH***: It indicates the pH value of the wine. The range of value is between 0 to 14.0, which indicates very high acidity, and 14 indicates basic acidity.
- ***Sulphates***: It indicates the amount of potassium sulphate in the wine. It is also measured in g/dm3.
- ***Alcohol***: It indicates the alcohol content in the wine.
- ***Quality***: It indicates the quality of the wine, which is ranged from 1 to 10. Here,
the higher the value is, the better the wine.


```python
df_red.dtypes
```




    fixed acidity           float64
    volatile acidity        float64
    citric acid             float64
    residual sugar          float64
    chlorides               float64
    free sulfur dioxide     float64
    total sulfur dioxide    float64
    density                 float64
    pH                      float64
    sulphates               float64
    alcohol                 float64
    quality                   int64
    dtype: object




```python
df_red.iloc[100:110]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100</th>
      <td>8.3</td>
      <td>0.610</td>
      <td>0.30</td>
      <td>2.1</td>
      <td>0.084</td>
      <td>11.0</td>
      <td>50.0</td>
      <td>0.9972</td>
      <td>3.40</td>
      <td>0.61</td>
      <td>10.2</td>
      <td>6</td>
    </tr>
    <tr>
      <th>101</th>
      <td>7.8</td>
      <td>0.500</td>
      <td>0.30</td>
      <td>1.9</td>
      <td>0.075</td>
      <td>8.0</td>
      <td>22.0</td>
      <td>0.9959</td>
      <td>3.31</td>
      <td>0.56</td>
      <td>10.4</td>
      <td>6</td>
    </tr>
    <tr>
      <th>102</th>
      <td>8.1</td>
      <td>0.545</td>
      <td>0.18</td>
      <td>1.9</td>
      <td>0.080</td>
      <td>13.0</td>
      <td>35.0</td>
      <td>0.9972</td>
      <td>3.30</td>
      <td>0.59</td>
      <td>9.0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>103</th>
      <td>8.1</td>
      <td>0.575</td>
      <td>0.22</td>
      <td>2.1</td>
      <td>0.077</td>
      <td>12.0</td>
      <td>65.0</td>
      <td>0.9967</td>
      <td>3.29</td>
      <td>0.51</td>
      <td>9.2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>104</th>
      <td>7.2</td>
      <td>0.490</td>
      <td>0.24</td>
      <td>2.2</td>
      <td>0.070</td>
      <td>5.0</td>
      <td>36.0</td>
      <td>0.9960</td>
      <td>3.33</td>
      <td>0.48</td>
      <td>9.4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>105</th>
      <td>8.1</td>
      <td>0.575</td>
      <td>0.22</td>
      <td>2.1</td>
      <td>0.077</td>
      <td>12.0</td>
      <td>65.0</td>
      <td>0.9967</td>
      <td>3.29</td>
      <td>0.51</td>
      <td>9.2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>106</th>
      <td>7.8</td>
      <td>0.410</td>
      <td>0.68</td>
      <td>1.7</td>
      <td>0.467</td>
      <td>18.0</td>
      <td>69.0</td>
      <td>0.9973</td>
      <td>3.08</td>
      <td>1.31</td>
      <td>9.3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>107</th>
      <td>6.2</td>
      <td>0.630</td>
      <td>0.31</td>
      <td>1.7</td>
      <td>0.088</td>
      <td>15.0</td>
      <td>64.0</td>
      <td>0.9969</td>
      <td>3.46</td>
      <td>0.79</td>
      <td>9.3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>108</th>
      <td>8.0</td>
      <td>0.330</td>
      <td>0.53</td>
      <td>2.5</td>
      <td>0.091</td>
      <td>18.0</td>
      <td>80.0</td>
      <td>0.9976</td>
      <td>3.37</td>
      <td>0.80</td>
      <td>9.6</td>
      <td>6</td>
    </tr>
    <tr>
      <th>109</th>
      <td>8.1</td>
      <td>0.785</td>
      <td>0.52</td>
      <td>2.0</td>
      <td>0.122</td>
      <td>37.0</td>
      <td>153.0</td>
      <td>0.9969</td>
      <td>3.21</td>
      <td>0.69</td>
      <td>9.3</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_red.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>8.319637</td>
      <td>0.527821</td>
      <td>0.270976</td>
      <td>2.538806</td>
      <td>0.087467</td>
      <td>15.874922</td>
      <td>46.467792</td>
      <td>0.996747</td>
      <td>3.311113</td>
      <td>0.658149</td>
      <td>10.422983</td>
      <td>5.636023</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.741096</td>
      <td>0.179060</td>
      <td>0.194801</td>
      <td>1.409928</td>
      <td>0.047065</td>
      <td>10.460157</td>
      <td>32.895324</td>
      <td>0.001887</td>
      <td>0.154386</td>
      <td>0.169507</td>
      <td>1.065668</td>
      <td>0.807569</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.600000</td>
      <td>0.120000</td>
      <td>0.000000</td>
      <td>0.900000</td>
      <td>0.012000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>0.990070</td>
      <td>2.740000</td>
      <td>0.330000</td>
      <td>8.400000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7.100000</td>
      <td>0.390000</td>
      <td>0.090000</td>
      <td>1.900000</td>
      <td>0.070000</td>
      <td>7.000000</td>
      <td>22.000000</td>
      <td>0.995600</td>
      <td>3.210000</td>
      <td>0.550000</td>
      <td>9.500000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>7.900000</td>
      <td>0.520000</td>
      <td>0.260000</td>
      <td>2.200000</td>
      <td>0.079000</td>
      <td>14.000000</td>
      <td>38.000000</td>
      <td>0.996750</td>
      <td>3.310000</td>
      <td>0.620000</td>
      <td>10.200000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>9.200000</td>
      <td>0.640000</td>
      <td>0.420000</td>
      <td>2.600000</td>
      <td>0.090000</td>
      <td>21.000000</td>
      <td>62.000000</td>
      <td>0.997835</td>
      <td>3.400000</td>
      <td>0.730000</td>
      <td>11.100000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>15.900000</td>
      <td>1.580000</td>
      <td>1.000000</td>
      <td>15.500000</td>
      <td>0.611000</td>
      <td>72.000000</td>
      <td>289.000000</td>
      <td>1.003690</td>
      <td>4.010000</td>
      <td>2.000000</td>
      <td>14.900000</td>
      <td>8.000000</td>
    </tr>
  </tbody>
</table>
</div>



pd.describe() method, indicates that each column has the same number of entries, 1,599, which is shown in the row count. By now, each row and column value should make sense.


```python
df_red.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1599 entries, 0 to 1598
    Data columns (total 12 columns):
     #   Column                Non-Null Count  Dtype  
    ---  ------                --------------  -----  
     0   fixed acidity         1599 non-null   float64
     1   volatile acidity      1599 non-null   float64
     2   citric acid           1599 non-null   float64
     3   residual sugar        1599 non-null   float64
     4   chlorides             1599 non-null   float64
     5   free sulfur dioxide   1599 non-null   float64
     6   total sulfur dioxide  1599 non-null   float64
     7   density               1599 non-null   float64
     8   pH                    1599 non-null   float64
     9   sulphates             1599 non-null   float64
     10  alcohol               1599 non-null   float64
     11  quality               1599 non-null   int64  
    dtypes: float64(11), int64(1)
    memory usage: 150.0 KB


Because we don't have any null values, we can skip over dealing with missing values. We can move on to the next step.

# Analysis of Red Wine


```python
import seaborn as sns

sns.set(rc={'figure.figsize': (14, 8)})
sns.countplot(x="quality", data=df_red)
```




    <Axes: xlabel='quality', ylabel='count'>




    
![png](lesson9_materials_files/lesson9_materials_14_1.png)
    


We want to understand what are the correlations that are happening in the dataset. We will use the correlation matrix to understand the relationships between the variables.


```python
sns.pairplot(df_red)
```




    <seaborn.axisgrid.PairGrid at 0x123fa1910>




    
![png](lesson9_materials_files/lesson9_materials_16_1.png)
    



```python
sns.heatmap(df_red.corr(), annot=True, fmt='.2f', linewidths=2)
```




    <Axes: >




    
![png](lesson9_materials_files/lesson9_materials_17_1.png)
    


- Alcohol is positively correlated with the quality of the red wine.
- Alcohol has a weak positive correlation with the pH value.
- Citric acid and density have a strong positive correlation with fixed acidity.
- pH has a negative correlation with density, fixed acidity, citric acid, and sulfates.


```python
sns.distplot(df_red['alcohol'])
```

    /var/folders/dt/tgmydmjd1g354lgxy3yn76g40000gn/T/ipykernel_22949/3186237779.py:1: UserWarning: 
    
    `distplot` is a deprecated function and will be removed in seaborn v0.14.0.
    
    Please adapt your code to use either `displot` (a figure-level function with
    similar flexibility) or `histplot` (an axes-level function for histograms).
    
    For a guide to updating your code to use the new functions, please see
    https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751
    
      sns.distplot(df_red['alcohol'])





    <Axes: xlabel='alcohol', ylabel='Density'>




    
![png](lesson9_materials_files/lesson9_materials_19_2.png)
    



```python
from scipy.stats import skew

skew(df_red['alcohol'])
```




    np.float64(0.8600210646566755)




```python
sns.boxplot(x='quality', y='alcohol', data = df_red)
```




    <Axes: xlabel='quality', ylabel='alcohol'>




    
![png](lesson9_materials_files/lesson9_materials_21_1.png)
    



```python
sns.boxplot(x='quality', y='alcohol', data = df_red, showfliers=False)
```




    <Axes: xlabel='quality', ylabel='alcohol'>




    
![png](lesson9_materials_files/lesson9_materials_22_1.png)
    


the quality of wine increases, so does the alcohol concentration. That would make sense, right? The higher the alcohol concentration is, the higher the quality of the wine.


```python
sns.jointplot(x='alcohol',y='pH',data=df_red, kind='reg')
```




    <seaborn.axisgrid.JointGrid at 0x13775c950>




    
![png](lesson9_materials_files/lesson9_materials_24_1.png)
    



```python
from scipy.stats import pearsonr

def get_correlation(column1, column2, df):
  pearson_corr, p_value = pearsonr(df[column1], df[column2])
  print("Correlation between {} and {} is {}".format(column1, column2, pearson_corr))
  print("P-value of this correlation is {}".format(p_value))
```


```python
get_correlation('alcohol','pH', df_red)
```

    Correlation between alcohol and pH is 0.2056325085054982
    P-value of this correlation is 9.964497741458061e-17



```python
df_white.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4898.000000</td>
      <td>4898.000000</td>
      <td>4898.000000</td>
      <td>4898.000000</td>
      <td>4898.000000</td>
      <td>4898.000000</td>
      <td>4898.000000</td>
      <td>4898.000000</td>
      <td>4898.000000</td>
      <td>4898.000000</td>
      <td>4898.000000</td>
      <td>4898.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>6.854788</td>
      <td>0.278241</td>
      <td>0.334192</td>
      <td>6.391415</td>
      <td>0.045772</td>
      <td>35.308085</td>
      <td>138.360657</td>
      <td>0.994027</td>
      <td>3.188267</td>
      <td>0.489847</td>
      <td>10.514267</td>
      <td>5.877909</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.843868</td>
      <td>0.100795</td>
      <td>0.121020</td>
      <td>5.072058</td>
      <td>0.021848</td>
      <td>17.007137</td>
      <td>42.498065</td>
      <td>0.002991</td>
      <td>0.151001</td>
      <td>0.114126</td>
      <td>1.230621</td>
      <td>0.885639</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3.800000</td>
      <td>0.080000</td>
      <td>0.000000</td>
      <td>0.600000</td>
      <td>0.009000</td>
      <td>2.000000</td>
      <td>9.000000</td>
      <td>0.987110</td>
      <td>2.720000</td>
      <td>0.220000</td>
      <td>8.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>6.300000</td>
      <td>0.210000</td>
      <td>0.270000</td>
      <td>1.700000</td>
      <td>0.036000</td>
      <td>23.000000</td>
      <td>108.000000</td>
      <td>0.991723</td>
      <td>3.090000</td>
      <td>0.410000</td>
      <td>9.500000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>6.800000</td>
      <td>0.260000</td>
      <td>0.320000</td>
      <td>5.200000</td>
      <td>0.043000</td>
      <td>34.000000</td>
      <td>134.000000</td>
      <td>0.993740</td>
      <td>3.180000</td>
      <td>0.470000</td>
      <td>10.400000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.300000</td>
      <td>0.320000</td>
      <td>0.390000</td>
      <td>9.900000</td>
      <td>0.050000</td>
      <td>46.000000</td>
      <td>167.000000</td>
      <td>0.996100</td>
      <td>3.280000</td>
      <td>0.550000</td>
      <td>11.400000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>14.200000</td>
      <td>1.100000</td>
      <td>1.660000</td>
      <td>65.800000</td>
      <td>0.346000</td>
      <td>289.000000</td>
      <td>440.000000</td>
      <td>1.038980</td>
      <td>3.820000</td>
      <td>1.080000</td>
      <td>14.200000</td>
      <td>9.000000</td>
    </tr>
  </tbody>
</table>
</div>



# White wine analysis


```python
print("white mean = ",df_white["quality"].mean())
print("red mean =",df_red["quality"].mean())
```

    white mean =  5.87790935075541
    red mean = 5.6360225140712945



```python
d = {'color': ['red','white'], 'mean_quality': [5.636023,5.877909]}
df_mean = pd.DataFrame(data=d)
df_mean
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>color</th>
      <th>mean_quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>red</td>
      <td>5.636023</td>
    </tr>
    <tr>
      <th>1</th>
      <td>white</td>
      <td>5.877909</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Let us add new attribute called wine_category  to the both dataframe

df_white['wine_category'] = 'white'
df_red['wine_category'] = 'red'

```


```python
print('RED WINE: List of "quality"', sorted(df_red['quality'].unique()))
print('WHITE WINE: List of "quality"', sorted(df_white['quality'].unique()))
```

    RED WINE: List of "quality" [np.int64(3), np.int64(4), np.int64(5), np.int64(6), np.int64(7), np.int64(8)]
    WHITE WINE: List of "quality" [np.int64(3), np.int64(4), np.int64(5), np.int64(6), np.int64(7), np.int64(8), np.int64(9)]


# Convert into categorical dataset


```python
df_red['quality_label'] = df_red['quality'].apply(lambda value: ('low' if value <= 5 else 'medium') if value <= 7 else 'high')
df_red['quality_label'] = pd.Categorical(df_red['quality_label'], categories=['low', 'medium', 'high'])

df_white['quality_label'] = df_white['quality'].apply(lambda value: ('low' if value <= 5 else 'medium') if value <= 7 else 'high')
df_white['quality_label'] = pd.Categorical(df_white['quality_label'], categories=['low', 'medium', 'high'])

```


```python
print(df_white['quality_label'].value_counts())
df_red['quality_label'].value_counts()
```

    quality_label
    medium    3078
    low       1640
    high       180
    Name: count, dtype: int64





    quality_label
    medium    837
    low       744
    high       18
    Name: count, dtype: int64




```python
df_wines = pd.concat([df_red, df_white])

# Re-shuffle records just to randomize data points.
# `drop=True`: this resets the index to the default integer index.
df_wines = df_wines.sample(frac=1.0, random_state=42).reset_index(drop=True)
df_wines.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
      <th>wine_category</th>
      <th>quality_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.0</td>
      <td>0.17</td>
      <td>0.74</td>
      <td>12.8</td>
      <td>0.045</td>
      <td>24.0</td>
      <td>126.0</td>
      <td>0.99420</td>
      <td>3.26</td>
      <td>0.38</td>
      <td>12.2</td>
      <td>8</td>
      <td>white</td>
      <td>high</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.7</td>
      <td>0.64</td>
      <td>0.21</td>
      <td>2.2</td>
      <td>0.077</td>
      <td>32.0</td>
      <td>133.0</td>
      <td>0.99560</td>
      <td>3.27</td>
      <td>0.45</td>
      <td>9.9</td>
      <td>5</td>
      <td>red</td>
      <td>low</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6.8</td>
      <td>0.39</td>
      <td>0.34</td>
      <td>7.4</td>
      <td>0.020</td>
      <td>38.0</td>
      <td>133.0</td>
      <td>0.99212</td>
      <td>3.18</td>
      <td>0.44</td>
      <td>12.0</td>
      <td>7</td>
      <td>white</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6.3</td>
      <td>0.28</td>
      <td>0.47</td>
      <td>11.2</td>
      <td>0.040</td>
      <td>61.0</td>
      <td>183.0</td>
      <td>0.99592</td>
      <td>3.12</td>
      <td>0.51</td>
      <td>9.5</td>
      <td>6</td>
      <td>white</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.4</td>
      <td>0.35</td>
      <td>0.20</td>
      <td>13.9</td>
      <td>0.054</td>
      <td>63.0</td>
      <td>229.0</td>
      <td>0.99888</td>
      <td>3.11</td>
      <td>0.50</td>
      <td>8.9</td>
      <td>6</td>
      <td>white</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7.2</td>
      <td>0.53</td>
      <td>0.14</td>
      <td>2.1</td>
      <td>0.064</td>
      <td>15.0</td>
      <td>29.0</td>
      <td>0.99323</td>
      <td>3.35</td>
      <td>0.61</td>
      <td>12.1</td>
      <td>6</td>
      <td>red</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7.5</td>
      <td>0.27</td>
      <td>0.31</td>
      <td>17.7</td>
      <td>0.051</td>
      <td>33.0</td>
      <td>173.0</td>
      <td>0.99900</td>
      <td>3.09</td>
      <td>0.64</td>
      <td>10.2</td>
      <td>5</td>
      <td>white</td>
      <td>low</td>
    </tr>
    <tr>
      <th>7</th>
      <td>6.8</td>
      <td>0.11</td>
      <td>0.27</td>
      <td>8.6</td>
      <td>0.044</td>
      <td>45.0</td>
      <td>104.0</td>
      <td>0.99454</td>
      <td>3.20</td>
      <td>0.37</td>
      <td>9.9</td>
      <td>6</td>
      <td>white</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9.0</td>
      <td>0.44</td>
      <td>0.49</td>
      <td>2.4</td>
      <td>0.078</td>
      <td>26.0</td>
      <td>121.0</td>
      <td>0.99780</td>
      <td>3.23</td>
      <td>0.58</td>
      <td>9.2</td>
      <td>5</td>
      <td>red</td>
      <td>low</td>
    </tr>
    <tr>
      <th>9</th>
      <td>7.1</td>
      <td>0.23</td>
      <td>0.30</td>
      <td>2.6</td>
      <td>0.034</td>
      <td>62.0</td>
      <td>148.0</td>
      <td>0.99121</td>
      <td>3.03</td>
      <td>0.56</td>
      <td>11.3</td>
      <td>7</td>
      <td>white</td>
      <td>medium</td>
    </tr>
  </tbody>
</table>
</div>




```python
subset_attr = ['alcohol', 'density', 'pH', 'quality']

low = round(df_wines[df_wines['quality_label'] == 'low'][subset_attr].describe(), 2)
medium = round(df_wines[df_wines['quality_label'] == 'medium'][subset_attr].describe(), 2)
high = round(df_wines[df_wines['quality_label'] == 'high'][subset_attr].describe(), 2)

pd.concat([low, medium, high], axis=1, 
          keys=['👎 Low Quality Wine', 
                '👌 Medium Quality Wine', 
                '👍 High Quality Wine'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="4" halign="left">👎 Low Quality Wine</th>
      <th colspan="4" halign="left">👌 Medium Quality Wine</th>
      <th colspan="4" halign="left">👍 High Quality Wine</th>
    </tr>
    <tr>
      <th></th>
      <th>alcohol</th>
      <th>density</th>
      <th>pH</th>
      <th>quality</th>
      <th>alcohol</th>
      <th>density</th>
      <th>pH</th>
      <th>quality</th>
      <th>alcohol</th>
      <th>density</th>
      <th>pH</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2384.00</td>
      <td>2384.00</td>
      <td>2384.00</td>
      <td>2384.00</td>
      <td>3915.00</td>
      <td>3915.00</td>
      <td>3915.00</td>
      <td>3915.00</td>
      <td>198.00</td>
      <td>198.00</td>
      <td>198.00</td>
      <td>198.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>9.87</td>
      <td>1.00</td>
      <td>3.21</td>
      <td>4.88</td>
      <td>10.81</td>
      <td>0.99</td>
      <td>3.22</td>
      <td>6.28</td>
      <td>11.69</td>
      <td>0.99</td>
      <td>3.23</td>
      <td>8.03</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.84</td>
      <td>0.00</td>
      <td>0.16</td>
      <td>0.36</td>
      <td>1.20</td>
      <td>0.00</td>
      <td>0.16</td>
      <td>0.45</td>
      <td>1.27</td>
      <td>0.00</td>
      <td>0.16</td>
      <td>0.16</td>
    </tr>
    <tr>
      <th>min</th>
      <td>8.00</td>
      <td>0.99</td>
      <td>2.74</td>
      <td>3.00</td>
      <td>8.40</td>
      <td>0.99</td>
      <td>2.72</td>
      <td>6.00</td>
      <td>8.50</td>
      <td>0.99</td>
      <td>2.88</td>
      <td>8.00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>9.30</td>
      <td>0.99</td>
      <td>3.11</td>
      <td>5.00</td>
      <td>9.80</td>
      <td>0.99</td>
      <td>3.11</td>
      <td>6.00</td>
      <td>11.00</td>
      <td>0.99</td>
      <td>3.13</td>
      <td>8.00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>9.60</td>
      <td>1.00</td>
      <td>3.20</td>
      <td>5.00</td>
      <td>10.80</td>
      <td>0.99</td>
      <td>3.21</td>
      <td>6.00</td>
      <td>12.00</td>
      <td>0.99</td>
      <td>3.23</td>
      <td>8.00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>10.40</td>
      <td>1.00</td>
      <td>3.31</td>
      <td>5.00</td>
      <td>11.70</td>
      <td>1.00</td>
      <td>3.33</td>
      <td>7.00</td>
      <td>12.60</td>
      <td>0.99</td>
      <td>3.33</td>
      <td>8.00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>14.90</td>
      <td>1.00</td>
      <td>3.90</td>
      <td>5.00</td>
      <td>14.20</td>
      <td>1.04</td>
      <td>4.01</td>
      <td>7.00</td>
      <td>14.00</td>
      <td>1.00</td>
      <td>3.72</td>
      <td>9.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline

```


```python
fig = df_wines.hist(bins=15, linewidth=1.0, xlabelsize=10, ylabelsize=10, xrot=45, yrot=0, figsize=(10,9), grid=False)

plt.tight_layout(rect=(0, 0, 1.5, 1.5)) 
```


    
![png](lesson9_materials_files/lesson9_materials_39_0.png)
    



```python
df_red.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
      <th>wine_category</th>
      <th>quality_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
      <td>red</td>
      <td>low</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.8</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>2.6</td>
      <td>0.098</td>
      <td>25.0</td>
      <td>67.0</td>
      <td>0.9968</td>
      <td>3.20</td>
      <td>0.68</td>
      <td>9.8</td>
      <td>5</td>
      <td>red</td>
      <td>low</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.8</td>
      <td>0.76</td>
      <td>0.04</td>
      <td>2.3</td>
      <td>0.092</td>
      <td>15.0</td>
      <td>54.0</td>
      <td>0.9970</td>
      <td>3.26</td>
      <td>0.65</td>
      <td>9.8</td>
      <td>5</td>
      <td>red</td>
      <td>low</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.2</td>
      <td>0.28</td>
      <td>0.56</td>
      <td>1.9</td>
      <td>0.075</td>
      <td>17.0</td>
      <td>60.0</td>
      <td>0.9980</td>
      <td>3.16</td>
      <td>0.58</td>
      <td>9.8</td>
      <td>6</td>
      <td>red</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
      <td>red</td>
      <td>low</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, (ax) = plt.subplots(1, 1, figsize=(14,8))
columns_of_interest = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
hm = sns.heatmap(df_wines[columns_of_interest].corr(),
                 ax=ax,           
                 cmap="bwr", 
                 annot=True, 
                 fmt='.2f',       
                 linewidths=.05)

fig.subplots_adjust(top=0.94)
fig.suptitle('Combined Wine Attributes and their Correlation Heatmap', 
              fontsize=14, 
              fontweight='bold')
```




    Text(0.5, 0.98, 'Combined Wine Attributes and their Correlation Heatmap')




    
![png](lesson9_materials_files/lesson9_materials_41_1.png)
    


## Discrete categorical attributes


```python
fig = plt.figure(figsize=(16, 8))

sns.countplot(data=df_wines, x="quality", hue="wine_category")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fc7e196da20>




    
![png](lesson9_materials_files/lesson9_materials_43_1.png)
    


## 3D visualization


```python
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

xscale = df_wines['residual sugar']
yscale = df_wines['free sulfur dioxide']
zscale = df_wines['total sulfur dioxide']
ax.scatter(xscale, yscale, zscale, s=50, alpha=0.6, edgecolors='w')

ax.set_xlabel('Residual Sugar')
ax.set_ylabel('free sulfur dioxide')
ax.set_zlabel('Total sulfur dioxide')

plt.show()
```


    
![png](lesson9_materials_files/lesson9_materials_45_0.png)
    



```python
fig = plt.figure(figsize=(16, 12))

plt.scatter(x = df_wines['fixed acidity'], 
            y = df_wines['free sulfur dioxide'], 
            s = df_wines['total sulfur dioxide'] * 2,
            alpha=0.4, 
            edgecolors='w')

plt.xlabel('Fixed Acidity')
plt.ylabel('free sulfur dioxide')
plt.title('Wine free sulfur dioxide Content - Fixed Acidity - total sulfur dioxide', y=1.05)
```




    Text(0.5, 1.05, 'Wine free sulfur dioxide Content - Fixed Acidity - total sulfur dioxide')




    
![png](lesson9_materials_files/lesson9_materials_46_1.png)
    

