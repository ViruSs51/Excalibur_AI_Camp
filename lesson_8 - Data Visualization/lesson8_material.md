# **Lesson 7: Data Visualization**

### **Main Libraries for Data Visualization in Python**
1. **Matplotlib**: The foundational library that offers flexibility and customization.
2. **Seaborn**: A statistical data visualization library built on top of Matplotlib, providing a high-level interface for drawing attractive graphics.
3. **Pandas Plotting**: Built-in plotting functions that work directly with DataFrames.


```python
# Installation of matplotlib: !pip install matplotlib
# Installation of seaborn: !pip install seaborn
```


```python
# All imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

**Basic Plot:**

To create a basic plot, you'll commonly use the functions `plot()`, `xlabel()`, `ylabel()`, `title()`, and `show()`.


```python
# Sample Data
x = [1, 2, 3, 4, 5]
y = [10, 11, 12, 13, 14]

# Plotting
plt.plot(x, y)
plt.xlabel('X-axis label')
plt.ylabel('Y-axis label')
plt.title('Basic Line Plot')
plt.show()
```


    
![png](lesson8_material_files/lesson8_material_5_0.png)
    


### Key Plotting Parameters

- **Line Styles**: `'-'`, `'--'`, `'-.'`, `':'`
- **Markers**: `'o'`, `'s'`, `'^'`, `'D'`, etc.
- **Colors**: `'b'`, `'g'`, `'r'`, `'c'`, `'m'`, `'y'`, `'k'`

**Example with Parameters:**


```python
plt.plot(x, y, linestyle=':', color='k', marker='D', markersize=8)
plt.show()
```


    
![png](lesson8_material_files/lesson8_material_7_0.png)
    


### Types of Plots

1. **Line Plot**: For continuous data. (`plot()`)

2. **Bar Plot**: For categorical data comparison. (`bar()`)




```python
categories = ['A', 'B', 'C']
values = [10, 20, 15]

plt.bar(categories, values, color=['red', 'green', 'blue'])
plt.show()
```


    
![png](lesson8_material_files/lesson8_material_9_0.png)
    


3. **Histogram**: For frequency distributions. (`hist()`)


```python
data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
plt.hist(data, bins=4, color='purple', alpha=0.7)
plt.show()
```


    
![png](lesson8_material_files/lesson8_material_11_0.png)
    


4. **Scatter Plot**: For relationships between two variables. (`scatter()`)


```python
plt.scatter(x, y, color='orange')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot Example')
plt.show()
```


    
![png](lesson8_material_files/lesson8_material_13_0.png)
    


5. **Pie Chart**: For proportional data. (`pie()`)


```python
sizes = [15, 30, 45, 10]
labels = ['A', 'B', 'C', 'D']

plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title('Pie Chart Example')
plt.show()
```


    
![png](lesson8_material_files/lesson8_material_15_0.png)
    


### Seaborn Enhancements

Seaborn can be used to create more aesthetically pleasing plots.


```python
# Sample Data
import numpy as np
data = np.random.rand(100)

# Kernel Density Estimate Plot
sns.kdeplot(data, fill=True)
plt.title('Seaborn KDE Plot')
plt.show()
```


    
![png](lesson8_material_files/lesson8_material_17_0.png)
    


### Advanced Features

- **Subplots**: Create multiple plots in one figure using `plt.subplot()` or `plt.subplots()`.


```python
fig, axs = plt.subplots(2, 1, figsize=(5, 10))
axs[0].plot(x, y, '--o')
axs[1].bar(categories, values, color='skyblue')
plt.show()
```


    
![png](lesson8_material_files/lesson8_material_19_0.png)
    


- **Annotations**: Add text or arrows for emphasis.



```python
plt.plot(x, y, 'g-o')
plt.text(2, 12, 'Peak Point', fontsize=12)
plt.annotate('Important', xy=(3, 13), xytext=(3, 11),
            arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate('Peak Point', xy=(3, 13), xytext=(4, 15),
            arrowprops=dict(facecolor='black', arrowstyle='->'),
            fontsize=12)
plt.show()
```


    
![png](lesson8_material_files/lesson8_material_21_0.png)
    


- **Themes**: Use `sns.set_theme()` to apply themes such as `darkgrid`, `whitegrid`, `dark`, `white`, or `ticks`.


```python
sns.set_theme(style="darkgrid")
sns.scatterplot(x=x, y=y)
plt.show()
```


    
![png](lesson8_material_files/lesson8_material_23_0.png)
    


### **Disabling Seaborn Modifications**

If Seaborn has modified the context or style of your plots and you want to revert to the default Matplotlib style settings, you can use:


```python
sns.set()  # To activate Seaborn settings
# Generate some plot here

# To revert to Matplotlib default settings:
sns.reset_orig()

# Now, any following plots will use Matplotlib's default styling
plt.plot([1, 2, 3], [4, 5, 6])
plt.show()
```


    
![png](lesson8_material_files/lesson8_material_25_0.png)
    


### **Key Elements of a Plot**
1. **Title**: Describes what the plot is about.


```python
plt.title('You can add a title here')  # Add at top of plot
```




    Text(0.5, 1.0, 'You can add a title here')




    
![png](lesson8_material_files/lesson8_material_27_1.png)
    


2. **Axes Labels**: Indicate what each axis represents.


```python
plt.xlabel('Scrii pe axa X')
plt.ylabel('Scrii pe axa Y')
```




    Text(0, 0.5, 'Scrii pe axa Y')




    
![png](lesson8_material_files/lesson8_material_29_1.png)
    


3. **Legend**: Helps distinguish different data series or categories within a plot.


```python
plt.plot(x, y, label='Data 1')
plt.legend(loc='best')  # Loc can be 'upper left', 'upper right', etc.
```




    <matplotlib.legend.Legend at 0x11989e030>




    
![png](lesson8_material_files/lesson8_material_31_1.png)
    


4. **Grid**: Assists in connecting data points with their respective axis values.


```python
plt.plot(x, y, label='Data 1')
plt.grid(True)  # Can customize with linestyle, linewidth, etc.
```


    
![png](lesson8_material_files/lesson8_material_33_0.png)
    


5. **Scales**: Adjust the plot scale to fit data distribution better. Axes can be set to logarithmic scale.

Changing the scale of an axis can be very useful for visualizing data with wide ranges or for highlighting certain growth patterns, such as exponential growth, in a more intuitive way.


```python
plt.plot(x, y, label='Data 1')
plt.xscale('log')  # Logarithmic scale for x-axis
plt.yscale('log')  # Logarithmic scale for y-axis
```


    
![png](lesson8_material_files/lesson8_material_35_0.png)
    



```python
# Sample Data: Exponential growth
x = np.linspace(1, 10, 100)
y = np.exp(x)  # Exponential curve

plt.figure()
plt.plot(x, y, 'r-', label='Exponential Growth')

# Logarithmic Scale
plt.xscale('log')  # or plt.yscale('log') for y-axis
plt.xlabel('Log Scale')
plt.ylabel('Value')
plt.title('Logarithmic Scale Example')
plt.legend()
plt.show()
```


    
![png](lesson8_material_files/lesson8_material_36_0.png)
    


6. **Ranges and Limits**: Define the area displayed on each axis.


```python
plt.plot(x, y, label='Data 1')
plt.xlim(0, 10)
plt.ylim(0, 20)

```




    (0.0, 20.0)




    
![png](lesson8_material_files/lesson8_material_38_1.png)
    


7. **Ticks**: Specify the marking points on axes for clarity.


```python
plt.plot(x, y, label='Data 1')
plt.xticks([0, 1, 2, 3, 4, 5])
plt.yticks([10, 15, 20])
```




    ([<matplotlib.axis.YTick at 0x118673e60>,
      <matplotlib.axis.YTick at 0x118c23560>,
      <matplotlib.axis.YTick at 0x118fc4320>],
     [Text(0, 10, '10'), Text(0, 15, '15'), Text(0, 20, '20')])




    
![png](lesson8_material_files/lesson8_material_40_1.png)
    


### **More on Subplots**

Creating multiple plots within a single figure allows for comparative analysis or displaying related data. This can be effectively accomplished using `subplot()` or `subplots()`.

**Using `subplots()`:**

This method gives you a grid of plots.


```python
# Sample Data
x = np.arange(0, 10, 1)
y1 = np.sin(x)
y2 = np.cos(x)

# Create Subplots
fig, axs = plt.subplots(2, 1, figsize=(8, 6))  # 2 rows, 1 column

# First Plot
axs[0].plot(x, y1, 'b-o', label='Sine')
axs[0].set_title('Sine Function')
axs[0].set_xlabel('x')
axs[0].set_ylabel('sin(x)')
axs[0].legend()

# Second Plot
axs[1].plot(x, y2, 'r-^', label='Cosine')
axs[1].set_title('Cosine Function')
axs[1].set_xlabel('x')
axs[1].set_ylabel('cos(x)')
axs[1].legend()

plt.tight_layout()  # Adjusts subplot parameters to give room for labels
plt.show()
```


    
![png](lesson8_material_files/lesson8_material_42_0.png)
    



```python
x = np.linspace(0, 10, 100)
y1 = x
y2 = x ** 2

fig, axs = plt.subplots(1, 3, figsize=(12, 4))  # 1 row, 3 columns

# Plot in each subplot
axs[0].plot(x, y1)
axs[0].set_title('First Plot')

axs[1].plot(x, y2)
axs[1].set_title('Second Plot')

axs[2].plot(x, np.sqrt(x))
axs[2].set_title('Third Plot')

plt.tight_layout()
plt.show()
```


    
![png](lesson8_material_files/lesson8_material_43_0.png)
    



```python
fig = plt.figure(figsize=(10, 8))

ax1 = plt.subplot2grid((2, 2), (0, 0))
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)  # Spans two columns

# Add data
ax1.plot(x, y1)
ax1.set_title('Plot 1')

ax2.plot(x, y2)
ax2.set_title('Plot 2')

ax3.plot(x, np.log(x))
ax3.set_title('Plot 3 - Span Two Columns')

plt.tight_layout()
plt.show()
```

    /var/folders/dt/tgmydmjd1g354lgxy3yn76g40000gn/T/ipykernel_20709/2410736388.py:14: RuntimeWarning: divide by zero encountered in log
      ax3.plot(x, np.log(x))



    
![png](lesson8_material_files/lesson8_material_44_1.png)
    


### **Advanced Plot Customization**

1. **Colormaps**: Applying different colormaps to visualize the intensity or categories.


```python
x = np.random.rand(100)
y = np.random.rand(100)
plt.scatter(x, y, c=x+y, cmap='viridis')
plt.colorbar()  # To show a color scale
```




    <matplotlib.colorbar.Colorbar at 0x118cd8a10>




    
![png](lesson8_material_files/lesson8_material_46_1.png)
    


### **Multilabels (Multiple Y-Axes)**

Sometimes, you may want to plot two different datasets with different scales on the same plot. This can be done using multiple y-axes.


```python
# Sample Data
x = [0, 1, 2, 3, 4]
y1 = [100, 200, 300, 400, 500]  # Revenue
y2 = [20, 30, 25, 35, 50]  # Number of Customers

fig, ax1 = plt.subplots()

# First dataset
ax1.plot(x, y1, 'g-')
ax1.set_xlabel('Time')
ax1.set_ylabel('Revenue ($)', color='g')

# Second dataset
ax2 = ax1.twinx()  # Create a second y-axis that shares the same x-axis
ax2.plot(x, y2, 'b-')
ax2.set_ylabel('Number of Customers', color='b')

plt.title('Multilabel Plot (Different Y-axes)')
plt.show()
```


    
![png](lesson8_material_files/lesson8_material_48_0.png)
    


### **Multidimensional Plots**

For plotting multidimensional data, the `scatter` plot can be extended using different techniques like color mapping or 3D plots with Matplotlib's `mplot3d` toolkit.

**3D Scatter Plot:**


```python
# Sample Data
x = np.random.rand(50)
y = np.random.rand(50)
z = np.random.rand(50)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, z, c='r', marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.title('3D Scatter Plot')
plt.show()
```


    
![png](lesson8_material_files/lesson8_material_50_0.png)
    


### **Saving**:
Save plots to files using `savefig()`.


```python
plt.savefig('plot.png', dpi=300, bbox_inches='tight')  # 'pdf' or 'svg' formats are also available
```
