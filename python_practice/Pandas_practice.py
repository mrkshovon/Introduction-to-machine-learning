import pandas as pd
###2
# (1) Create a pandas Series s
s = pd.Series([4, 5, 6])

# (2) Print s
print("Series s:")
print(s)

# (3) Print the type of s
print("\nType of s:")
print(type(s))

# (4) Print the shape of s
print("\nShape of s:")
print(s.shape)
###3
import pandas as pd

# Create a DataFrame based on the given dictionary
dic = {
    'name': ['Andy', 'James', 'Lucy'],
    'age': [18, 20, 22],
    'gender': ['male', 'male', 'female']
}
d = pd.DataFrame(dic)

# (1) Print the shape of d
print("Shape of d:")
print(d.shape)

# (2) Print the type of values in d
print("\nType of values in d:")
print(d.dtypes)

# (3) Print the index of d
print("\nIndex of d:")
print(d.index)

# (4) Print the columns of d
print("\nColumns of d:")
print(d.columns)

# (5) Print the summary of d using describe()
print("\nSummary of d:")
print(d.describe(include='all'))

###4

# Create the DataFrame based on the given dictionary
dic = {
    'name': ['Andy', 'James', 'Lucy'],
    'age': [18, 20, 22],
    'gender': ['male', 'male', 'female']
}
d = pd.DataFrame(dic)

# (1) Select and print one column ('name') from DataFrame d
print("Column 'name':")
print(d['name'])

# (2) Select and print the first two rows from DataFrame d
print("\nFirst two rows:")
print(d.head(2))

# (3) Select and print the first two columns from DataFrame d
print("\nFirst two columns:")
print(d.iloc[:, :2])

#####
import pandas as pd

# Read the CSV file
df = pd.read_csv('salary.csv')

# Print the DataFrame to verify the contents
print(df)

###6
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('salary.csv')

# Plot the 'Salary' column
plt.figure(figsize=(10, 6))  # Optional: Set the figure size
plt.plot(df['Salary'], marker='o', linestyle='-', color='b')  # Plotting the salary data
plt.xlabel('Index')  # Label for the x-axis
plt.ylabel('Salary')  # Label for the y-axis
plt.title('Salary Data')  # Title of the plot
plt.grid(True)  # Show grid
plt.show()  # Display the plot
