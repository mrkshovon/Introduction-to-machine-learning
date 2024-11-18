import numpy as np
### 2
# Create a one-dimensional array a and initialize it as [4, 5, 6]
a = np.array([4, 5, 6])

# (1) Print the type of a
print("Type of a:", type(a))

# (2) Print the shape of a
print("Shape of a:", a.shape)

# (3) Print the first element in a (the value should be 4)
print("First element of a:", a[0])

#### 3
# Create a two-dimensional array b and initialize it as [ [4,5,6], [1,2,3] ]
b = np.array([[4, 5, 6], [1, 2, 3]])

# (1) Print the shape of b
print("Shape of b:", b.shape)

# (2) Print b(0,0), b(0,1), b(1,1) (the values should be 4, 5, 2)
print("b(0,0):", b[0, 0])
print("b(0,1):", b[0, 1])
print("b(1,1):", b[1, 1])
# #### 4. 

# (1) Create a matrix a, which is all 0, of size 3x3
a = np.zeros((3, 3))
print("Matrix a (all 0's, 3x3):\n", a)

# (2) Create a matrix b, which is all 1, of size 4x5
b = np.ones((4, 5))
print("\nMatrix b (all 1's, 4x5):\n", b)

# (3) Create a unit matrix c, of size 4x4 (identity matrix)
c = np.eye(4)
print("\nUnit matrix c (identity matrix, 4x4):\n", c)

# (4) Create a random matrix d, of size 3x2
d = np.random.rand(3, 2)
print("\nRandom matrix d (3x2):\n", d)
# #### 5


# Create array a and initialize it as [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# (1) Print array a
print("Array a:\n", a)

# (2) Put the 0th and 1st rows, 2nd and 3rd columns of array a into array b, then print b
b = a[0:2, 2:4]  # Select 0th and 1st rows, 2nd and 3rd columns
print("\nArray b (0th and 1st rows, 2nd and 3rd columns of a):\n", b)

# (3) Print b(0,0)
print("\nb(0,0):", b[0, 0])

# #### 6.

# Array a from question 5
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# (1) Put all the elements of the last two rows of array a into a new array c
c = a[-2:, :]  # Select the last two rows of array a
print("Array c (last two rows of a):\n", c)

# (2) Print the last element of the first row in c (using -1 for the last element)
print("\nLast element of the first row in c:", c[0, -1])
###7

# (1) Create an array x
x = np.array([[1, 2], [3, 4]], dtype=np.float64)

# (2) Create an array y
y = np.array([[5, 6], [7, 8]], dtype=np.float64)

# (3) Print x + y and np.add(x, y)
print("x + y:\n", x + y)
print("np.add(x, y):\n", np.add(x, y))

# (4) Print x - y and np.subtract(x, y)
print("x - y:\n", x - y)
print("np.subtract(x, y):\n", np.subtract(x, y))

# (5) Print x * y, np.multiply(x, y) and np.dot(x, y), and compare the results
print("x * y:\n", x * y)
print("np.multiply(x, y):\n", np.multiply(x, y))
print("np.dot(x, y):\n", np.dot(x, y))

# (6) Print x / y and np.divide(x, y)
print("x / y:\n", x / y)
print("np.divide(x, y):\n", np.divide(x, y))

# (7) Print the square root of x
print("sqrt(x):\n", np.sqrt(x))

# (8) Print x.dot(y) and np.dot(x, y)
print("x.dot(y):\n", x.dot(y))
print("np.dot(x, y):\n", np.dot(x, y))

####8
# (1) Print the sum of x
print("Sum of x:\n", np.sum(x))

# (2) Print the sum of the rows of x
print("Sum of the rows of x:\n", np.sum(x, axis=0))

# (3) Print the sum of the columns of x
print("Sum of the columns of x:\n", np.sum(x, axis=1))


###9
# (1) Print the mean of x
print("Mean of x:\n", np.mean(x))

# (2) Print the mean of the rows of x
print("Mean of the rows of x:\n", np.mean(x, axis=0))

# (3) Print the mean of the columns of x
print("Mean of the columns of x:\n", np.mean(x, axis=1))

###10
# Print the matrix transpose of x
print("Transpose of x:\n", x.T)

###11
# (1) Print the index of the max element of x
print("Index of the max element of x:\n", np.argmax(x))

# (2) Print the index of the max element in the rows of x
print("Index of the max element in the rows of x:\n", np.argmax(x, axis=0))

# (3) Print the index of the max element in the columns of x
print("Index of the max element in the columns of x:\n", np.argmax(x, axis=1))

###12
import matplotlib.pyplot as plt

x = np.arange(0, 100, 0.1)
y = x * x

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y = x * x')
plt.title('Plot of y = x * x')
plt.grid(True)
plt.show()

###13
x = np.arange(0, 3 * np.pi, 0.1)

# (1) Plot sin(x)
y_sin = np.sin(x)
plt.plot(x, y_sin, label='sin(x)')
plt.xlabel('x')
plt.ylabel('y = sin(x)')
plt.title('Plot of y = sin(x)')
plt.legend()
plt.grid(True)
plt.show()

# (2) Plot cos(x)
y_cos = np.cos(x)
plt.plot(x, y_cos, label='cos(x)')
plt.xlabel('x')
plt.ylabel('y = cos(x)')
plt.title('Plot of y = cos(x)')
plt.legend()
plt.grid(True)
plt.show()
