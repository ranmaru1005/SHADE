import numpy as np

# Example numpy array
array = np.array([[1, 0.0000123456789, 3], [4, 5, 6], [7, 8, 9]])
array2 = np.array([["2",3]])

# Save the numpy array to a text file
with open ('MRR/data/array_output.txt',"a") as f:
    np.savetxt(f, array ,fmt = "%s")  # Using integer format