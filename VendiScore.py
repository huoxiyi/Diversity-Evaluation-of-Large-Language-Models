import os
import math
from vendi_score import vendi
import numpy as np

bandwidth = 3

# Directory containing the text files
directory = 'E:\programming\.vscode\programming\python\openai\\bertembedding-gpt4o'

# Initialize a list to hold the vectors
vectors = []

# Loop through each file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        filepath = directory+'\\'+filename
        #print(filepath)
        # Read the file contents
        with open(filepath, 'r') as file:
            content = file.read().strip()
            
            # Convert the string representation of the vector to a numpy array
            vector = np.fromstring(content.strip('[[]]tensor()'), sep=', ')
            #print(vector)
            vectors.append(vector)


# Now vectors is a list of numpy arrays, index from 0 to 999
# Compute K(i,j): exp(-||vectors[i] - vectors[j]||^2 (dot product of itself) / bandwidth)
K_list = []
# i stands for row, j stands for column
for i in range(1000):
    K_i = []
    for j in range(1000):
        diff = vectors[i] - vectors[j]
        k_ij = math.exp(-(np.dot(diff,diff))/bandwidth)
        K_i.append(k_ij)
    K_list.append(K_i)

K = np.array(K_list)

print(K.shape)

print(vendi.score_K(K))