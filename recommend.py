import pickle
import pandas as pd

with open("matrix_indices.pickle", "rb") as input_file:
    similarity_matrix, df, indices = pickle.load(input_file)

# User Input 
name='India'

# Get the index corresponding to country name
index = indices[name]

# Get the cosine similarity scores 
similarity_scores = list(enumerate(similarity_matrix[index]))

# Sort the similarity scores in descending order
sorted_similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

# Top-10 most similar country scores
top_10_country_scores = sorted_similarity_scores[1:11]

# Get country indices
top_10_country_indices=[]
for i in top_10_country_scores:
    top_10_country_indices.append(i[0])
    
# Top 10 recommended country

print(list(df['name'].iloc[top_10_country_indices]))
