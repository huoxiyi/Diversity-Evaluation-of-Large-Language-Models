from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

import os

vecSize = 100

# load data
data = []
# Loop through file names 1.txt to 1000.txt
for i in range(1, 1001):
    file_name = os.path.join('openai/content-gpt4o', f'content-gpt4o-{i}.txt')
    
    # Open and read the content of each file
    try:
        with open(file_name, 'r', encoding = 'gb2312', errors = 'ignore') as file:
            content = file.read()
            data.append(content)
    except FileNotFoundError:
        print(f"File {file_name} not found.")
    except Exception as e:
        print(f"An error occurred while reading {file_name}: {e}")

# Print the number of files successfully loaded
print(f"Successfully loaded {len(data)} files.")

# create TaggedDocuments
tagged_data = [TaggedDocument(words = word_tokenize(doc.lower()), tags=[str(i+1)]) for i, doc in enumerate(data)]

# train Doc2vec
model = Doc2Vec(vector_size = vecSize)
model.build_vocab(tagged_data)
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

# get doc embeddings and save them into 'embeddings/' dir from 1.txt to 1000.txt
for i in range(1000): #start from 0 to 999
    fileName = str(i+1) + ".txt"
    f = open('openai/4o-Gemini/' + fileName, 'w')
    f.write(str(model.dv[str(i+1)]))
    f.close()
