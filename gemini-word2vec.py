# Python program to generate word vectors using Word2Vec
 
# importing all necessary modules
import numpy
from gensim.models import Word2Vec
import gensim
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
import nltk
#nltk.download('punkt')

f_gen = "3.5tur.txt"
f_emb_CBOW = "gpt3.5tur-word2vec-CBOW.txt"
f_emb_SG = "gpt3.5tur-word2vec-SG.txt"

warnings.filterwarnings(action='ignore')

 
#  Reads ‘f_gen’ file
sample = open(f_gen)
s = sample.read()
 
# Replaces escape character with space
f = s.replace("\n", " ")
 
data = []
 
# iterate through each sentence in the file
for i in sent_tokenize(f):
    temp = []
 
    # tokenize the sentence into words
    for j in word_tokenize(i):
        temp.append(j.lower())
    
    #print(temp)
    data.append(temp)
 

#print(data)

f1 = open(f_emb_CBOW,'a')
# Create CBOW model
model1 = gensim.models.Word2Vec(data, min_count=1, vector_size=20, window=5)

# write to file
for sentence in data:
    #print("sentence:",end = '')
    #print(sentence)
    for word in sentence:
        #print('word:',end = '')
        #print(word)
        numpy.savetxt(f1, model1.wv[word].reshape(1,-1),fmt = '%1.4f',delimiter=',')

f1.close()

f2 = open(f_emb_SG,'a')
# Create Skip Gram model
model2 = gensim.models.Word2Vec(data, min_count=1, vector_size=20, window=5, sg=1)

# write to file
for sentence in data:
    for word in sentence:
        numpy.savetxt(f2, model2.wv[word].reshape(1,-1),fmt = '%1.4f',delimiter=',')

f2.close()