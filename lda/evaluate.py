import pickle
import torch
from tqdm import tqdm
import os
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import TSNE

num_topics = 90
top_words = []
with open('./reuters_output/train_topics_keys_%d.txt'%(num_topics), 'r') as topic_top_words:
    for line in topic_top_words.readlines():
        top_words.append(line.split('\t')[2])

    n_unique = len(set(' '.join(top_words).split(' ')))
with open('./mallet_%d_top_words.txt'%(num_topics), 'w') as topic_top_words:
    topic_top_words.write(''.join(top_words))

def topic_coherence(num_topics):
    print("Calculating topic coherence...")
    cmd = "java -jar /research2/tools/palmetto/palmetto-0.1.0-jar-with-dependencies.jar /research2/tools/palmetto/wikipedia_bd C_V mallet_%d_top_words.txt"%(num_topics)
    os.system(cmd)
    print("*"*50)

def topic_diversity(num_topics, n_unique):
    print("Calculating topic diveristy...")
    TD = n_unique / (10 * num_topics)
    print('Topic diveristy is: %f'%(TD))
    print("*"*50)

def cosine_similarity():
    print("Calculating topic similarity...")
    data = []

    with open('./conala_output/train_tww_%d.txt'%(num_topics), 'r') as topic_word_weights:
        for line in topic_word_weights.readlines():
            data.append([int(line.split('\t')[0]),str(line.split('\t')[1]),float(line.split('\t')[2])])

    df_data = pd.DataFrame(data, columns=['topic','word','uweight'])
    df_data = df_data.sort_values(by=['topic', 'word'])
    concepts_emb = []
    for i in range(num_topics):
        sub_data = df_data[df_data['topic']==i]
        concept_row = sub_data['uweight'].values
        concept_row = concept_row/max(concept_row)
        concepts_emb.append(concept_row)

    concepts_emb = np.array(concepts_emb)
    concepts_emb = torch.from_numpy(concepts_emb)
    #concepts_emb = torch.softmax(concepts_emb, dim =-1)

    d = torch.matmul(concepts_emb, concepts_emb.t())
    norm = (concepts_emb * concepts_emb).sum(1, keepdims=True) ** .5
    cs = d / norm / norm.t()
    total_cs = torch.sum(torch.triu(cs, diagonal=1))
    print("Total Cosine Similairty: %f"%(total_cs))
    print("*"*50)

def distance_similarity():
    print("Calculating distance similairty between topic embeddings...")
    data = []

    with open('./conala_output/train_tww_%d.txt'%(num_topics), 'r') as topic_word_weights:
        for line in topic_word_weights.readlines():
            data.append([int(line.split('\t')[0]),str(line.split('\t')[1]),float(line.split('\t')[2])])

    df_data = pd.DataFrame(data, columns=['topic','word','uweight'])
    df_data = df_data.sort_values(by=['topic', 'word'])
    concepts_emb = []
    for i in range(num_topics):
        sub_data = df_data[df_data['topic']==i]
        concept_row = sub_data['uweight'].values
        concept_row = concept_row/max(concept_row)
        concepts_emb.append(concept_row)

    concepts_emb = np.array(concepts_emb)
    tsne = TSNE(n_components=300,method='exact', random_state=2019)
    concepts_emb = tsne.fit_transform(concepts_emb)
    print(concepts_emb.shape)

    d = euclidean_distances(concepts_emb, concepts_emb, squared=True)
    d = torch.from_numpy(d)
    d = d / torch.max(d)

    total_cs = torch.sum(torch.triu(d, diagonal=1))
    print("Total Distance Similairty: %f"%(total_cs/((d.shape[0]*d.shape[0]/2)-d.shape[0])))
    print("*"*50)
    return total_cs
#Evaulate
topic_coherence(num_topics)
topic_diversity(num_topics, n_unique)
#cosine_similarity()
#distance_similarity()
