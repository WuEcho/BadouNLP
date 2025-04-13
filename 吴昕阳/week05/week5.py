import jieba
from gensim.models import Word2Vec
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
import math

def load_word_voc(model_path):
    model = Word2Vec.load(model_path)
    return model

def load_corpus(corpus_path):
    sentences = []
    with open(corpus_path,'r',encoding='utf-8') as f:
       for line in f:
           sentence = line.strip()
           sentences.append(" ".join(jieba.lcut(sentence)))
    return sentences

def sentence_to_vector(model_path,corpus_path):
    model = load_word_voc(model_path) #词向量
    sentences = load_corpus(corpus_path) #切完词的语料
    vocabs = []
    for sentence in sentences:
        words = sentence.split(" ")
        vocab = np.zeros(model.vector_size)
        for word in words:
            try:
                vocab += model.wv[word]
            except KeyError:
                vocab += np.zeros(model.vector_size)    
        vocabs.append(vocab/len(words))
    return np.array(vocabs)

def main():
    sentense = load_corpus("corpus.txt")
    vectors = sentence_to_vector("word2vec.model","corpus.txt")

    n_clusters = int(math.sqrt(len(sentense)))
    print("指定列别的个数：",n_clusters)

    kmeans = KMeans(n_clusters)
    kmeans.fit(vectors) 

    sentence_label_dic = defaultdict(list)
    for sentence,label in zip(sentense,kmeans.labels_):
        sentence_label_dic[label].append(sentence)
    
    for label,sentences in sentence_label_dic.items():
        print(f"第{label}类的句子：")
        for i in range(min(3,len(sentences))):
            print(sentences[i].replace(" ", ""))
        print("......")

if __name__ == "__main__":
    main()