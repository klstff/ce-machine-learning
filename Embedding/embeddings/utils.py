from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import pickle
from sklearn.neighbors import KDTree
import numpy as np
from typing import List,Union


def get_embedding(str_dataset:str,embedding_size:int=100,overwrite=False):
    dict_embedding = {}
    erro_value = 0
    #a principio, é feito uma tentativa de obter o embedding 
    #já transformado em dict_embedding, caso ele não exista,
    #o mesmo é criado
    file_not_found = False
    if not overwrite:
        try:
            with open("assignment_cache/"+str_dataset+".p",'rb') as embedding_file:
                dict_embedding = pickle.load(embedding_file)
        
        except IOError: 
            file_not_found = True

    if file_not_found or overwrite:
        with  open("embeddings_data/"+str_dataset,"r", encoding='utf-8') as f_e:
            try:
                for i,line in enumerate(f_e):
                    arr_line = line.strip().split()
                    #nesses arquivos, por algum motivo, algumas palavras possuiam
                    #dimensões erradas. Essas palavras foram ignoradas.
                    if len(arr_line)<embedding_size+1:
                        erro_value += 1
                        continue

                    #obtem a palavra
                    word = " ".join(arr_line[:-embedding_size])

                    #obtem o embedding
                    dict_embedding[word] = np.array(arr_line[-embedding_size::], dtype=np.float16)
                    if(i%10000 == 0):
                        print(f"{i}: {word}")
            except ValueError:
                print(f"LInha com erro: '{line}'")
                erro_value += 1
            
        with open(str_dataset+".p",'wb') as embedding_file:
            pickle.dump(dict_embedding,embedding_file)
    
    print(f"Palavras ignoradas: {erro_value}")
    return dict_embedding


def plot_words_embeddings_vectors(word_vectors, color, vectors_names=[], create_figure=True):
    #usa o Principal Component ANalysis (PCA) para redução 
    #dos embeddings em word_vectors em um espaço bidimensional
    pca =  PCA(random_state=0)
    word_vectors_2d = pca.fit_transform(word_vectors)[:,:2]

    #plota a representação criada word_vectors_2d 
    plt.figure(figsize=(13,7))
    plt.scatter(word_vectors_2d[:,0],word_vectors_2d[:,1],linewidths=2,color=color)
    plt.xlabel("PC1",size=15)
    plt.ylabel("PC2",size=15)
    plt.title("Word Embedding Space",size=20)
    #plota os labels se necessario
    if vectors_names is not None and len(vectors_names) > 0:
        for i, name in enumerate(vectors_names):
            plt.annotate(name,xy=(word_vectors_2d[i,0],word_vectors_2d[i,1]))


def plot_words_embeddings(dict_embedding, words_to_use):
    #cria a variavel word_vectors que será uma matriz
    #com todas as palavras a serem usadas.
    #em que, cada linha é uma palavra que é representada pelo seu
    # embedding 
    some_embedding = list(dict_embedding.values())[0]
    embeding_dim = len(some_embedding)
    word_vectors = np.zeros((len(words_to_use), embeding_dim)) 
    for i,word in enumerate(words_to_use):
        if word in dict_embedding:
            word_vectors[i,:] = dict_embedding[word]
        else:
            print(f"Não foi possivel encontrar a palavra {word}")

    plot_words_embeddings_vectors(word_vectors, "blue",words_to_use, create_figure=True)
    
    
    
class Analogy:
    def __init__(self, dict_embedding,kdtree_file, overwrite_kdtree=False):
        
        self.dict_embedding = dict_embedding
        self.kdtree_embedding = KDTreeEmbedding(dict_embedding, kdtree_file, overwrite_kdtree) 


    def calcula_embedding_analogia(self, palavra_x:str, esta_para:str, assim_como:str) -> np.array:
        #checa se as palavras existem 
        for palavra in [palavra_x,esta_para,assim_como]:
            if palavra not in self.dict_embedding:
                print(f"Não foi possivel encontrar a palavra: {palavra}")
                return None    

        #obtem o embedding de cada palavra usando self.dict_embedding       
        embedding_x = self.dict_embedding[palavra_x]
        embedding_x_esta_para = self.dict_embedding[esta_para]
        embedding_y = self.dict_embedding[assim_como]
        #print(f"x: {embedding_x} esta para: {embedding_x_esta_para} assim_como: {embedding_y}" )

        #retorna o calculo da analogia
        embedding_y_esta_para = embedding_y - embedding_x + embedding_x_esta_para

        return embedding_y_esta_para         

    def analogia(self, palavra_x:str, esta_para:str, assim_como:str) -> List:
        
        #calcula o embeding da analogia
        embedding = self.calcula_embedding_analogia(palavra_x, esta_para, assim_como)

        #caso não exista uma das palavras, é retornado uma lista vazia
        if embedding is None:
            return []

        #obtem as palavras mais similares     
        _,words = self.kdtree_embedding.get_most_similar_embedding(embedding)
        
        for word in words:
            if word == palavra_x or word == esta_para or word == assim_como: words.remove(word)

        return words[:4]



class KDTreeEmbedding:
    def __init__(self, dict_embedding, kdtree_file, overwrite_kdtree=False):
        #obtem a dimensão do embedding para utilizar na inicialização
        #da matriz
        some_embedding = list(dict_embedding.values())[0]
        embeding_dim = len(some_embedding)

        self.mat_embedding = np.zeros((len(dict_embedding), embeding_dim)) 
        self.dict_embedding = dict_embedding
        
        #caso já tenha sido salvo o arquivo com o KTree já criado,
        #é apenas lido a estrutura do KDTreeEmbedding
        file_not_found = False
        if not overwrite_kdtree:
            try:
                with open(f"assignment_cache/{kdtree_file}", 'rb') as f:
                    dict_data = pickle.load(f)
                    self.kd_embedding = dict_data["kd"]
                    self.pos_to_word = dict_data["pos_to_word"]
                    self.word_to_pos = dict_data["word_to_pos"]
                    
                    for pos,word in self.pos_to_word.items():
                        self.mat_embedding[pos] = dict_embedding[word]

            #caso não tenha sido salvo ainda, é criado um arquivo para armazenar a estrutura
            #e é criado o KDTree e os parametros pos_to_word e word_to_pos
            except IOError:
                file_not_found = True

        if overwrite_kdtree or file_not_found: 
            try:           
                with open(f"assignment_cache/{kdtree_file}.p",'wb') as kd_file:
                    i = 0
                    self.pos_to_word = {}
                    self.word_to_pos = {}
                    
                    for word,embedding in dict_embedding.items():
                        self.mat_embedding[i,:] = embedding
                        self.pos_to_word[i] = word
                        self.word_to_pos[word] = i
                        i += 1
                    self.kd_embedding = KDTree(self.mat_embedding, leaf_size=30, metric='euclidean')
                    pickle.dump({"kd":self.kd_embedding,
                                "pos_to_word":self.pos_to_word,
                                "word_to_pos":self.word_to_pos},kd_file)
            except IOError: 
                pass


    def positions_to_word(self, nearest_dist:List, positions:List, words_to_ignore:List=[]):
        words = [self.pos_to_word[pos] for pos in positions]
        
        nearest_dist_final = []
        nearest_words = []
        
        #ignora as palavras de words_to_ignore
        for i,word in enumerate(words):
            if word not in words_to_ignore:
                nearest_dist_final.append(nearest_dist[i])
                nearest_words.append(word)

        return nearest_dist_final, nearest_words


    def get_most_similar_embedding(self,query:Union[np.array,str],k_most_similar:int=5,words_to_ignore:List=[]):
        #o parametro query pode ser a palavra (string) ou o proprio embedding a ser procurado.
        #Caso seja a palavra, é necessario obter o embedding correspondente
        query_embedding = query
        if type(query) == str:
            if query not in self.dict_embedding:
                return [],[]
            query_embedding = self.dict_embedding[query]

        #obtém o embedding
        nearest_dist, nearest_ind = self.kd_embedding.query([query_embedding], k_most_similar, return_distance=True)
        return self.positions_to_word(nearest_dist[0], nearest_ind[0], words_to_ignore)
    
    
    def get_embeddings_by_similarity(self,query:np.array, max_distance:float, words_to_ignore:List=[]):
        embedding = query
        if type(query) == str:
            if query not in self.dict_embedding:
                return [],[]
            embedding = self.dict_embedding[query]

        nearest_ind, nearest_dist = self.kd_embedding.query_radius([embedding], max_distance, return_distance=True)   
        return self.positions_to_word(nearest_dist[0], nearest_ind[0], words_to_ignore)

