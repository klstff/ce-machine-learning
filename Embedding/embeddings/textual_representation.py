from nltk.tokenize import word_tokenize
from scipy.spatial import distance
import pandas as pd
import numpy as np
from base_am.preprocessamento_atributos import PreprocessDataset
from sklearn.feature_extraction.text import TfidfVectorizer




class InstanceWisePreprocess(PreprocessDataset):
    def __init__(self, nome, counter_function,text_col="text"):
        super().__init__(nome)
        self.text_col = text_col
        self.counter_function = counter_function

    def generate_preproc_test(self, df_text_reviews:pd.DataFrame,class_col:str=None):
        data_per_text = {}

        for id_pos,text in df_text_reviews[self.text_col].items():
            dict_count_key_words = self.counter_function(text) 
            if class_col:           
                dict_count_key_words[class_col] = df_text_reviews[class_col].loc[id_pos]
            data_per_text[id_pos] = dict_count_key_words
        return pd.DataFrame.from_dict(data_per_text,orient="index")

class CountWords:
    def __init__(self,dict_embedding, keywords_features, max_distance):
        self.keywords_features = keywords_features
        self.max_distance = max_distance
        self.dict_embedding = dict_embedding
        
    def __call__(self,text):
        dict_word_per_key_words = {}
        dict_count_key_words = {}
        for key_word in self.keywords_features.keys():
            dict_count_key_words[key_word] = 0
            dict_word_per_key_words[key_word] = []
            #print('key_word: '+str(key_word))
        
        words_in_text = word_tokenize(text.lower())
        #para cada palavra no texto, é buscado no dicionario keywords_features
        #se existe uma palavra similar para contabiliza-la
        for word_in_text in words_in_text:
            if word_in_text not in self.dict_embedding:
                continue
            
            embedding_word_in_text = self.dict_embedding[word_in_text] 
            
            for key_word, list_syms in self.keywords_features.items():
                words_to_search = [key_word]
                words_to_search.extend(list_syms)
                #print('key_word: '+key_word)
                for word_to_search in words_to_search:
                    if word_to_search not in self.dict_embedding:
                        continue
                    embedding_to_search = self.dict_embedding[word_to_search]
                    val_distance = distance.cosine(embedding_word_in_text, embedding_to_search)
                    #caso tenha encontrado a palavra, para de procurar
                    if val_distance <= self.max_distance:
                        dict_count_key_words[key_word] += 1
                        dict_word_per_key_words[key_word].append(word_in_text)
                        break
                    

                        
        #print(f"==============\n{text}\n{dict_word_per_key_words}\n\n")
        return dict_count_key_words


class AggregateEmbeddings:
    def __init__(self, dict_embedding, aggregate_method, 
                        words_to_filter=set(),words_to_consider=set()):
        self.dict_embedding = dict_embedding
        self.aggregate_method = aggregate_method
        self.words_to_filter = words_to_filter
        self.words_to_consider = words_to_consider
    


    def text_embedding_representation(self, word_embeddings:np.array):
        """
            word_embeddings: Matriz no formato n x m em que n é o número
            de palavra no texto e m é o tamanho do embedding

            Retorna um dicionário  com m elemenetos agregando as n palavras deste texto 
        """

        result = {}
        for j in range(word_embeddings.shape[1]):
            if self.aggregate_method == "avg":
                result[j] = np.average(word_embeddings[:,j])
            elif self.aggregate_method == "max":
                result[j] = np.max(word_embeddings[:,j])


        return result
    def __call__(self,text):

        words_in_text = word_tokenize(text.lower())

        #para cada palavra no texto, adiciona na lista de embeddings para fazer
        #a operação
        text_embeddings = []
        for word in words_in_text:
            if word not in self.dict_embedding or word in self.words_to_filter or\
                (len(self.words_to_consider)>0 and word not in self.words_to_consider):
                continue
            text_embeddings.append(self.dict_embedding[word])
        #transforma a lista em np.array
        #print(f"Tamanho: {len(text_embeddings)}")
        #quando nao foi possivel encontrar nenhuma palavra, 
        #cria um vetor de zeros
        if len(text_embeddings) == 0:
            #obtem uma palavra qualquer para saber a dimensão do embedding
            any_embedding = next(iter(self.dict_embedding.values()))
            text_embeddings = np.zeros((1,any_embedding.shape[0]))        
        else:
            text_embeddings = np.array(text_embeddings)

        #transforma em dicionario como representação
        dict_representation = self.text_embedding_representation(text_embeddings)

             
        return dict_representation
    


    


class BagOfWords(PreprocessDataset):
    def __init__(self, nome,text_col="text", stop_words=None, words_to_consider=None):
        super().__init__(nome)
        #norm: normalização para que todos os valores fiquem entre 0 e 1
        self.vectorizer = TfidfVectorizer(norm="l2", stop_words=stop_words, vocabulary=words_to_consider)
        self.text_col = text_col

    def generate_preproc_train(self,df_data:pd.DataFrame, class_col:str) -> pd.DataFrame:
        mat_bow = self.vectorizer.fit_transform(df_data[self.text_col])
        return pd.DataFrame(mat_bow.toarray(),columns=self.vectorizer.get_feature_names(),index=df_data.index)

    def generate_preproc_test(self, df_data:pd.DataFrame, class_col:str) -> pd.DataFrame:
        mat_bow = self.vectorizer.transform(df_data[self.text_col])
        return pd.DataFrame(mat_bow.toarray(),columns=self.vectorizer.get_feature_names(),index=df_data.index)