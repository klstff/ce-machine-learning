from abc import abstractmethod
from .resultado import Resultado
import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin
from base_am.preprocessamento_atributos import PreprocessDataset
from typing import List,Union
class MetodoAprendizadoDeMaquina:

    @abstractmethod
    def eval(self,df_treino:pd.DataFrame, df_data_to_predict:pd.DataFrame, col_classe:str, params_preprocessamento) -> Resultado:
        raise NotImplementedError

class ScikitLearnAprendizadoDeMaquina(MetodoAprendizadoDeMaquina):
    x_treino = None
    #Dica: Union é usado quando um parametro pode receber mais de um tipo
    #neste caso, ml_method é um ClassifierMixin ou RegressorMixin
    #essas duas classes são superclasses dos classficadores e métodos de regressão
    def __init__(self,ml_method:Union[ClassifierMixin,RegressorMixin]):
        self.ml_method = ml_method

    def eval(self, preproc_method:PreprocessDataset, df_treino:pd.DataFrame, df_data_to_predict:pd.DataFrame, col_classe:str, seed:int=1) -> Resultado:

        #faz o preprocessamento
        df_preproc_treino = preproc_method.preprocess_train_dataset(df_treino, col_classe)
        df_preproc_to_predict = preproc_method.preprocess_test_dataset(df_data_to_predict, col_classe)
        
        #a partir de df_preproc_treino, separe os atributos  da classe
        x_treino = df_preproc_treino.drop(col_classe,axis=1)
        y_treino = df_preproc_treino[col_classe]
        


        #crie o modelo
        model = self.ml_method.fit(x_treino,y_treino)

        #faça a mesma separação nos dados a serem previstos
        x_to_predict = df_preproc_to_predict.drop(col_classe,axis=1)
        y_to_predict = df_preproc_to_predict[col_classe]


        #Impressao do x e y
        #print("X_treino: "+str(x_treino))
        #print("y_treino: "+str(y_treino))
        #print("X_to_predict: "+str(x_to_predict))
        #print("y_to_predict: "+str(y_to_predict))

        #retorne o resultado
        y_predictions = model.predict(x_to_predict)
        return Resultado(y_to_predict,y_predictions)
