import warnings
from abc import abstractmethod
import optuna
import numpy as np
import csv
import pandas as pd
from typing import List,Union
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from base_am.preprocessamento_atributos import PreprocessDataset
from .resultado import Fold,Resultado
from .metodo import MetodoAprendizadoDeMaquina

class Experimento():
    def __init__(self,nom_experimento,folds:List[Fold], 
                    ClasseObjetivoOtimizacao=None, preproc_method = None,
                    num_trials:int=100, sampler=optuna.samplers.TPESampler(seed=1, n_startup_trials=10),
                    load_if_exists=False):
        """
        folds: folds a serem usados no experimentos
        ml_method: Método de aprendizado de máquina a ser usado
        ClasseObjetivoOtimizacao: CLASSE a ser usada para otimização dos parametros
        """
        self.nom_experimento = nom_experimento
        self.preproc_method = preproc_method
        self.folds = folds
        self._resultados = None
        self.ClasseObjetivoOtimizacao = ClasseObjetivoOtimizacao
        self.num_trials = num_trials
        self.sampler = sampler
        self.load_if_exists = load_if_exists

        self.studies_per_fold = []

    @property
    def resultados(self) -> List[Resultado]:

        if self._resultados:
            return self._resultados
        return self.calcula_resultados()

    def calcula_resultados(self)  -> List[Resultado]:
        """
        Retorna, para cada fold, o seu respectivo resultado
        """
        self._resultados = []
        self.arr_validacao_por_fold = []#experimentos de validacao por fold
        #seed para mater a reprodutibilidade dos experimentos
        np.random.seed(1)
        ## Para cada fold
        for i,fold in enumerate(self.folds):

            ##1. Caso haja um metodo de otimizacao, obtenha o melhor metodo com ele
            if(self.ClasseObjetivoOtimizacao is not None):
                try:
                    if not self.load_if_exists:
                        optuna.delete_study(study_name=f"{self.nom_experimento}_fold_{i}", storage=f'sqlite:///resultados/optuna_studies.db')
                except KeyError:
                    pass
                study = optuna.create_study(study_name=f"{self.nom_experimento}_fold_{i}",sampler=self.sampler, direction="maximize", 
                                            storage=f'sqlite:///resultados/optuna_studies.db', load_if_exists=self.load_if_exists)
                objetivo_otimizacao = self.ClasseObjetivoOtimizacao(fold,self.preproc_method)
                study.optimize(objetivo_otimizacao, self.num_trials)
                #obtem o melhor metodo da otimizacao
                best_method = objetivo_otimizacao.arr_evaluated_methods[study.best_trial.number]
                self.studies_per_fold.append(study)
            else:
                #caso contrario, o metodo, atributo da classe Experimento (sem modificações) é usado
                best_method = self.ClasseObjetivoOtimizacao.ml_method_default

            ##2. adiciona em resultados o resultado predito usando o melhor metodo
            resultado = best_method.eval(self.preproc_method,fold.df_treino,fold.df_data_to_predict,fold.col_classe)
            print(resultado.macro_f1)
            self._resultados.append(resultado)
        return self._resultados
    def salva_resultados(self):
        with open(f'resultados/{self.nom_experimento}.csv', 'w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',',
                                    quotechar='"', quoting=csv.QUOTE_MINIMAL)
            
            csv_writer.writerow(["id", "fold", "true_y", "predicted_y"])     
            for fold_num,resultado in enumerate(self.resultados):
                ids = None
                if type(resultado.y) == pd.Series:
                    ids = resultado.y.index
                elif type(resultado.predict_y) == pd.Series:
                    ids = resultado.y.index
                
            
                for i,y in enumerate(resultado.y):
                    id_value = ids[i] if ids is not None else None
                    csv_writer.writerow([id_value, fold_num, y, resultado.predict_y[i]])

    def carrega_resultados_existentes(self):
        try:
            resultados = pd.read_csv(f'resultados/{self.nom_experimento}.csv',index_col="id")
        except ValueError:
            resultados = pd.read_csv(f'resultados/{self.nom_experimento}.csv')

        self._resultados = []
        self.studies_per_fold = []
        num_folds = resultados["fold"].unique()

        for fold_num in num_folds:
            study = optuna.create_study(study_name=f"{self.nom_experimento}_fold_{fold_num}",sampler=self.sampler, direction="maximize", 
                                        storage=f'sqlite:///resultados/optuna_studies.db', load_if_exists=True)
            self.studies_per_fold.append(study)
            resultados_fold = resultados[resultados["fold"] == fold_num]
            predicted_list = list(resultados_fold["predicted_y"])
            self._resultados.append(Resultado(resultados_fold["true_y"],predicted_list))
        

    @property
    def macro_f1_avg(self) -> float:
        """
        Calcula a média do f1 dos resultados.
        """
        return np.average([r.macro_f1 for r in self.resultados])





class OtimizacaoObjetivo:
    #será definido nas subclasses
    ml_method = None
    def __init__(self,  fold: Fold, preproc_method:PreprocessDataset):
        self.fold = fold
        self.arr_evaluated_methods = []
        self.preproc_method = preproc_method

    @abstractmethod
    def obtem_metodo(self, trial: optuna.Trial) ->MetodoAprendizadoDeMaquina:
        raise NotImplementedError

    @abstractmethod
    def resultado_metrica_otimizacao(self,resultado:Resultado) -> float:
        raise NotImplementedError

    def __call__(self, trial: optuna.Trial) -> float:
        #para cada fold, executa o método e calcula o resultado
        sum = 0
        metodo = self.obtem_metodo(trial)
        self.arr_evaluated_methods.append(metodo)
        for fold_validacao in self.fold.arr_folds_validacao:
            resultado = metodo.eval(self.preproc_method,fold_validacao.df_treino,fold_validacao.df_data_to_predict,self.fold.col_classe)
            sum += self.resultado_metrica_otimizacao(resultado)

        return sum/len(self.fold.arr_folds_validacao)
