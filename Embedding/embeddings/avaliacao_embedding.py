from base_am.avaliacao import OtimizacaoObjetivo, Experimento
from base_am.metodo import MetodoAprendizadoDeMaquina,ScikitLearnAprendizadoDeMaquina
from base_am.resultado import Fold, Resultado
import optuna
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle
from base_am.preprocessamento_atributos import PreprocessDataset
from .textual_representation import AggregateEmbeddings, InstanceWisePreprocess, BagOfWords
from sklearn.neighbors import KNeighborsClassifier

def calcula_experimento_representacao(nom_experimento:str, preproc_method:PreprocessDataset, 
                                        df_amostra:pd.DataFrame, col_classe:str, 
                                        num_folds:int, num_folds_validacao:int ,
                                        num_trials:int,ClasseObjetivoOtimizacao,sampler):

    try:
        arr_folds = Fold.gerar_k_folds(df_amostra, val_k=num_folds, col_classe=col_classe,
                                    num_repeticoes=1, num_folds_validacao=num_folds_validacao, num_repeticoes_validacao=1)


        experimento = Experimento(nom_experimento, arr_folds, 
                            ClasseObjetivoOtimizacao=ClasseObjetivoOtimizacao,
                            num_trials=num_trials, preproc_method=preproc_method,
                            sampler=sampler, load_if_exists=False)
        experimento.carrega_resultados_existentes()
    except IOError: 
        experimento.calcula_resultados()
        experimento.salva_resultados()

    return experimento
    


class OtimizacaoObjetivoRandomForest(OtimizacaoObjetivo):

    ml_method_default = ScikitLearnAprendizadoDeMaquina(RandomForestClassifier())

    def __init__(self, fold:Fold, preproc_method:PreprocessDataset):
        super().__init__(fold,preproc_method)

    def obtem_metodo(self,trial: optuna.Trial)->MetodoAprendizadoDeMaquina:

        min_samples = trial.suggest_int('min_samples_split', 1, 21, step=2)/100
        max_features = trial.suggest_int('max_features', 70, 100,step=5)/100
        num_arvores = trial.suggest_int('num_arvores', 30, 50, step=5)

        rf_method = RandomForestClassifier(min_samples_split=min_samples,n_estimators=num_arvores,
                                            max_features=max_features,random_state=2,n_jobs=6,
                                            max_samples = 0.8
                                            )

        #caso seja embedding, setar of parametros da janela
        if isinstance(self.preproc_method, InstanceWisePreprocess) and\
            isinstance(self.preproc_method.counter_function, AggregateEmbeddings):
            aggregate_embedding = self.preproc_method.counter_function
            if aggregate_embedding.aggregate_method == "hier":
                aggregate_embedding.hier_window = trial.suggest_int('hier_window', 1, 21, step=5)


        return ScikitLearnAprendizadoDeMaquina(rf_method)

    def resultado_metrica_otimizacao(self,resultado: Resultado) -> float:
        return resultado.macro_f1
class OtimizacaoObjetivoKNN(OtimizacaoObjetivo):

    ml_method_default = ScikitLearnAprendizadoDeMaquina(RandomForestClassifier())

    def __init__(self, fold:Fold, preproc_method:PreprocessDataset):
        super().__init__(fold,preproc_method)

    def obtem_metodo(self,trial: optuna.Trial)->MetodoAprendizadoDeMaquina:

        n_neighbors = trial.suggest_int('n_neighbors', 2, 6)

        knn_method = KNeighborsClassifier(n_neighbors=2**n_neighbors,n_jobs=6)

        #caso seja embedding, setar of parametros da janela
        if isinstance(self.preproc_method, InstanceWisePreprocess) and\
            isinstance(self.preproc_method.counter_function, AggregateEmbeddings):
            aggregate_embedding = self.preproc_method.counter_function
            if aggregate_embedding.aggregate_method == "hier":
                aggregate_embedding.hier_window = trial.suggest_int('hier_window', 1, 21, step=5)


        return ScikitLearnAprendizadoDeMaquina(knn_method)

    def resultado_metrica_otimizacao(self,resultado: Resultado) -> float:
        return resultado.macro_f1