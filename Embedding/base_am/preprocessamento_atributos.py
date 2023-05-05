from abc import abstractmethod
from typing import Set

class PreprocessDataset:
    def __init__(self, nome):
        self.nome = nome

    def _preprocessed_to_dataframe(self,df_data,df_preprocessed_data,class_col):
        df_preprocessed_data[class_col] = df_data[class_col]
        return df_preprocessed_data
    
    def preprocess_train_dataset(self,df_data, class_col):
        df_preprocessed_data = self.generate_preproc_train(df_data, class_col)
        return self._preprocessed_to_dataframe(df_data, df_preprocessed_data,class_col)

    def preprocess_test_dataset(self,df_data, class_col):
        df_preprocessed_data = self.generate_preproc_test(df_data, class_col)
        return self._preprocessed_to_dataframe(df_data, df_preprocessed_data,class_col)

    def generate_preproc_train(self,df_data, class_col):
        return self.generate_preproc_test(df_data, class_col)

    @abstractmethod
    def generate_preproc_test(self,df_data, class_col):
        raise NotImplementedError

