import unittest
from embeddings.utils import *

class TestEmbeddings(unittest.TestCase):
    def setUp(self):
        self.dict_embeddings = {"rei":np.array([1, 2, 3], dtype=np.float16),
                                  "pé de moleque":np.array([-1.2, 3.2, 1.2], dtype=np.float16),
                                  "minas gerais":np.array([12.2, 31.2, 11.2], dtype=np.float16),
                                  
                                  "rainha":np.array([-3, 0, 1], dtype=np.float16),
                                  "junina":np.array([11, 56, 32.2], dtype=np.float16),
                                  "mg":np.array([0, 0.2, 0.4], dtype=np.float16),
                                  
                                  "homem":np.array([2, 1, 1], dtype=np.float16),
                                  "ovo":np.array([0.21, 12, 1.3], dtype=np.float16),
                                  "amazonas":np.array([1.23, 0.1, 1.2], dtype=np.float16),

                                  "mulher":np.array([0.1,0.3,0], dtype=np.float16),
                                  "pascoa":np.array([1,2,1], dtype=np.float16),
                                  "am":np.array([1,1,2], dtype=np.float16),
                                  }
        self.analogy = Analogy(self.dict_embeddings, "kdtree.teste", overwrite_kdtree=True)

    def test_get_embeddings(self):
        dict_resultado_esperado = self.dict_embeddings
        #testa com embeddings de 3 dimensões
        dict_resultado = get_embedding("teste.3.txt", 3, overwrite=True)
        self.assertSetEqual(set(dict_resultado.keys()),set(dict_resultado_esperado.keys()))
        for termo,embedding in dict_resultado.items():
            self.assertEqual(len(embedding),3,"O embedding deveria possuir 3 dimensões")
            self.assertListEqual(list(embedding), list(dict_resultado_esperado[termo]),f"embedding do termo {termo} obtido incorretamente. Esperado: {dict_resultado_esperado[termo]} obtido: {list(embedding)}" )

    def test_calculo_analogia(self):
        arr_palavra_x = ["rei","pé de moleque","minas gerais"]
        arr_esta_para = ["rainha","junina","mg"]
        arr_assim_como = ["homem", "pascoa","amazonas"]
        arr_resultado_esperado = [[-2, -1, -1],
                                    [ 13.203125, 54.8125, 31.984375],
                                    [ -10.96875, -30.90625, -9.6015625 ]]

        
        for i,palavra_x in enumerate(arr_palavra_x):
            esta_para = arr_esta_para[i]
            assim_como = arr_assim_como[i]

            embedding_esperado = arr_resultado_esperado[i]
            embedding = self.analogy.calcula_embedding_analogia(palavra_x, esta_para, assim_como)
            for j,val_posicao in enumerate(embedding):
                self.assertAlmostEqual(embedding_esperado[j], float(val_posicao),msg=f"O embedding do teste com a frase "+\
                                        f"{arr_palavra_x[i]} esta para {arr_esta_para[i]} assim como {arr_assim_como[i]}"+\
                                        " resultou em um embedding incorreto' "+\
                                        f"embedding esperado: {embedding_esperado} resultante: {embedding}")
    
    
    def check_distances_and_words(self, word, distances, words, 
                                        distances_expected, words_expected):
        self.assertEqual(len(distances),len(distances_expected),"Não retornou o mesmo número de distancias")
        self.assertEqual(len(words),len(words_expected),"Não retornou o mesmo número de palavras próximas)")
        
        self.assertListEqual(words, words_expected,"Palavras próximas não esperadas" )
        for i,distance in enumerate(distances_expected):
            self.assertAlmostEqual(distance, distances_expected[i],"Distancia inesperada")

    def test_get_most_similar_embedding(self):
        dict_distances_expected = {"rei":[0, 1.41421356],
                              "rainha":[0, 3.06594194],
                              "pé de moleque":[] ,
                              "pascoa":[1.4142135623730951]}
        dict_words_expected = {"rei": ['rei', 'am'],
                        "rainha": ['rainha', 'mg'],
                        "pé de moleque": [],
                        "pascoa": ['homem']}
        for word,arr_distances_expected in dict_distances_expected.items():
            distances, words = self.analogy.kdtree_embedding.get_most_similar_embedding(word,k_most_similar=2,words_to_ignore=['pé de moleque','pascoa'])
            #print(f"Palavra: {word} distances: {distances} words: {words}")
            self.check_distances_and_words(word, distances, words,
                                            arr_distances_expected, dict_words_expected[word])
    def test_embeddings_by_similarity(self):
        dict_distances_expected = {"rei":[0, 2, 1.41421356],
                              "rainha":[],
                              "pé de moleque": [0, 2.51377469e+00] ,
                              "pascoa":[2.0, 2.513774688918143, 2.144829359336292, 
                                        2.167920185954209, 0.0, 1.4142135623730951]}
        dict_words_expected = {"rei": ['rei', 'pascoa', 'am'],
                        "rainha": [],
                        "pé de moleque": ['pé de moleque', 'pascoa'],
                        "pascoa": ['rei', 'pé de moleque', 'mg', 'mulher', 'pascoa', 'am']}

        for word, arr_distances_expected in dict_distances_expected.items():
            distances,words = self.analogy.kdtree_embedding.get_embeddings_by_similarity(word,max_distance=3,words_to_ignore=['rainha','homem', 'amazonas'])
            #print(f"Palavra: {word} distances: {distances} words: {words}")
            self.check_distances_and_words(word, distances,words,
                                    arr_distances_expected, dict_words_expected[word])



    def test_analogy(self):
        arr_analogias = [("rei","rainha","homem"),
                            ("pé de moleque","junina","ovo"),
                            ("minas gerais","mg","amazonas")]
        arr_palavras_esperadas=  [["mulher","mg","amazonas","pascoa"],
                                ["minas gerais","rei"],
                                ["rainha","mulher","am"]]

        for i,(palavra, esta_para, assim_como) in enumerate(arr_analogias):           
            palavras = self.analogy.analogia(palavra,esta_para,assim_como)
            #print(f"{palavra} está para {esta_para} assim como {assim_como} está para {palavras[0]} (ou {palavras[1:]})")
            self.assertListEqual(arr_palavras_esperadas[i], palavras, "Resultado inesperado da analogia")
if __name__ == "__main__":
    unittest.main()