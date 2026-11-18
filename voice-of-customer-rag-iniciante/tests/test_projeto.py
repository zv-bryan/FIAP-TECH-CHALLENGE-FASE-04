from pathlib import Path
import sys
import unittest
from unittest.mock import patch, Mock
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.rag_simples import buscar, responder, criar_contexto


class TestarDados(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pedidos = pd.read_csv(RAIZ / 'data/processed/pedidos_relacionados.csv.gz')
        cls.documentos = pd.read_csv(RAIZ / 'data/processed/avaliacoes_tratadas.csv.gz')

    def test_merge_nao_duplicou_pedidos(self):
        self.assertEqual(len(self.pedidos), 99441)
        self.assertTrue(self.pedidos['order_id'].is_unique)

    def test_avaliacoes_textuais(self):
        self.assertEqual(len(self.documentos), 42343)
        self.assertTrue(self.documentos['id_documento'].is_unique)
        self.assertTrue(self.documentos['texto'].str.len().ge(3).all())

    def test_metadados_e_identificadores(self):
        for campo in ['review_id', 'order_id', 'review_score', 'ano_compra', 'categorias']:
            self.assertIn(campo, self.documentos.columns)


class TestarBusca(unittest.TestCase):
    def setUp(self):
        dados = pd.DataFrame([
            {'review_id':'a','order_id':'1','review_score':1,'texto':'Entrega atrasada demais','ano_compra':2018,'categorias':'books'},
            {'review_id':'b','order_id':'2','review_score':1,'texto':'Entrega demorou muitos dias','ano_compra':2018,'categorias':'books'},
            {'review_id':'c','order_id':'3','review_score':5,'texto':'Entrega muito rápida','ano_compra':2018,'categorias':'toys'},
        ])
        v = TfidfVectorizer().fit(dados['texto'])
        self.indice = {'metodo':'tfidf','dados':dados,'tfidf':v,'vetores':v.transform(dados['texto'])}

    def test_busca_preserva_fonte(self):
        encontrados = buscar('Problemas de entrega atrasada',self.indice,k=2,nota_max=2)
        self.assertTrue(encontrados)
        self.assertTrue(all('review_id' in item and 'order_id' in item for item in encontrados))

    def test_filtro_ano_sem_registros(self):
        encontrados = buscar('Entregas atrasadas',self.indice,ano=2035)
        self.assertEqual(encontrados, [])

    def test_pergunta_fora_do_tema(self):
        saida = responder('Qual a previsão do tempo?',self.indice)
        self.assertEqual(saida['modo'],'sem evidências')

    @patch('requests.post')
    def test_geracao_com_citacao_valida_mock(self, post):
        # Mock testa a integração, não é resultado real de um modelo LLM.
        post.return_value = Mock(status_code=200)
        post.return_value.raise_for_status.return_value = None
        post.return_value.json.return_value = {'response': 'Houve atraso na entrega [E1].'}
        saida = responder('Problemas de entrega atrasada', self.indice, usar_ollama=True)
        self.assertEqual(saida['modo'], 'RAG com Ollama')
        self.assertTrue(post.called)

    @patch('requests.post')
    def test_citacao_inexistente_nao_e_aceita(self, post):
        post.return_value = Mock(status_code=200)
        post.return_value.raise_for_status.return_value = None
        post.return_value.json.return_value = {'response': 'Comentário inexistente [E99].'}
        saida = responder('Problemas de entrega atrasada', self.indice, usar_ollama=True)
        self.assertEqual(saida['modo'], 'consulta de evidências')

    def test_contexto_menciona_fonte(self):
        contexto = criar_contexto(buscar('Entrega atrasada',self.indice,k=2))
        self.assertIn('[E1]',contexto)
        self.assertIn('review_id=',contexto)


if __name__ == '__main__':
    unittest.main()
