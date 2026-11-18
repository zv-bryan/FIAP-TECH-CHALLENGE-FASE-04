"""Testes de integridade das bases e das principais funções de busca."""

from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.rag_simples import buscar, criar_contexto, responder


class TestarDados(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pasta = RAIZ / 'data' / 'processed'
        if not (pasta / 'pedidos_relacionados.csv.gz').exists():
            raise unittest.SkipTest('Execute o notebook para gerar data/processed primeiro.')
        cls.pedidos = pd.read_csv(pasta / 'pedidos_relacionados.csv.gz')
        cls.documentos = pd.read_csv(pasta / 'avaliacoes_tratadas.csv.gz')

    def test_pedidos_sem_duplicidade(self):
        self.assertEqual(len(self.pedidos), 99441)
        self.assertTrue(self.pedidos['order_id'].is_unique)

    def test_documentos(self):
        self.assertEqual(len(self.documentos), 42343)
        self.assertTrue(self.documentos['id_documento'].is_unique)
        self.assertTrue(self.documentos['texto'].str.len().ge(3).all())
        for campo in ['review_id', 'order_id', 'review_score', 'ano_compra', 'categorias']:
            self.assertIn(campo, self.documentos.columns)


class TestarBusca(unittest.TestCase):
    def setUp(self):
        dados = pd.DataFrame([
            {'review_id': 'a', 'order_id': '1', 'review_score': 1,
             'texto': 'Entrega atrasada demais', 'ano_compra': 2018, 'categorias': 'books'},
            {'review_id': 'b', 'order_id': '2', 'review_score': 1,
             'texto': 'Entrega demorou muitos dias', 'ano_compra': 2018, 'categorias': 'books'},
            {'review_id': 'c', 'order_id': '3', 'review_score': 5,
             'texto': 'Entrega muito rápida', 'ano_compra': 2018, 'categorias': 'toys'},
            {'review_id': 'd', 'order_id': '4', 'review_score': 1,
             'texto': 'Entrega atrasada demais', 'ano_compra': 2018, 'categorias': 'books'},
        ])
        vetor = TfidfVectorizer().fit(dados['texto'])
        self.indice = {'metodo': 'tfidf', 'dados': dados,
                       'tfidf': vetor, 'vetores': vetor.transform(dados['texto'])}

    def test_preserva_identificadores(self):
        encontrados = buscar('Problemas de entrega atrasada', self.indice, k=3, nota_max=2)
        self.assertGreaterEqual(len(encontrados), 2)
        self.assertTrue(all(x['review_id'] and x['order_id'] for x in encontrados))

    def test_nao_repete_textos(self):
        encontrados = buscar('Entrega atrasada', self.indice, k=4)
        self.assertEqual(len({x['texto'] for x in encontrados}), len(encontrados))

    def test_filtros(self):
        self.assertEqual(buscar('Entrega atrasada', self.indice, ano=2035), [])
        self.assertEqual(buscar('Entrega atrasada', self.indice, categoria='nao_existe'), [])
        self.assertTrue(all(x['nota'] <= 2 for x in buscar(
            'Entrega atrasada', self.indice, nota_max=2)))

    def test_pergunta_fora_do_tema(self):
        self.assertEqual(responder('Qual a previsão do tempo?', self.indice)['modo'],
                         'sem evidências')

    def test_contexto_tem_ids(self):
        texto = criar_contexto(buscar('Entrega atrasada', self.indice, k=2))
        self.assertIn('[E1]', texto)
        self.assertIn('review_id=', texto)
        self.assertIn('order_id=', texto)

    @patch('requests.post')
    def test_integracao_ollama_simulada(self, post):
        post.return_value = Mock()
        post.return_value.json.return_value = {'response': 'Há relatos de atraso na entrega [E1].'}
        saida = responder('Problemas de entrega atrasada', self.indice, usar_ollama=True)
        self.assertEqual(saida['modo'], 'RAG com Ollama')
        self.assertTrue(post.called)

    @patch('requests.post')
    def test_rejeita_citacao_que_nao_foi_recuperada(self, post):
        post.return_value = Mock()
        post.return_value.json.return_value = {'response': 'Comentário [E99].'}
        saida = responder('Problemas de entrega atrasada', self.indice, usar_ollama=True)
        self.assertEqual(saida['modo'], 'consulta de evidências')

    @patch('requests.post')
    def test_rejeita_resposta_sem_citacao(self, post):
        post.return_value = Mock()
        post.return_value.json.return_value = {'response': 'Há atrasos nas entregas.'}
        saida = responder('Problemas de entrega atrasada', self.indice, usar_ollama=True)
        self.assertEqual(saida['modo'], 'consulta de evidências')

    @patch('requests.post')
    def test_recupera_resposta_sem_citacoes_na_segunda_tentativa(self, post):
        post.return_value = Mock()
        post.return_value.json.side_effect = [
            {'response': 'Houve atrasos na entrega.'},
            {'response': 'Houve atrasos [E1] e demora [E2].'},
        ]
        saida = responder('Problemas de entrega atrasada', self.indice, usar_ollama=True)
        self.assertEqual(saida['modo'], 'RAG com Ollama')
        self.assertEqual(saida['diagnostico']['tentativas'], 2)

    @patch('requests.post')
    def test_diagnostico_quando_ollama_nao_cita_fontes(self, post):
        post.return_value = Mock()
        post.return_value.json.return_value = {'response': 'Há atraso nas entregas.'}
        saida = responder('Problemas de entrega atrasada', self.indice, usar_ollama=True)
        self.assertEqual(saida['modo'], 'consulta de evidências')
        self.assertIn('sem referências', saida['observacao'])
        self.assertEqual(saida['diagnostico']['tentativas'], 2)
        self.assertEqual(saida['diagnostico']['resposta_bruta'], 'Há atraso nas entregas.')


if __name__ == '__main__':
    unittest.main()

# Os testes a seguir verificam o problema observado no notebook executado:
# o modelo citou fontes existentes, mas negou um acessório quebrado presente em [E1].
class TestarFidelidadeBasica(unittest.TestCase):
    def setUp(self):
        from src.rag_simples import conferir_citacoes_e_contradicoes
        self.conferir = conferir_citacoes_e_contradicoes
        self.fontes = [
            {'fonte': 'E1', 'texto': 'Produto de qualidade ruim, acessório veio quebrado.'},
            {'fonte': 'E2', 'texto': 'Produto com defeito; solicitei devolução.'},
        ]

    def test_rejeita_negacao_explicita_contraditoria(self):
        resposta = 'Não há menções explícitas a produtos quebrados [E2].'
        aceito, motivo = self.conferir(resposta, self.fontes)
        self.assertFalse(aceito)
        self.assertIn('negou', motivo)

    def test_aceita_relato_citado_sem_contradicao(self):
        aceito, _ = self.conferir('Há um relato de acessório quebrado [E1].', self.fontes)
        self.assertTrue(aceito)

    def test_nao_aceita_fonte_inexistente(self):
        aceito, _ = self.conferir('Há relatos de defeito [E99].', self.fontes)
        self.assertFalse(aceito)

    def test_nao_aceita_resposta_sem_citacao(self):
        aceito, _ = self.conferir('O produto veio quebrado.', self.fontes)
        self.assertFalse(aceito)
