# Validação registrada do projeto

A execução foi feita com Python 3.13, Pandas 2.2.3, scikit-learn 1.8.0 e Jupyter. O README recomenda Python 3.11/3.12 pela compatibilidade mais ampla.

## Resultado da execução do notebook

- Notebook executado completamente, sem exceções: **15 células de código**.
- Avaliações CSV original: **99.224**.
- Avaliações com texto após limpeza: **42.343**.
- Pedidos originais: **99.441**.
- Pedidos após cruzar clientes, itens e pagamentos: **99.441**.
- Itens originais: **112.650**; após juntar produtos e categorias: **112.650**.
- Matriz TF-IDF: **42.343 × 12.000**.
- Embeddings LSA: **42.343 × 64**.
- Busca padrão após a comparação exploratória: **TF-IDF**.
- Abstenção testada para pergunta fora do escopo, ano inexistente e categoria inexistente.

## Comparação de recuperação

A análise de três perguntas, recuperando cinco comentários com nota até 2 em cada método, verificou se os comentários continham expressões associadas à pergunta.

| Assunto | TF-IDF | LSA |
|---|---:|---:|
| Atraso | 5/5 | 5/5 |
| Produto quebrado | 1/5 | 0/5 |
| Embalagem | 1/5 | 0/5 |
| **Total exploratório** | **7/15** | **5/15** |

Essa medida **não mede precisão semântica de verdade**: é apenas presença de palavras. Ela explica a escolha local do TF-IDF, mas não estabelece superioridade em qualquer pergunta.

## Geração

A chamada HTTP ao Ollama está implementada e os testes com **resposta simulada (mock)** validam o fluxo e a rejeição de citações desconhecidas. **O LLM real não foi executado**, portanto não existem aqui métricas de factualidade gerada. Para completar a demonstração generativa, instale Ollama e execute a seção 15 do notebook com `USAR_OLLAMA=True`.

## Fontes

Os textos apresentados no notebook podem ser localizados pelos campos `review_id` e `order_id` na base Olist fornecida. O notebook contém as saídas salvas para revisão, e os arquivos CSV tratados estão no pacote.
