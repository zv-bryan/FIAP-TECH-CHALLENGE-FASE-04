# Tech Challenge — Fase 4 | Voz do Cliente (Olist + RAG)

Projeto acadêmico de **NLP e RAG** para consultar avaliações de clientes da Olist. A ideia é fazer uma pergunta, encontrar avaliações relacionadas e usá-las como evidências para uma resposta.

O projeto foi escrito para quem está estudando o assunto. Os tratamentos são feitos diretamente no notebook com **Pandas**, e os relacionamentos aparecem passo a passo com `merge()` e verificações de quantidade de linhas. Não usei agentes nem treinei redes neurais.

## 1. O que foi construído

- Leitura de **7 tabelas CSV** da Olist (`pd.read_csv`).
- Limpeza de avaliações: `fillna()`, `str.strip()`, `str.replace()` e exclusão de textos vazios e duplicidades.
- Relacionamentos entre avaliações, pedidos, clientes, itens, produtos, categorias e pagamentos.
- Três bases tratadas, com comentários e informações dos pedidos.
- Vetorização de textos com **TF-IDF** e embeddings com **LSA** (TF-IDF + `TruncatedSVD`).
- Comparação exploratória entre buscas e uma alternativa com **MiniLM** pré-treinado.
- Recuperação de até cinco avaliações com ID, nota e texto original.
- Construção de contexto e integração com **Ollama** para gerar respostas com citações.
- Tratamento de perguntas fora do tema e consultas sem evidências suficientes.

### Resultado da preparação

| Etapa | Registros |
|---|---:|
| Avaliações originais | 99.224 |
| Avaliações com texto, após a limpeza | **42.343** |
| Pedidos originais | 99.441 |
| Pedidos após todos os cruzamentos | **99.441** |
| Itens originais | 112.650 |

Todos esses números são verificados no notebook. O dado **42.343** se refere a avaliações com título ou comentário de pelo menos 3 caracteres, depois da remoção de duplicidades por (`review_id`, `order_id`).

## 2. Como rodar

Recomendo Python **3.11 ou 3.12**.

```bash
# 1. Na pasta principal do projeto
python -m venv .venv

# 2. Ative o ambiente
# Windows:
.venv\Scripts\activate
# Linux ou macOS:
# source .venv/bin/activate

# 3. Instale as bibliotecas
pip install -r requirements.txt

# 4. Abra o notebook
jupyter notebook notebooks/tech_challenge_olist_rag_iniciante.ipynb
```

No Jupyter, escolha **Run All / Executar tudo**. A origem está em `data/raw/olist_dataset_original.zip`; o próprio notebook extrai os sete CSVs necessários. Os arquivos brutos extraídos são ignorados pelo Git porque já estão dentro do ZIP.

O notebook roda **sem Ollama e sem baixar modelos neurais**: nesse caso, a busca vetorial usa TF-IDF e o resultado mostra as avaliações encontradas em vez de fingir uma resposta generativa.

### Para gerar respostas usando um LLM

1. Instale [Ollama](https://ollama.com/).
2. No terminal, execute `ollama pull llama3.2:3b`.
3. Deixe o Ollama em execução.
4. No notebook, altere `USAR_OLLAMA = True` e execute a seção 15 novamente.

O código envia **a pergunta e os comentários recuperados**, não a base inteira. O prompt pede citações como `[E1]`, `[E2]`. Se a chamada falhar, o código volta a mostrar as avaliações sem fabricar uma resposta.

### Para testar embeddings de rede neural já treinada

```bash
pip install -r requirements-minilm.txt
```

Na seção 13 do notebook, mude `USAR_MINILM = True` e execute a partir daquela célula. O modelo utilizado é `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. A primeira execução exige acesso à internet e a indexação de 42 mil textos pode demorar sem GPU. Os embeddings ficam salvos em `artifacts/embeddings_minilm.npy` para reaproveitamento.

**Importante:** A geração com Ollama foi integrada, mas não executada com .

## 3. Como foram feitos os cruzamentos

```text
avaliações ─ (order_id) ─> pedidos ─ (customer_id) ─> clientes
                              |
                              +─ (order_id) ─> itens ─ (product_id) ─> produtos ─> categorias
                              |
                              +─ (order_id) ─> pagamentos
```

| Cruzamento | Como foi feito | Resultado |
|---|---|---|
| Itens + produtos | `merge(on='product_id', how='left')` | Categoria de cada produto |
| Itens + tradução de categorias | `merge(on='product_category_name', how='left')` | Nome da categoria em inglês |
| Itens por pedido | `groupby('order_id')` | Uma linha por pedido |
| Pagamentos por pedido | `groupby('order_id')` | Uma linha por pedido |
| Pedidos + clientes | `merge(on='customer_id', how='left')` | Estado e cidade |
| Pedidos + itens resumidos | `merge(on='order_id', how='left')` | Quantidade, valor e categorias |
| Pedidos + pagamentos resumidos | `merge(on='order_id', how='left')` | Valor pago |
| Avaliações + pedidos completos | `merge(on='order_id', how='left')` | **Base final do RAG** |

Os **itens e pagamentos são agrupados antes de entrar nos pedidos**, pq um pedido pode ter vários itens e vários pagamentos. Sem esse cuidado, um `merge()` multiplicaria os registros. Coloquei `validate='many_to_one'` e `validate='one_to_one'` para detectar esse erro.


## 4. Como funciona o RAG

```text
Pergunta
   ↓
TF-IDF / MiniLM (vetor da pergunta)
   ↓
Similaridade de cosseno com as avaliações
   ↓
Até 5 comentários encontrados e filtros opcionais
   ↓
Contexto com texto + review_id + nota + [E1], [E2]...
   ↓
LLM local (Ollama) quando ativado
   ↓
Resposta fundamentada com citações ou abstenção
```

A preparação **também cria embeddings LSA**, estudados na comparação. Nos testes executados com três perguntas, o TF-IDF teve um resultado melhor no indicador simples de ocorrência de palavras do tema; por isso ficou como escolha padrão enquanto o MiniLM não estiver disponível.

| Tema pesquisado | TF-IDF: termos nos 5 resultados | LSA: termos nos 5 resultados |
|---|---:|---:|
| Atraso | 5 | 5 |
| Produto quebrado | 1 | 0 |
| Embalagem | 1 | 0 |

Essa medida é **apenas uma aproximação** (presença de termos em comentários recuperados). Não é uma avaliação humana de relevância e não prova superioridade geral do TF-IDF. O arquivo original da comparação fica em `results/comparacao_buscas.csv`.

## 5. Exemplo de pergunta

> **Pergunta:** O que os clientes relatam sobre entrega atrasada?

A busca recupera avaliações com nota 1 ou 2 e mostra `review_id`, `order_id`, texto e similaridade. No modo sem Ollama, o sistema imprime trechos como “Entrega atrasada. Ainda não recebi o produto” e “entrega atrasada. o prazo terminou e nao recebi meu produto”, com os IDs originais.

Quando o Ollama estiver habilitado, ele deve usar **somente essas fontes** para compor a resposta. O código confere se as citações apresentadas pertencem aos documentos recuperados, mas **não consegue garantir sozinho que cada frase do modelo é verdadeira**. Por isso é importante conferir os comentários.

**Exemplos de abstenção:** previsão do tempo, reclamações em 2035 ou uma categoria que não aparece na base.

## 6. Estrutura do repositório

```text
voice-of-customer-rag-iniciante/
├── README.md
├── requirements.txt
├── requirements-minilm.txt
├── .gitignore
├── notebooks/
│   └── tech_challenge_olist_rag_iniciante.ipynb
├── data/
│   ├── raw/olist_dataset_original.zip
│   └── processed/
│       ├── avaliacoes_tratadas.csv.gz
│       ├── pedidos_relacionados.csv.gz
│       └── itens_enriquecidos.csv.gz
├── src/
│   └── rag_simples.py
├── tests/
│   └── test_projeto.py
├── artifacts/
│   └── embeddings_lsa.npy
├── results/
│   └── comparacao_buscas.csv
├── images/
│   └── notas_avaliacoes.png
├── docs/
│   ├── relacionamentos.md
│   └── checklist_requisitos.md
└── slides/
    ├── apresentacao_fase4_iniciante.pptx
    └── apresentacao_fase4_iniciante.pdf
```

**Testes:** `python -m unittest discover -s tests -v`.

## 7. Principais limites

- Os comentários são históricos (2016 a 2018) e não descrevem a experiência atual dos consumidores.
- TF-IDF depende das palavras presentes; LSA pode perder termos específicos. MiniLM é uma alternativa a ser avaliada, **não uma melhoria já comprovada**.
- O indicador da comparação não substitui análise manual nem uma base rotulada.
- A busca mostra apenas exemplos relevantes e **não mede a frequência de reclamações em toda a população**. Não conclua “o principal problema é X” só com cinco comentários.
- Filtros de categoria podem incluir múltiplas categorias de um pedido; isso não significa que a avaliação se refere a todos os itens.
- A etapa generativa requer Ollama local. A execução disponibilizada valida leitura, limpeza, cruzamentos, buscas e abstenção. A resposta gerada precisa ser testada em um computador com o LLM disponível.
