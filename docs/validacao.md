# Validação do projeto

## 1. O que está registrado

O notebook que foi publicado contém saídas de execução, inclusive registros de consultas e de uma resposta do Ollama. Essas saídas foram preservadas durante a correção: não são provas de uma nova execução, mas ajudam a documentar o trabalho realizado.

As conferências salvas são:

| Verificação | Resultado registrado |
|---|---:|
| Avaliações no CSV original | 99.224 |
| Avaliações com texto após o tratamento | 42.343 |
| Pedidos originais | 99.441 |
| Pedidos após relacionamentos | 99.441 |
| Itens originais e após os cruzamentos | 112.650 |
| Matriz TF-IDF | 42.343 × 12.000 |
| Embeddings LSA | 42.343 × 64 |

Essas contagens dão evidência de que o cruzamento não multiplicou pedidos. Elas não substituem a execução completa em outra máquina.

## 2. O que o teste de recuperação realmente mediu

Para três perguntas, foram observados os cinco primeiros comentários com notas até 2. O critério foi verificar a presença de termos associados ao assunto nos textos recuperados.

| Assunto | TF-IDF | LSA |
|---|---:|---:|
| Atraso | 5/5 | 5/5 |
| Produto quebrado | 1/5 | 0/5 |
| Embalagem | 1/5 | 0/5 |
| **Total exploratório** | **7/15** | **5/15** |

A contagem confirma apenas a presença das palavras procuradas. Um comentário pode abordar o assunto com termos diferentes, e um comentário pode mencionar a palavra sem trazer informação útil. Por isso não chamei esse indicador de precisão, recall ou acurácia do RAG. Os valores originais estão em `results/comparacao_buscas.csv`.

## 3. Evidências recuperadas e resposta do Ollama

A consulta salva sobre **atraso na entrega** exibe cinco comentários com nota 1, incluindo relatos de entrega fora do prazo e de produto não recebido. Essa etapa demonstra recuperação e rastreabilidade, mas não foi um resumo gerado por LLM.

Há também uma **saída salva do Ollama sobre embalagens**. Nela, o modelo organiza relatos sobre riscos, amassados e avarias e cita `[E1]`, `[E4]` e `[E5]`. Foi uma geração registrada, mas não apresentamos no mesmo bloco a lista das avaliações usadas naquela pergunta. Portanto, o texto do Ollama não foi classificado como confirmado apenas porque contém citações. Para validar as afirmações, precisamos abrir as avaliações correspondentes e conferir cada detalhe.

Para comparar as duas etapas sem misturar perguntas diferentes, foi acrescentada uma célula específica ao final do notebook: ela mostra os comentários e, quando habilitada, gera a síntese **para a mesma pergunta e com os mesmos filtros**.

## 4. Testes automatizados

Execute na raiz:

```bash
python -m unittest discover -s tests -v
```

Os testes verificam cardinalidade e campos das bases processadas, busca por similaridade, filtros, identificadores, repetição de textos e formato das citações. Os testes de integração com Ollama usam **respostas simuladas** para não depender de um servidor local. Eles confirmam a chamada da API e a rejeição de citações desconhecidas, não a factualidade de uma resposta real.

## 5. O que ainda precisa de conferência manual

- Rodar o notebook completo em um ambiente limpo, com as versões das dependências instaladas.
- Executar a comparação da seção 18 com o Ollama funcionando e guardar as evidências da mesma pergunta.
- Ler as avaliações citadas e marcar quais afirmações da síntese são sustentadas por elas.
- Verificar se o cache `embeddings_minilm.npy` corresponde à mesma ordem e versão dos textos processados.
- Se for necessário medir qualidade de forma mais robusta, montar uma pequena amostra com relevância avaliada manualmente.

As saídas antigas permaneceram no `.ipynb` por decisão de rastreabilidade. Células cujo código foi ajustado podem exibir **resultados históricos**, e esses resultados devem ser atualizados por uma nova execução antes de serem apresentados como recentes.
