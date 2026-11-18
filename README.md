# Tech Challenge — Fase 4 | Voz do Cliente com NLP e RAG

A ideia deste projeto é investigar os relatos dos clientes da Olist a partir de perguntas em linguagem natural. Para isso, tratei e relacionei as tabelas originais com Pandas, comparei diferentes formas de recuperar comentários e integrei a geração de respostas com o Ollama.

O notebook reúne a preparação dos dados, os testes dos três métodos de recuperação e um consolidado gerado ao final de cada execução. O código das buscas e da consulta ao Ollama fica em `src/rag_simples.py`.

## Dados utilizados

O arquivo de origem fica em `data/raw/olist_dataset_original.zip`. São lidas sete tabelas: avaliações, pedidos, clientes, itens, produtos, categorias e pagamentos.

Na execução registrada originalmente, havia 99.224 avaliações e 99.441 pedidos. Depois da limpeza dos textos, foram mantidas 42.343 avaliações. Os cruzamentos preservaram os 99.441 pedidos: itens e pagamentos são agrupados por `order_id` antes da junção, para evitar multiplicar linhas.

Essas quantidades são verificadas novamente durante a execução do notebook.

## Como executar

Use Python 3.11 ou 3.12. Na raiz do repositório:

```bash
python -m venv .venv
```

Ative o ambiente virtual com `source .venv/Scripts/activate` no Git Bash do Windows ou `.venv\Scripts\Activate.ps1` no PowerShell. Depois instale as bibliotecas:

```bash
python -m pip install -r requirements-minilm.txt
```

A lista `requirements-minilm.txt` também instala as dependências de `requirements.txt`.

Instale o [Ollama](https://ollama.com/), inicie o serviço e baixe o modelo que será usado nas respostas:

```bash
ollama pull llama3.2:3b
```

Abra o notebook:

```bash
jupyter notebook notebooks/tech_challenge_olist_rag.ipynb
```

Use **Run All** para refazer os cruzamentos, índices, comparações e o consolidado. O notebook está configurado com **MiniLM na recuperação principal** e **Ollama na geração**.

Se o Ollama não estiver disponível ou não gerar citações reconhecidas, a saída mostrará os comentários recuperados sem apresentar uma síntese como válida. A função registra o motivo da falha e, quando há texto sem referências, mostra a resposta bruta para diagnóstico. Se detectar uma referência inválida ou uma negação explícita que contrarie as evidências, faz mais uma tentativa com instruções mais específicas. Essa verificação de contradições é limitada e não substitui a análise humana. Isso não é contabilizado como uma síntese aceita no consolidado.

## Como as técnicas entram no fluxo

1. **TF-IDF:** representa o texto a partir de palavras e expressões.
2. **LSA:** reduz a matriz TF-IDF para representar os documentos em menos dimensões.
3. **MiniLM:** produz embeddings com um modelo multilíngue pré-treinado.
4. **RAG com Ollama:** recupera comentários, monta um contexto com os marcadores `[E1]`, `[E2]` etc. e envia esse contexto ao Llama 3.2 para gerar uma resposta citada.

Os três métodos de busca passam pelas mesmas perguntas e filtros no experimento. A contagem de palavras relacionadas ao tema é um indicador exploratório, não uma nota definitiva de relevância. Também é preciso diferenciar **recuperar um comentário** de **resumir o comentário corretamente**.

## O que sai da execução

Ao final, a pasta `results/` terá:

- `comparacao_buscas.csv`: comparativo de TF-IDF, LSA e MiniLM, com as três perguntas e a quantidade de termos relacionados nos resultados.
- `temas_avaliacoes_negativas.csv`: menções a atraso, defeito e embalagem entre as avaliações de nota 1 ou 2 com texto.
- `respostas_ollama.csv`: sínteses e marcadores de citação quando o Ollama responde, além da identificação das tentativas sem geração válida.
- `consolidado_execucao.md`: resumo construído com os números da execução corrente.
- `insights_execucao.md` e `insights_graficos_execucao.md`: leituras dos resultados calculadas pelo notebook.
- `grafico_comparacao_metodos.png`, `grafico_comparacao_por_tema.png`, `grafico_mencoes_negativas.png` e `grafico_status_ollama.png`: gráficos gerados a cada execução.

As tabelas tratadas ficam em `data/processed/`, e os embeddings em `artifacts/`. Os embeddings MiniLM são reaproveitados somente quando o registro de controle corresponde à mesma sequência de documentos; caso contrário, são recalculados.

## Como verificar

```bash
python -m unittest discover -s tests -v
```

Os testes automatizados verificam a estrutura dos dados e as funções de busca. As chamadas simuladas ao Ollama testam o contrato da integração, inclusive o diagnóstico de respostas sem referências. As respostas reais precisam ser conferidas após a execução local. O notebook preserva os resultados anteriores: saídas antigas não são uma nova validação do código alterado.

### Conferindo a geração no Ollama

No terminal, confirme o modelo com `ollama list`. Em outra janela, execute `ollama serve` somente se o serviço ainda não estiver iniciado. Depois execute o notebook, observando `Modo`, `Motivo` e, quando aplicável, `Texto recebido (não validado)`.

- **Falha de conexão:** o serviço não está acessível em `http://localhost:11434`.
- **Erro HTTP 404:** o nome do modelo pode não corresponder ao instalado (`llama3.2:3b`).
- **Texto sem `[E1]`:** o modelo respondeu, mas não seguiu o formato pedido; o código tenta mais uma vez.
- **Referência não recuperada:** uma referência como `[E99]` é rejeitada.
- **Contradição explícita:** se a resposta disser que não há produtos quebrados, mas uma evidência mencionar um acessório quebrado, a resposta é rejeitada e uma nova tentativa é feita. Essa regra cobre casos evidentes, não toda contradição possível.

Uma resposta aprovada nas verificações automáticas ainda precisa ser confrontada com os comentários originais para verificar seu conteúdo. As saídas antigas do notebook continuam salvas, mas só as células reexecutadas refletem esse novo diagnóstico.

## Limites que considero na análise

Os comentários são históricos e não representam a experiência atual dos consumidores. Uma busca que retorna cinco avaliações oferece exemplos, não uma estimativa da frequência de um problema em toda a base. A contagem por palavras depende dos termos escolhidos, e uma avaliação pode mencionar mais de um assunto.

A validação das citações verifica se os marcadores pertencem aos comentários recuperados; não comprova automaticamente que cada afirmação da resposta corresponde à fonte. Por isso, no notebook mostro as evidências usadas ao lado das respostas do Ollama.

## Organização

```text
notebooks/tech_challenge_olist_rag.ipynb
src/rag_simples.py
tests/test_projeto.py
data/raw/olist_dataset_original.zip
data/processed/
artifacts/
results/
images/
docs/
slides/
requirements.txt
requirements-minilm.txt
```

O arquivo `slides/Apresentacao.mp4` é mantido no Git LFS..
