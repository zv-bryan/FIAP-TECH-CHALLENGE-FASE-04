# Leitura dos resultados desta execução

## Preparação e relacionamentos
A partir de 99,224 avaliações originais, foram mantidas 42,343 com texto. Os relacionamentos preservaram 99,441 pedidos. Essa verificação mostra que os cruzamentos não multiplicaram a tabela central de pedidos.

## Comparação entre as buscas
Os três métodos foram consultados com as mesmas perguntas, cinco resultados por pergunta e filtro de nota até 2.
- TF-IDF: 7 de 15 comentários com expressões do tema (46.7%).
- LSA: 5 de 15 comentários com expressões do tema (33.3%).
- MiniLM: 8 de 15 comentários com expressões do tema (53.3%).
Esse indicador registra presença de palavras selecionadas; não corresponde a precisão semântica nem demonstra que um modelo é melhor em qualquer contexto.

### Diferenças por pergunta
- **Atraso** — TF-IDF: 5/5; LSA: 5/5; MiniLM: 5/5.
- **Quebrado** — TF-IDF: 1/5; LSA: 0/5; MiniLM: 1/5.
- **Embalagem** — TF-IDF: 1/5; LSA: 0/5; MiniLM: 2/5.
Na pergunta sobre produtos quebrados, comentários sobre defeito podem ser pertinentes sem conter as raízes específicas usadas na contagem. É preciso ler as evidências antes de julgar a qualidade da busca.

## Avaliações negativas
Foram encontradas 10,971 avaliações com texto e nota 1 ou 2.
- Atraso ou prazo: 3,520 menções (32.1% das avaliações negativas com texto).
- Produto danificado ou com defeito: 691 menções (6.3% das avaliações negativas com texto).
- Embalagem ou caixa: 475 menções (4.3% das avaliações negativas com texto).
Entre 9,354 avaliações negativas com datas suficientes, 2,784 (29.8%) tiveram entrega após a data estimada.
Menções e atrasos calculados são medidas diferentes, com denominadores diferentes. Os temas podem coexistir e não é possível atribuir causalidade a partir dessas contagens.

## Geração com Ollama
Houve 3 respostas que passaram pelas verificações automáticas e foram registradas como RAG com Ollama em 3 tentativas. Dessas, 3 tiveram marcadores de citação aceitos pelo verificador.
A verificação automática confere referências e algumas negações contraditórias, mas não garante fidelidade integral. As referências existem entre as fontes recuperadas; é necessário revisar a fidelidade de cada afirmação.

## Limites
Os dados são históricos, as contagens por palavras são exploratórias e as evidências do top 5 não representam toda a população de pedidos.
