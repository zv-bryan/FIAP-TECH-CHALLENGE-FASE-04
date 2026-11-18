# Checklist — Tech Challenge Fase 4

Esta tabela indica onde encontrei cada entrega no repositório e o que ainda depende de revisão manual.

| Item | Onde conferir | Situação |
|---|---|---|
| Seleção e limpeza de avaliações | Notebook §2 e §4 | Executado no registro original |
| Tratamento de valores ausentes e comentários vazios | Notebook §4 | Executado no registro original |
| Cruzamento entre pedidos, clientes, itens e produtos | Notebook §3 e §5–9; `docs/relacionamentos.md` | Executado no registro original |
| Embeddings e indexação | Notebook §11 (TF-IDF/LSA), §13 (MiniLM opcional) | TF-IDF/LSA registrados; MiniLM depende de configuração |
| Busca por perguntas e seleção de comentários | `src/rag_simples.py`; notebook §14 | Implementado |
| Construção de contexto com fontes | `criar_contexto()` | Implementado |
| Resposta com LLM local | `responder()`; notebook §15 e §18 | Integração implementada; há saída anterior salva |
| Conferência de citações | Código e testes | IDs validados; sentido das afirmações exige revisão |
| Respostas sem evidências | Notebook §16 e testes | Implementado |
| Comparação de estratégias de recuperação | Notebook §12; `results/comparacao_buscas.csv` | Comparação lexical exploratória |
| Documentação e arquivos de execução | README, código e `requirements.txt` | Incluídos |
| Apresentação PowerPoint | `slides/TechChallenge_Fase4_RAG_Olist.pptx` | Incluída |
| Vídeo de apresentação | `slides/Apresentacao.mp4` | Incluído, cerca de 6min06s |

**Limite da validação:** ainda não há avaliação manual concluída para determinar se todas as afirmações do texto gerado pelo Ollama são sustentadas pelas avaliações recuperadas. A execução antiga foi mantida, e a comparação reproduzível está na seção 18 do notebook.
