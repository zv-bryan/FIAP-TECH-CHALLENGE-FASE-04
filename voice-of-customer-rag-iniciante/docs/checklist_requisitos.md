# Checklist do desafio (PDF Tech Challenge Fase 4)

O objetivo é mostrar onde cada parte do enunciado foi atendida no material.

| Requisito | Onde está | Situação |
|---|---|---|
| Seleção e preparação dos textos | Notebook §2, §4 | Executado |
| Valores ausentes / remoção de textos vazios | Notebook §4 | Executado |
| Organização dos documentos e relacionamentos | Notebook §3, §5–9; `docs/relacionamentos.md` | Executado |
| Embeddings | Notebook §11 (LSA) e §13 (MiniLM opcional) | LSA executado; MiniLM opcional não executado |
| Armazenamento e indexação | `data/processed/`, `artifacts/`, notebook §9–11 | Executado |
| Processar pergunta e recuperar avaliações | `src/rag_simples.py`; notebook §14 | Executado |
| Selecionar comentários relevantes e construir contexto | `src/rag_simples.py`; notebook §14 | Executado |
| Geração de resposta por LLM | `src/rag_simples.py`; notebook §15 | Implementado; requer Ollama para executar |
| Evidências e rastreabilidade | `review_id`, `order_id`, `[E1]` etc. no notebook §14–15 | Executado |
| Perguntas sem evidências / fora do escopo | Notebook §16 e testes | Executado |
| Metadados (extra) | Nota, ano, categoria e atraso | Executado |
| Comparação de técnicas e resultados | Notebook §12; `results/comparacao_buscas.csv` | Executado; métrica exploratória |
| Código, instruções e README | `README.md`, `src/`, `requirements.txt` | Incluído |
| Arquitetura e decisões técnicas | `README.md`; notebook §3, §11, §14 | Incluído |
| PowerPoint de apresentação | `slides/` | Incluído |
| Vídeo até 7 minutos | Não incluído por solicitação | **Pendente em relação ao PDF oficial** |
| Trilha complementar para pontos adicionais | Não incluída | Opcional; bônus não solicitado |

**O que falta para uma demonstração 100% executada de ponta a ponta?** Rodar o Ollama com `llama3.2:3b`, gerar e revisar respostas e gravar o vídeo, caso o avaliador o exija. A etapa de geração foi programada, mas não existem resultados reais de LLM local neste ambiente.
