"""Ajustes pontuais no notebook sem descartar as execuções já registradas.

Execute na raiz do projeto: python atualizar_notebook.py
O script cria uma cópia .bak e modifica somente as fontes de células específicas.
Os valores de outputs, execution_count e metadados de execução são conferidos
antes e depois da gravação.
"""

from copy import deepcopy
from pathlib import Path
import json
import shutil
import sys

ARQUIVO = Path('notebooks/tech_challenge_olist_rag.ipynb')
COPIA = ARQUIVO.with_suffix('.ipynb.bak')
MARCA = '## 18. Comparação das evidências com o resumo gerado'


def texto(celula):
    origem = celula.get('source', [])
    return ''.join(origem) if isinstance(origem, list) else origem


def alterar(celula, novo_texto):
    # O formato em lista é usado pelo arquivo original do Jupyter.
    celula['source'] = novo_texto.splitlines(keepends=True)


def celula_markdown(frases):
    return {'cell_type': 'markdown', 'id': 'a7f091c1', 'metadata': {},
            'source': frases.splitlines(keepends=True)}


def celula_codigo(frases):
    return {'cell_type': 'code', 'id': 'f4c8e162', 'execution_count': None,
            'metadata': {}, 'outputs': [], 'source': frases.splitlines(keepends=True)}


def aplicar(arquivo):
    doc = json.loads(arquivo.read_text(encoding='utf-8-sig'))
    celulas = doc.get('cells', [])
    if not celulas or not any('Olist' in texto(c) for c in celulas[:3]):
        raise ValueError('Este não parece ser o notebook do Tech Challenge da Olist.')
    if any(MARCA in texto(c) for c in celulas):
        raise ValueError('As correções já foram aplicadas neste notebook. Não é preciso repetir.')

    historico = {c.get('id', str(i)): (deepcopy(c.get('outputs', [])),
                                       c.get('execution_count'),
                                       deepcopy(c.get('metadata', {})))
                 for i, c in enumerate(celulas) if c.get('cell_type') == 'code'}

    alteracoes = []
    inicio = next((c for c in celulas if c.get('cell_type') == 'markdown'
                   and texto(c).startswith('# Tech Challenge')), None)
    if not inicio:
        raise ValueError('Não encontrei o título do notebook.')
    alterar(inicio, '''# Tech Challenge — Fase 4 | Olist: Voz do Cliente com RAG

Neste trabalho, usei comentários da Olist para testar uma forma de consultar avaliações em linguagem natural, sem perder a ligação com os pedidos originais.

Primeiro organizei as tabelas com Pandas. Depois comparei buscas com TF-IDF e LSA e utilizei os comentários recuperados como contexto para o Ollama. O MiniLM também ficou disponível como opção, sem substituir os resultados da comparação inicial.

Uma resposta feita a partir de cinco comentários representa **esses cinco exemplos**, não a opinião de todos os clientes. Por isso mantenho o `review_id` e o `order_id` e separo a etapa de busca da etapa de geração.
''')
    alteracoes.append('Apresentação do notebook')

    # No repositório publicado há duas células MiniLM, sendo a segunda uma cópia
    # sem execução. A primeira guarda o resultado histórico; ela é mantida.
    minilm = [c for c in celulas if c.get('cell_type') == 'code'
              and 'USAR_MINILM = ' in texto(c) and 'indice_rag' in texto(c)]
    if not minilm:
        raise ValueError('Não encontrei a configuração MiniLM. Nenhum arquivo foi alterado.')
    principal = minilm[0]
    original = texto(principal)
    import re
    corrigido = re.sub(r'^USAR_MINILM\s*=\s*(True|False).*$',
                       "USAR_MINILM = False  # Ative só depois de instalar requirements-minilm.txt",
                       original, count=1, flags=re.MULTILINE)
    if corrigido == original and 'USAR_MINILM = False' not in original:
        raise ValueError('Não consegui ajustar a opção MiniLM.')
    corrigido = corrigido.replace(
        '# Na comparação de 3 perguntas, o TF-IDF encontrou mais termos relevantes.\n# Por isso usei TF-IDF como padrão e deixei o MiniLM como evolução opcional.',
        '# Deixo TF-IDF como padrão para reproduzir a comparação registrada.\n# O MiniLM pode ser testado separadamente, após a instalação da biblioteca.')
    alterar(principal, corrigido)
    alteracoes.append('MiniLM desativado por padrão, mantendo o histórico')

    for copia in minilm[1:]:
        if copia.get('execution_count') is None and not copia.get('outputs'):
            celulas.remove(copia)
            alteracoes.append('Removida a cópia não executada da célula MiniLM')
        else:
            # Se houver duas versões executadas, mantenho ambas por segurança.
            alteracoes.append('Existe outra célula MiniLM executada, mantida por segurança')

    secao15 = next((c for c in celulas if c.get('cell_type') == 'markdown'
                    and '## 15. Resposta final' in texto(c)), None)
    if secao15:
        alterar(secao15, '''## 15. Resultado da busca e registro do Ollama

A primeira consulta mostra os comentários recuperados para a pergunta sobre **atraso na entrega**, sem gerar uma resposta. A outra saída que já estava salva no notebook é um registro de uso do **Ollama com uma pergunta sobre embalagens**.

Os dois registros foram mantidos como estavam. São consultas diferentes e, portanto, não devem ser comparadas como se respondessem à mesma pergunta. O texto gerado pelo Ollama cita `[E1]`, `[E4]` e `[E5]`, mas a presença dessas citações não confirma, sozinha, que cada detalhe da resposta está apoiado nas avaliações.

Para comparar corretamente as duas etapas, a seção 18 executa **a mesma pergunta e os mesmos filtros**, primeiro mostrando as evidências e depois, caso o Ollama esteja habilitado, a síntese gerada. A saída antiga desta seção é um registro histórico: não foi recalculada durante esta revisão.
''')
        alteracoes.append('Explicação dos registros anteriores do Ollama')

    secao13 = next((c for c in celulas if c.get('cell_type') == 'markdown'
                    and '## 13. Modelo de NLP' in texto(c)), None)
    if secao13:
        alterar(secao13, """## 13. Modelo de NLP pré-treinado: MiniLM (opcional)

O MiniLM é outra forma de transformar comentários em vetores. Deixei o TF-IDF como padrão para reproduzir a comparação registrada com LSA. Para experimentar o MiniLM, instale `requirements-minilm.txt`, altere `USAR_MINILM = True` e execute a partir daquela célula.

As saídas antigas foram mantidas, mesmo que apresentem uma configuração diferente. Quando a célula for executada novamente, a saída será atualizada.
""")
        alteracoes.append('Descrição do MiniLM e distinção das saídas anteriores')

    # O notebook original guarda uma saída de geração sobre embalagens. Não a
    # apago nem forço nova chamada quando alguém usar Executar Tudo.
    saidas_ollama = [c for c in celulas if c.get('cell_type') == 'code'
                     and 'USAR_OLLAMA = True' in texto(c)
                     and 'responder(' in texto(c)]
    for c in saidas_ollama:
        fonte = texto(c)
        fonte = fonte.replace('USAR_OLLAMA = True', 'USAR_OLLAMA = False')
        # Ajusta a pergunta para o tema da saída histórica; em uma nova execução,
        # a consulta continuará desativada até haver Ollama local.
        fonte = fonte.replace('resultado = responder(',
            "pergunta_ollama = 'Existem reclamações sobre embalagens dos produtos?'\nresultado_ollama = responder(")
        fonte = fonte.replace('    pergunta, indice_rag,', '    pergunta_ollama, indice_rag,')
        fonte = fonte.replace("resultado['modo']", "resultado_ollama['modo']")
        fonte = fonte.replace("resultado['resposta']", "resultado_ollama['resposta']")
        alterar(c, fonte)
        alteracoes.append('Célula de Ollama alinhada ao tema embalagem (saída anterior preservada)')

    fim = '''## 18. Comparação das evidências com o resumo gerado

Nesta comparação uso **uma única pergunta e os mesmos filtros** para as duas etapas. Primeiro vejo os comentários recuperados e seus identificadores. Depois posso pedir ao Ollama que resuma esses mesmos comentários.

O resultado da busca mede **quais exemplos foram encontrados**; a resposta do Ollama deve ser avaliada pela **fidelidade a esses exemplos**, e não pela quantidade de termos que aparecem. Uma resposta bem escrita pode continuar errada se acrescentar informações que não existem nas fontes.

O registro anterior de Ollama sobre embalagem foi preservado na seção 15. A célula abaixo é uma verificação reproduzível, ainda sem resultados salvos: só haverá nova saída do modelo depois que ele for executado no seu computador.
'''
    codigo = '''# Defina True apenas depois de instalar o Ollama e baixar llama3.2:3b.
COMPARAR_COM_OLLAMA = False
pergunta_comparacao = 'Quais reclamações aparecem sobre embalagens dos produtos?'
filtros_comparacao = {'nota_max': 2}

fontes_comparacao = buscar(pergunta_comparacao, indice_rag, k=5, **filtros_comparacao)
print('Pergunta:', pergunta_comparacao)
print('Evidências encontradas:', len(fontes_comparacao))
if fontes_comparacao:
    display(pd.DataFrame(fontes_comparacao)[['fonte', 'review_id', 'order_id', 'nota', 'texto']])
else:
    print('Nenhuma avaliação atende aos critérios da consulta.')

if COMPARAR_COM_OLLAMA:
    resultado_comparacao = responder(
        pergunta_comparacao, indice_rag, usar_ollama=True, **filtros_comparacao
    )
    print('Modo:', resultado_comparacao['modo'])
    print('Resposta:', resultado_comparacao['resposta'])
    if resultado_comparacao['modo'] == 'RAG com Ollama':
        print('Conferência: compare cada afirmação da resposta com o texto das fontes acima.')
    else:
        print('Não houve uma síntese válida do Ollama nesta consulta.')
else:
    print('Ollama desativado. Apenas as avaliações foram exibidas.')
'''
    celulas.append(celula_markdown(fim))
    celulas.append(celula_codigo(codigo))

    for i, c in enumerate(celulas):
        if c.get('cell_type') != 'code' or c.get('id') not in historico:
            continue
        outs, numero, meta = historico[c['id']]
        if (c.get('outputs') != outs or c.get('execution_count') != numero
                or c.get('metadata', {}) != meta):
            raise AssertionError(f'Uma saída ou metadado foi alterado na célula {c["id"]}.')

    # Confere que todas as células executadas originalmente permaneceram.
    for identidade, (_, numero, _) in historico.items():
        if numero is not None and not any(c.get('id') == identidade for c in celulas):
            raise AssertionError(f'A execução {numero} deixou de existir no notebook.')
    return doc, alteracoes, historico


def main():
    if not ARQUIVO.exists():
        sys.exit(f'Arquivo não encontrado: {ARQUIVO}. Execute pela raiz do repositório.')
    try:
        doc, alteracoes, historico = aplicar(ARQUIVO)
    except (ValueError, AssertionError, KeyError) as e:
        sys.exit(f'Não alterei o notebook: {e}')
    if COPIA.exists():
        sys.exit(f'Já existe o backup {COPIA}. Guarde-o em outro lugar antes de repetir.')
    shutil.copy2(ARQUIVO, COPIA)
    temp = ARQUIVO.with_suffix('.ipynb.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as arq:
        json.dump(doc, arq, ensure_ascii=False, indent=1)
        arq.write('\n')
    # Verificação final após serializar e reler.
    relido = json.loads(temp.read_text(encoding='utf-8'))
    for c in relido['cells']:
        if c.get('cell_type') == 'code' and c.get('id') in historico:
            saidas, numero, metadados = historico[c['id']]
            assert c.get('outputs') == saidas
            assert c.get('execution_count') == numero
            assert c.get('metadata', {}) == metadados
    temp.replace(ARQUIVO)
    print('Notebook atualizado:', ARQUIVO)
    print('Cópia intacta:', COPIA)
    print('Resultados e números de execução originais: preservados.')
    for item in alteracoes:
        print('-', item)


if __name__ == '__main__':
    main()
