# Entendendo os relacionamentos entre as tabelas

Comecei pela avaliação porque ela contém o comentário que será pesquisado. Usei o `order_id` para encontrar o pedido, e depois busquei informações de cliente, item, produto e pagamento.

| Base | Chaves principais | O que usei |
|---|---|---|
| `olist_order_reviews_dataset` | `review_id`, `order_id` | Nota, título, comentário |
| `olist_orders_dataset` | `order_id`, `customer_id` | Situação, compra e datas de entrega |
| `olist_customers_dataset` | `customer_id` | Estado e cidade |
| `olist_order_items_dataset` | `order_id`, `order_item_id`, `product_id` | Quantidade, valores |
| `olist_products_dataset` | `product_id`, `product_category_name` | Categoria |
| `product_category_name_translation` | `product_category_name` | Categoria em inglês |
| `olist_order_payments_dataset` | `order_id` | Valores de pagamento |

## Ordem que segui no Pandas

**1. Juntar produtos aos itens** — usei `product_id`, que identifica um produto.

```python
itens_produtos = itens.merge(
    produtos[['product_id', 'product_category_name']],
    on='product_id', how='left', validate='many_to_one'
)
```

**2. Traduzir a categoria** — usei `product_category_name`.

```python
itens_produtos = itens_produtos.merge(
    categorias, on='product_category_name', how='left', validate='many_to_one'
)
```

**3. Agrupar itens por pedido** — um pedido pode ter vários itens. Após o `groupby('order_id')`, temos uma linha por pedido, com quantidade, preço total, frete e categorias.

**4. Agrupar pagamentos por pedido** — um pedido também pode ter vários registros de pagamento. Somo os valores e fico com uma linha por `order_id`.

**5. Juntar os resumos aos pedidos** — os `merge()` com clientes, itens e pagamentos não devem mudar a quantidade de pedidos (**99.441**). O notebook confere isso com `assert`.

**6. Juntar a avaliação tratada aos pedidos** — `on='order_id'`. A avaliação ganha metadados sem deixar de ser a unidade de análise do RAG (**42.343** registros com texto).

## Cuidados

- Usei `how='left'` para manter a tabela da esquerda mesmo quando faltam informações na tabela auxiliar.
- Não substituí valores ausentes de atraso por zero: um pedido sem data de entrega não necessariamente chegou no prazo.
- Usei `fillna('')` apenas nos campos de texto para distinguir comentário vazio de uma avaliação textual.
- O mesmo `order_id` pode ter vários itens, mas o resumo por pedido mantém só uma linha; isso evita que valores e avaliações sejam multiplicados.
- Categoria do pedido pode reunir vários itens. Por isso a categoria **não prova** qual item recebeu o elogio ou reclamação.

Toda a preparação, incluindo as verificações, está nas seções **2 a 9** do notebook.
