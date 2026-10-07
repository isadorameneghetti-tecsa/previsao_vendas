-- =====================================================================
-- QUERIES — PREVISÃO DE VENDAS LIAH v5 (bigframes)
-- Mesmas consultas usadas pelos notebooks (geradas a partir de liah_dados.py).
--
-- O QUE MUDOU EM RELAÇÃO À v3/v4
--   - Os notebooks NÃO precisam mais destes CSVs: leem direto do BigQuery via
--     bigframes (liah_dados.py). Este arquivo serve para conferência no console,
--     para o Metabase e para quem quiser rodar à mão.
--   - Regra de limpeza e regra de pedidos extremos agora vivem em UM lugar
--     (liah_dados.py). Se mudar lá, regere este arquivo:
--         python construir/construir_sql.py > queries_previsao_vendas_v5.sql
--   - Nova dimensão de produto: prj-data-lakehouse-prd-9459.gold_core.dim_liah_produtos
--     (marca, grupo, subgrupo). Query 0 mostra as colunas reais; os nomes usados
--     abaixo (nome_produto, marca, grupo, subgrupo, uf) são SUPOSTOS — ajuste se
--     a Query 0 mostrar outros. O notebook detecta automaticamente.
--   - Query 2 (diária) passa a trazer pedidos/faturamento extremos.
--   - Novas: Queries 10a/10b (cubos semanais de produto e geográfico) e
--            Query 11 (cesta: produtos comprados juntos).
--   - Consultas M1–M5 para o Metabase, sobre as tabelas gravadas pelos notebooks.
--
-- CUIDADOS
--   - Somente leitura (SELECT). As tabelas ficam em prj-data-lakehouse-prd-9459,
--     mas a execução/faturamento é no sandbox prj-sbx-data-liah-4956: no console,
--     selecione esse projeto no topo antes de rodar. Confira os bytes estimados.
--   - Resultados só com números agregados e ids de pedido/produto. UF é usada
--     apenas agregada. Não acrescente e-mail, nome, endereço ou dados de
--     paciente/nutricionista.
--   - As tabelas de saída (M1–M5) devem ficar num dataset de sandbox/analytics
--     revisado, NUNCA em gold_core/silver de produção.
-- =====================================================================

-- =====================================================================
-- QUERY 0 — COLUNAS DAS TABELAS (schema.csv)  [CONFERÊNCIA]
-- Confirme os nomes de marca/grupo/subgrupo/UF antes de usar as queries 10 e 11.
-- =====================================================================
SELECT 'produtos' AS tabela, column_name, data_type
FROM `prj-data-lakehouse-prd-9459.gold_core.INFORMATION_SCHEMA.COLUMNS` WHERE table_name = 'dim_liah_produtos'
UNION ALL
SELECT 'itens', column_name, data_type
FROM `prj-data-lakehouse-prd-9459.gold_core.INFORMATION_SCHEMA.COLUMNS` WHERE table_name = 'fct_liah_pedidos_itens'
UNION ALL
SELECT 'pedidos', column_name, data_type
FROM `prj-data-lakehouse-prd-9459.silver_b2c.INFORMATION_SCHEMA.COLUMNS` WHERE table_name = 'silver_b2c_pedidos';


-- =====================================================================
-- QUERY 1 — MENSAL (mensal.csv)  [BASE DA PREVISÃO]
-- Janela de ±7 dias dos pedidos idênticos está no RANGE BETWEEN 7 PRECEDING AND 7 FOLLOWING
-- (parâmetro EXTREMO['janela_dias_grupo'] em liah_dados.py).
-- =====================================================================
WITH pedidos_validos AS (
  SELECT
    id AS id_pedido,
    LOGICAL_OR(id_nutricionista IS NOT NULL) AS canal_tem_nutri,
    ANY_VALUE(UPPER(TRIM(CAST(uf AS STRING)))) AS uf_pedido
  FROM `prj-data-lakehouse-prd-9459.silver_b2c.silver_b2c_pedidos`
  WHERE (status IS NULL OR status NOT IN ('CANCELADO', 'CANCELED', 'DECLINED', 'PIX_GERADO', 'WAITING', 'PENDING'))
    AND (id_nutricionista IS NULL OR id_nutricionista NOT IN (18444, 299919, 30511))
    AND (id_paciente IS NULL OR id_paciente NOT IN (5789260))
    AND (email IS NULL OR email NOT IN ('teste@teste.com.br'))
  GROUP BY id
),
base AS (
  SELECT i.*, v.* EXCEPT (id_pedido)
  FROM `prj-data-lakehouse-prd-9459.gold_core.fct_liah_pedidos_itens` AS i
  INNER JOIN pedidos_validos AS v USING (id_pedido)
  WHERE i.pedido_date IS NOT NULL
),
itens AS (
  SELECT *, MIN(pedido_date) OVER (PARTITION BY id_produto) AS primeira_venda_produto
  FROM base
),
pedidos AS (
  SELECT
    id_pedido,
    MIN(pedido_date)                                       AS pedido_date,
    SUM(preco_total)                                       AS valor_pedido,
    SUM(COALESCE(preco_cheio, preco_unitario) * unidades)  AS valor_cheio,
    SUM(unidades)                                          AS unidades,
    COUNT(*)                                               AS linhas,
    COUNTIF(preco_unitario < preco_cheio)                  AS linhas_com_desconto,
    SUM(IF(DATE_DIFF(pedido_date, primeira_venda_produto, DAY) <= 90, preco_total, 0)) AS valor_produtos_novos
  FROM itens
  GROUP BY id_pedido
),
pedidos_flag AS (
  SELECT p.*,
    COUNT(*) OVER (
      PARTITION BY ROUND(p.valor_pedido, 2), p.linhas, p.unidades
      ORDER BY UNIX_DATE(p.pedido_date)
      RANGE BETWEEN 7 PRECEDING AND 7 FOLLOWING
    ) AS qtd_identicos_janela
  FROM pedidos p
),
extremos AS (
  SELECT
    id_pedido,
    valor_pedido > 5000 AS acima_valor_extremo,
    (qtd_identicos_janela >= 2 AND valor_pedido >= 1000 AND linhas >= 2) AS em_grupo_identico,
    (valor_pedido > 5000
     OR (qtd_identicos_janela >= 2 AND valor_pedido >= 1000 AND linhas >= 2)) AS eh_extremo
  FROM pedidos_flag
),
limiar AS (
  SELECT APPROX_QUANTILES(valor_pedido, 100)[OFFSET(99)] AS p99_geral FROM pedidos
),
produtos_mes AS (
  SELECT DATE_TRUNC(pedido_date, MONTH) AS ds, COUNT(DISTINCT id_produto) AS produtos_distintos
  FROM itens GROUP BY ds
),
mensal AS (
  SELECT
    DATE_TRUNC(p.pedido_date, MONTH)                      AS ds,
    COUNT(*)                                              AS y_total_pedidos,
    SUM(valor_pedido)                                     AS faturamento_total,
    SUM(valor_cheio)                                      AS faturamento_preco_cheio,
    SUM(unidades)                                         AS total_unidades_vendidas,
    SUM(linhas)                                           AS total_linhas,
    SUM(linhas_com_desconto)                              AS linhas_com_desconto,
    SUM(valor_produtos_novos)                             AS faturamento_produtos_novos,
    COUNT(DISTINCT p.pedido_date)                         AS dias_com_venda,
    MAX(p.pedido_date)                                    AS ultima_data,
    APPROX_QUANTILES(valor_pedido, 100)[OFFSET(50)]       AS ticket_mediano,
    APPROX_QUANTILES(valor_pedido, 100)[OFFSET(90)]       AS ticket_p90,
    MAX(valor_pedido)                                     AS maior_pedido,
    COUNTIF(valor_pedido > l.p99_geral)                   AS pedidos_grandes,
    SUM(IF(valor_pedido > l.p99_geral, valor_pedido, 0))  AS faturamento_pedidos_grandes,
    COUNTIF(e.eh_extremo)                                 AS pedidos_extremos,
    SUM(IF(e.eh_extremo, valor_pedido, 0))                AS faturamento_extremos,
    COUNTIF(e.em_grupo_identico)                          AS pedidos_grupos_identicos
  FROM pedidos p
  CROSS JOIN limiar l
  LEFT JOIN extremos e USING (id_pedido)
  GROUP BY ds
)
SELECT
  m.ds,
  EXTRACT(YEAR FROM m.ds) AS ano, EXTRACT(MONTH FROM m.ds) AS mes, EXTRACT(QUARTER FROM m.ds) AS trimestre,
  y_total_pedidos,
  ROUND(faturamento_total, 2)                                       AS faturamento_total,
  ROUND(SAFE_DIVIDE(faturamento_total, y_total_pedidos), 2)         AS ticket_medio,
  total_unidades_vendidas,
  ROUND(SAFE_DIVIDE(total_unidades_vendidas, y_total_pedidos), 2)   AS media_itens_por_pedido,
  ROUND(SAFE_DIVIDE(faturamento_total, total_unidades_vendidas), 2) AS preco_medio_unidade,
  ROUND(SAFE_DIVIDE(faturamento_preco_cheio, total_unidades_vendidas), 2) AS preco_cheio_medio_unidade,
  ROUND((1 - SAFE_DIVIDE(faturamento_total, faturamento_preco_cheio)) * 100, 2) AS desconto_medio_pct,
  ROUND(SAFE_DIVIDE(linhas_com_desconto, total_linhas) * 100, 1)   AS pct_itens_com_desconto,
  pm.produtos_distintos,
  ROUND(SAFE_DIVIDE(faturamento_produtos_novos, faturamento_total) * 100, 1) AS pct_fat_produtos_novos_90d,
  ROUND(ticket_mediano, 2) AS ticket_mediano, ROUND(ticket_p90, 2) AS ticket_p90, ROUND(maior_pedido, 2) AS maior_pedido,
  pedidos_grandes,
  ROUND(SAFE_DIVIDE(faturamento_pedidos_grandes, faturamento_total) * 100, 1) AS pct_fat_pedidos_grandes,
  ROUND(SAFE_DIVIDE(faturamento_total - faturamento_pedidos_grandes, y_total_pedidos - pedidos_grandes), 2) AS ticket_medio_sem_grandes,
  ROUND(SAFE_DIVIDE(total_linhas, y_total_pedidos), 2)              AS linhas_por_pedido,
  pedidos_extremos,
  ROUND(faturamento_extremos, 2)                                    AS faturamento_extremos,
  pedidos_grupos_identicos,
  ROUND(SAFE_DIVIDE(faturamento_total - faturamento_extremos, y_total_pedidos - pedidos_extremos), 2) AS ticket_medio_sem_extremos,
  dias_com_venda,
  EXTRACT(DAY FROM LAST_DAY(m.ds)) AS dias_no_mes,
  m.ds < DATE_TRUNC(CURRENT_DATE('America/Sao_Paulo'), MONTH) AS mes_completo,
  ultima_data
FROM mensal m
LEFT JOIN produtos_mes pm USING (ds)
ORDER BY m.ds;


-- =====================================================================
-- QUERY 2 — DIÁRIA (diaria.csv)  [BASE DA PREVISÃO]
-- Uma linha por dia (dias sem venda = 0), até ontem.
-- =====================================================================
WITH pedidos_validos AS (
  SELECT
    id AS id_pedido,
    LOGICAL_OR(id_nutricionista IS NOT NULL) AS canal_tem_nutri,
    ANY_VALUE(UPPER(TRIM(CAST(uf AS STRING)))) AS uf_pedido
  FROM `prj-data-lakehouse-prd-9459.silver_b2c.silver_b2c_pedidos`
  WHERE (status IS NULL OR status NOT IN ('CANCELADO', 'CANCELED', 'DECLINED', 'PIX_GERADO', 'WAITING', 'PENDING'))
    AND (id_nutricionista IS NULL OR id_nutricionista NOT IN (18444, 299919, 30511))
    AND (id_paciente IS NULL OR id_paciente NOT IN (5789260))
    AND (email IS NULL OR email NOT IN ('teste@teste.com.br'))
  GROUP BY id
),
base AS (
  SELECT i.*, v.* EXCEPT (id_pedido)
  FROM `prj-data-lakehouse-prd-9459.gold_core.fct_liah_pedidos_itens` AS i
  INNER JOIN pedidos_validos AS v USING (id_pedido)
  WHERE i.pedido_date IS NOT NULL
),
itens AS (
  SELECT *, MIN(pedido_date) OVER (PARTITION BY id_produto) AS primeira_venda_produto
  FROM base
),
pedidos AS (
  SELECT
    id_pedido,
    MIN(pedido_date)                                       AS pedido_date,
    SUM(preco_total)                                       AS valor_pedido,
    SUM(COALESCE(preco_cheio, preco_unitario) * unidades)  AS valor_cheio,
    SUM(unidades)                                          AS unidades,
    COUNT(*)                                               AS linhas,
    COUNTIF(preco_unitario < preco_cheio)                  AS linhas_com_desconto,
    SUM(IF(DATE_DIFF(pedido_date, primeira_venda_produto, DAY) <= 90, preco_total, 0)) AS valor_produtos_novos
  FROM itens
  GROUP BY id_pedido
),
pedidos_flag AS (
  SELECT p.*,
    COUNT(*) OVER (
      PARTITION BY ROUND(p.valor_pedido, 2), p.linhas, p.unidades
      ORDER BY UNIX_DATE(p.pedido_date)
      RANGE BETWEEN 7 PRECEDING AND 7 FOLLOWING
    ) AS qtd_identicos_janela
  FROM pedidos p
),
extremos AS (
  SELECT
    id_pedido,
    valor_pedido > 5000 AS acima_valor_extremo,
    (qtd_identicos_janela >= 2 AND valor_pedido >= 1000 AND linhas >= 2) AS em_grupo_identico,
    (valor_pedido > 5000
     OR (qtd_identicos_janela >= 2 AND valor_pedido >= 1000 AND linhas >= 2)) AS eh_extremo
  FROM pedidos_flag
),
dias AS (
  SELECT d AS ds
  FROM UNNEST(GENERATE_DATE_ARRAY((SELECT MIN(pedido_date) FROM pedidos),
                                  DATE_SUB(CURRENT_DATE('America/Sao_Paulo'), INTERVAL 1 DAY))) AS d
)
SELECT
  dias.ds,
  EXTRACT(DAYOFWEEK FROM dias.ds)                                  AS dia_semana,
  COUNT(p.id_pedido)                                               AS pedidos,
  ROUND(COALESCE(SUM(p.valor_pedido), 0), 2)                       AS faturamento,
  COALESCE(SUM(p.unidades), 0)                                     AS unidades,
  ROUND(SAFE_DIVIDE(SUM(p.valor_pedido), COUNT(p.id_pedido)), 2)   AS ticket_medio,
  ROUND((1 - SAFE_DIVIDE(SUM(p.valor_pedido), SUM(p.valor_cheio))) * 100, 2) AS desconto_medio_pct,
  COUNTIF(e.eh_extremo)                                            AS pedidos_extremos,
  ROUND(COALESCE(SUM(IF(e.eh_extremo, p.valor_pedido, 0)), 0), 2)  AS faturamento_extremos
FROM dias
LEFT JOIN pedidos p ON p.pedido_date = dias.ds
LEFT JOIN extremos e ON e.id_pedido = p.id_pedido
GROUP BY dias.ds, dia_semana
ORDER BY dias.ds;


-- =====================================================================
-- QUERY 3 — VALIDAÇÃO DA TABELA (validacao.csv)  [DIAGNÓSTICO]
-- pares_pedido_produto < linhas -> linhas repetidas: use DEDUP_ITENS = True nos notebooks.
-- =====================================================================
WITH pedidos_validos AS (
  SELECT
    id AS id_pedido,
    LOGICAL_OR(id_nutricionista IS NOT NULL) AS canal_tem_nutri
  FROM `prj-data-lakehouse-prd-9459.silver_b2c.silver_b2c_pedidos`
  WHERE (status IS NULL OR status NOT IN ('CANCELADO', 'CANCELED', 'DECLINED', 'PIX_GERADO', 'WAITING', 'PENDING'))
    AND (id_nutricionista IS NULL OR id_nutricionista NOT IN (18444, 299919, 30511))
    AND (id_paciente IS NULL OR id_paciente NOT IN (5789260))
    AND (email IS NULL OR email NOT IN ('teste@teste.com.br'))
  GROUP BY id
)
SELECT
  DATE_TRUNC(i.pedido_date, MONTH)                                     AS ds,
  COUNT(*)                                                             AS linhas,
  COUNT(DISTINCT FORMAT('%d-%d', i.id_pedido, i.id_produto))           AS pares_pedido_produto,
  COUNTIF(ABS(i.preco_total - i.preco_unitario * i.unidades) > 0.01)   AS linhas_total_diferente_unit_x_qtd,
  ROUND(SUM(i.preco_total), 2)                                         AS soma_preco_total,
  ROUND(SUM(i.preco_unitario * i.unidades), 2)                         AS soma_unitario_x_qtd,
  COUNTIF(i.preco_unitario < i.preco_cheio)                            AS linhas_com_desconto,
  COUNTIF(i.preco_total <= 0 OR i.unidades <= 0)                       AS linhas_zeradas_ou_negativas,
  MAX(i.data_atualizacao)                                              AS ultima_atualizacao
FROM `prj-data-lakehouse-prd-9459.gold_core.fct_liah_pedidos_itens` AS i
INNER JOIN pedidos_validos USING (id_pedido)
WHERE i.pedido_date IS NOT NULL
GROUP BY ds
ORDER BY ds;


-- =====================================================================
-- QUERY 4 — DIVISÃO POR ORIGEM 1P / 3P (divisao.csv)
-- Um pedido pode ter itens 1P e 3P: a soma de pedidos por origem pode passar do total.
-- =====================================================================
WITH pedidos_validos AS (
  SELECT
    id AS id_pedido,
    LOGICAL_OR(id_nutricionista IS NOT NULL) AS canal_tem_nutri,
    ANY_VALUE(UPPER(TRIM(CAST(uf AS STRING)))) AS uf_pedido
  FROM `prj-data-lakehouse-prd-9459.silver_b2c.silver_b2c_pedidos`
  WHERE (status IS NULL OR status NOT IN ('CANCELADO', 'CANCELED', 'DECLINED', 'PIX_GERADO', 'WAITING', 'PENDING'))
    AND (id_nutricionista IS NULL OR id_nutricionista NOT IN (18444, 299919, 30511))
    AND (id_paciente IS NULL OR id_paciente NOT IN (5789260))
    AND (email IS NULL OR email NOT IN ('teste@teste.com.br'))
  GROUP BY id
),
base AS (
  SELECT i.*, v.* EXCEPT (id_pedido)
  FROM `prj-data-lakehouse-prd-9459.gold_core.fct_liah_pedidos_itens` AS i
  INNER JOIN pedidos_validos AS v USING (id_pedido)
  WHERE i.pedido_date IS NOT NULL
)
SELECT
  DATE_TRUNC(pedido_date, MONTH)                                    AS ds,
  COALESCE(origem_produto, 'SEM_INFO')                              AS origem,
  COUNT(DISTINCT id_pedido)                                         AS pedidos,
  COUNT(*)                                                          AS linhas,
  SUM(unidades)                                                     AS unidades,
  ROUND(SUM(preco_total), 2)                                        AS faturamento,
  ROUND(SAFE_DIVIDE(SUM(preco_total), COUNT(DISTINCT id_pedido)), 2) AS ticket_medio
FROM base
GROUP BY 1, 2
ORDER BY 1, 2;


-- =====================================================================
-- QUERY 5 — MAIORES PEDIDOS DOS ÚLTIMOS 60 DIAS (maiores_pedidos.csv)  [DIAGNÓSTICO]

-- =====================================================================
WITH pedidos_validos AS (
  SELECT
    id AS id_pedido,
    LOGICAL_OR(id_nutricionista IS NOT NULL) AS canal_tem_nutri
  FROM `prj-data-lakehouse-prd-9459.silver_b2c.silver_b2c_pedidos`
  WHERE (status IS NULL OR status NOT IN ('CANCELADO', 'CANCELED', 'DECLINED', 'PIX_GERADO', 'WAITING', 'PENDING'))
    AND (id_nutricionista IS NULL OR id_nutricionista NOT IN (18444, 299919, 30511))
    AND (id_paciente IS NULL OR id_paciente NOT IN (5789260))
    AND (email IS NULL OR email NOT IN ('teste@teste.com.br'))
  GROUP BY id
),
base AS (
  SELECT i.*, v.* EXCEPT (id_pedido)
  FROM `prj-data-lakehouse-prd-9459.gold_core.fct_liah_pedidos_itens` AS i
  INNER JOIN pedidos_validos AS v USING (id_pedido)
  WHERE i.pedido_date IS NOT NULL
)
SELECT id_pedido, MIN(pedido_date) AS pedido_date, COUNT(*) AS linhas, SUM(unidades) AS unidades,
       ROUND(SUM(preco_total), 2) AS valor_pedido,
       ROUND(SAFE_DIVIDE(SUM(preco_total), SUM(unidades)), 2) AS valor_por_unidade,
       STRING_AGG(DISTINCT COALESCE(origem_produto, 'SEM_INFO')) AS origem
FROM base
WHERE pedido_date >= DATE_SUB(CURRENT_DATE('America/Sao_Paulo'), INTERVAL 60 DAY)
GROUP BY id_pedido ORDER BY valor_pedido DESC LIMIT 50;


-- =====================================================================
-- QUERY 7 — IMPACTO DA LIMPEZA (impacto_limpeza.csv)
-- (Queries 6 e 8 da v3 continuam válidas e não mudaram.)
-- =====================================================================
WITH pedidos_validos AS (
  SELECT
    id AS id_pedido,
    LOGICAL_OR(id_nutricionista IS NOT NULL) AS canal_tem_nutri
  FROM `prj-data-lakehouse-prd-9459.silver_b2c.silver_b2c_pedidos`
  WHERE (status IS NULL OR status NOT IN ('CANCELADO', 'CANCELED', 'DECLINED', 'PIX_GERADO', 'WAITING', 'PENDING'))
    AND (id_nutricionista IS NULL OR id_nutricionista NOT IN (18444, 299919, 30511))
    AND (id_paciente IS NULL OR id_paciente NOT IN (5789260))
    AND (email IS NULL OR email NOT IN ('teste@teste.com.br'))
  GROUP BY id
),
todos_b2c AS (SELECT DISTINCT id AS id_pedido FROM `prj-data-lakehouse-prd-9459.silver_b2c.silver_b2c_pedidos`),
ped AS (
  SELECT id_pedido, MIN(pedido_date) AS pedido_date, SUM(preco_total) AS valor
  FROM `prj-data-lakehouse-prd-9459.gold_core.fct_liah_pedidos_itens` WHERE pedido_date IS NOT NULL GROUP BY id_pedido
),
lim AS (SELECT APPROX_QUANTILES(valor, 100)[OFFSET(99)] AS p99 FROM ped),
classif AS (
  SELECT p.*,
    CASE WHEN v.id_pedido IS NOT NULL THEN 'VALIDO'
         WHEN t.id_pedido IS NOT NULL THEN 'REMOVIDO_STATUS_OU_TESTE'
         ELSE 'FORA_DA_BASE_B2C' END AS situacao,
    p.valor > l.p99 AS atipico
  FROM ped p CROSS JOIN lim l
  LEFT JOIN pedidos_validos v USING (id_pedido)
  LEFT JOIN todos_b2c t USING (id_pedido)
)
SELECT DATE_TRUNC(pedido_date, MONTH) AS ds, situacao, COUNT(*) AS pedidos, ROUND(SUM(valor), 2) AS faturamento,
       COUNTIF(atipico) AS pedidos_atipicos, ROUND(SUM(IF(atipico, valor, 0)), 2) AS faturamento_atipicos
FROM classif GROUP BY ds, situacao ORDER BY ds, situacao;


-- =====================================================================
-- QUERY 9 — PEDIDOS COM x SEM NUTRICIONISTA (canal.csv)
-- Hipótese a confirmar: sem id_nutricionista = público aberto.
-- =====================================================================
WITH pedidos_validos AS (
  SELECT
    id AS id_pedido,
    LOGICAL_OR(id_nutricionista IS NOT NULL) AS canal_tem_nutri,
    ANY_VALUE(UPPER(TRIM(CAST(uf AS STRING)))) AS uf_pedido
  FROM `prj-data-lakehouse-prd-9459.silver_b2c.silver_b2c_pedidos`
  WHERE (status IS NULL OR status NOT IN ('CANCELADO', 'CANCELED', 'DECLINED', 'PIX_GERADO', 'WAITING', 'PENDING'))
    AND (id_nutricionista IS NULL OR id_nutricionista NOT IN (18444, 299919, 30511))
    AND (id_paciente IS NULL OR id_paciente NOT IN (5789260))
    AND (email IS NULL OR email NOT IN ('teste@teste.com.br'))
  GROUP BY id
),
base AS (
  SELECT i.*, v.* EXCEPT (id_pedido)
  FROM `prj-data-lakehouse-prd-9459.gold_core.fct_liah_pedidos_itens` AS i
  INNER JOIN pedidos_validos AS v USING (id_pedido)
  WHERE i.pedido_date IS NOT NULL
),
itens AS (
  SELECT *, MIN(pedido_date) OVER (PARTITION BY id_produto) AS primeira_venda_produto
  FROM base
),
pedidos AS (
  SELECT
    id_pedido,
    MIN(pedido_date)                                       AS pedido_date,
    SUM(preco_total)                                       AS valor_pedido,
    SUM(COALESCE(preco_cheio, preco_unitario) * unidades)  AS valor_cheio,
    SUM(unidades)                                          AS unidades,
    COUNT(*)                                               AS linhas,
    COUNTIF(preco_unitario < preco_cheio)                  AS linhas_com_desconto,
    SUM(IF(DATE_DIFF(pedido_date, primeira_venda_produto, DAY) <= 90, preco_total, 0)) AS valor_produtos_novos
  FROM itens
  GROUP BY id_pedido
),
pedidos_flag AS (
  SELECT p.*,
    COUNT(*) OVER (
      PARTITION BY ROUND(p.valor_pedido, 2), p.linhas, p.unidades
      ORDER BY UNIX_DATE(p.pedido_date)
      RANGE BETWEEN 7 PRECEDING AND 7 FOLLOWING
    ) AS qtd_identicos_janela
  FROM pedidos p
),
extremos AS (
  SELECT
    id_pedido,
    valor_pedido > 5000 AS acima_valor_extremo,
    (qtd_identicos_janela >= 2 AND valor_pedido >= 1000 AND linhas >= 2) AS em_grupo_identico,
    (valor_pedido > 5000
     OR (qtd_identicos_janela >= 2 AND valor_pedido >= 1000 AND linhas >= 2)) AS eh_extremo
  FROM pedidos_flag
)
SELECT
  DATE_TRUNC(p.pedido_date, MONTH)                                   AS ds,
  IF(v.canal_tem_nutri, 'COM_NUTRICIONISTA', 'SEM_NUTRICIONISTA')    AS canal,
  COUNT(*)                                                           AS pedidos,
  ROUND(SUM(p.valor_pedido), 2)                                      AS faturamento,
  ROUND(AVG(p.valor_pedido), 2)                                      AS ticket_medio,
  COUNTIF(e.eh_extremo)                                              AS pedidos_extremos
FROM pedidos p
JOIN pedidos_validos v USING (id_pedido)
LEFT JOIN extremos e USING (id_pedido)
GROUP BY ds, canal
ORDER BY ds, canal;


-- =====================================================================
-- QUERY 10a — CUBO SEMANAL DE PRODUTO (cubo_produto)  [GRANULARIDADES E CANIBALIZAÇÃO]
-- semana × produto × marca/grupo/subgrupo × canal × extremo. Base dos níveis grupo, subgrupo, marca e produto.
-- Faturamento e unidades somam certo em qualquer corte; 'pedidos' não (um pedido tem vários produtos).
-- =====================================================================
WITH pedidos_validos AS (
  SELECT
    id AS id_pedido,
    LOGICAL_OR(id_nutricionista IS NOT NULL) AS canal_tem_nutri,
    ANY_VALUE(UPPER(TRIM(CAST(uf AS STRING)))) AS uf_pedido
  FROM `prj-data-lakehouse-prd-9459.silver_b2c.silver_b2c_pedidos`
  WHERE (status IS NULL OR status NOT IN ('CANCELADO', 'CANCELED', 'DECLINED', 'PIX_GERADO', 'WAITING', 'PENDING'))
    AND (id_nutricionista IS NULL OR id_nutricionista NOT IN (18444, 299919, 30511))
    AND (id_paciente IS NULL OR id_paciente NOT IN (5789260))
    AND (email IS NULL OR email NOT IN ('teste@teste.com.br'))
  GROUP BY id
),
base AS (
  SELECT i.*, v.* EXCEPT (id_pedido)
  FROM `prj-data-lakehouse-prd-9459.gold_core.fct_liah_pedidos_itens` AS i
  INNER JOIN pedidos_validos AS v USING (id_pedido)
  WHERE i.pedido_date IS NOT NULL
),
itens AS (
  SELECT *, MIN(pedido_date) OVER (PARTITION BY id_produto) AS primeira_venda_produto
  FROM base
),
pedidos AS (
  SELECT
    id_pedido,
    MIN(pedido_date)                                       AS pedido_date,
    SUM(preco_total)                                       AS valor_pedido,
    SUM(COALESCE(preco_cheio, preco_unitario) * unidades)  AS valor_cheio,
    SUM(unidades)                                          AS unidades,
    COUNT(*)                                               AS linhas,
    COUNTIF(preco_unitario < preco_cheio)                  AS linhas_com_desconto,
    SUM(IF(DATE_DIFF(pedido_date, primeira_venda_produto, DAY) <= 90, preco_total, 0)) AS valor_produtos_novos
  FROM itens
  GROUP BY id_pedido
),
pedidos_flag AS (
  SELECT p.*,
    COUNT(*) OVER (
      PARTITION BY ROUND(p.valor_pedido, 2), p.linhas, p.unidades
      ORDER BY UNIX_DATE(p.pedido_date)
      RANGE BETWEEN 7 PRECEDING AND 7 FOLLOWING
    ) AS qtd_identicos_janela
  FROM pedidos p
),
extremos AS (
  SELECT
    id_pedido,
    valor_pedido > 5000 AS acima_valor_extremo,
    (qtd_identicos_janela >= 2 AND valor_pedido >= 1000 AND linhas >= 2) AS em_grupo_identico,
    (valor_pedido > 5000
     OR (qtd_identicos_janela >= 2 AND valor_pedido >= 1000 AND linhas >= 2)) AS eh_extremo
  FROM pedidos_flag
),
produtos AS (
  SELECT
    CAST(id_produto AS STRING) AS id_produto,
    CAST(nome_produto AS STRING) AS produto_nome,
    CAST(marca AS STRING) AS marca,
    CAST(grupo AS STRING) AS grupo,
    CAST(subgrupo AS STRING) AS subgrupo
  FROM `prj-data-lakehouse-prd-9459.gold_core.dim_liah_produtos`
  WHERE id_produto IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (PARTITION BY CAST(id_produto AS STRING) ORDER BY 1) = 1
)
SELECT
  DATE_TRUNC(b.pedido_date, WEEK(MONDAY)) AS semana,
  CAST(b.id_produto AS STRING) AS id_produto,
  COALESCE(pr.produto_nome, CONCAT('produto ', CAST(b.id_produto AS STRING))) AS produto_nome,
  COALESCE(pr.marca, 'SEM_MARCA') AS marca,
  COALESCE(pr.grupo, 'SEM_GRUPO') AS grupo,
  COALESCE(pr.subgrupo, 'SEM_SUBGRUPO') AS subgrupo,
  IF(b.canal_tem_nutri, 'COM_NUTRICIONISTA', 'SEM_NUTRICIONISTA') AS canal,
  COALESCE(e.eh_extremo, FALSE) AS eh_extremo,
  COUNT(DISTINCT b.id_pedido)                                       AS pedidos,
  SUM(b.unidades)                                                   AS unidades,
  ROUND(SUM(b.preco_total), 2)                                      AS faturamento,
  ROUND(SUM(COALESCE(b.preco_cheio, b.preco_unitario) * b.unidades), 2) AS faturamento_preco_cheio
FROM base b
LEFT JOIN extremos e USING (id_pedido)
LEFT JOIN produtos pr ON pr.id_produto = CAST(b.id_produto AS STRING)
WHERE b.pedido_date < CURRENT_DATE('America/Sao_Paulo')
GROUP BY 1, 2, 3, 4, 5, 6, 7, 8;


-- =====================================================================
-- QUERY 10b — CUBO SEMANAL GEOGRÁFICO (cubo_geo)  [GRANULARIDADES]
-- semana × canal × origem × UF/região × extremo. Base dos níveis total, canal, origem, região e UF.
-- Os notebooks usam 10a + 10b (bem menores que um cubo único produto × UF, que pode estourar a memória do Colab).
-- =====================================================================
WITH pedidos_validos AS (
  SELECT
    id AS id_pedido,
    LOGICAL_OR(id_nutricionista IS NOT NULL) AS canal_tem_nutri,
    ANY_VALUE(UPPER(TRIM(CAST(uf AS STRING)))) AS uf_pedido
  FROM `prj-data-lakehouse-prd-9459.silver_b2c.silver_b2c_pedidos`
  WHERE (status IS NULL OR status NOT IN ('CANCELADO', 'CANCELED', 'DECLINED', 'PIX_GERADO', 'WAITING', 'PENDING'))
    AND (id_nutricionista IS NULL OR id_nutricionista NOT IN (18444, 299919, 30511))
    AND (id_paciente IS NULL OR id_paciente NOT IN (5789260))
    AND (email IS NULL OR email NOT IN ('teste@teste.com.br'))
  GROUP BY id
),
base AS (
  SELECT i.*, v.* EXCEPT (id_pedido)
  FROM `prj-data-lakehouse-prd-9459.gold_core.fct_liah_pedidos_itens` AS i
  INNER JOIN pedidos_validos AS v USING (id_pedido)
  WHERE i.pedido_date IS NOT NULL
),
itens AS (
  SELECT *, MIN(pedido_date) OVER (PARTITION BY id_produto) AS primeira_venda_produto
  FROM base
),
pedidos AS (
  SELECT
    id_pedido,
    MIN(pedido_date)                                       AS pedido_date,
    SUM(preco_total)                                       AS valor_pedido,
    SUM(COALESCE(preco_cheio, preco_unitario) * unidades)  AS valor_cheio,
    SUM(unidades)                                          AS unidades,
    COUNT(*)                                               AS linhas,
    COUNTIF(preco_unitario < preco_cheio)                  AS linhas_com_desconto,
    SUM(IF(DATE_DIFF(pedido_date, primeira_venda_produto, DAY) <= 90, preco_total, 0)) AS valor_produtos_novos
  FROM itens
  GROUP BY id_pedido
),
pedidos_flag AS (
  SELECT p.*,
    COUNT(*) OVER (
      PARTITION BY ROUND(p.valor_pedido, 2), p.linhas, p.unidades
      ORDER BY UNIX_DATE(p.pedido_date)
      RANGE BETWEEN 7 PRECEDING AND 7 FOLLOWING
    ) AS qtd_identicos_janela
  FROM pedidos p
),
extremos AS (
  SELECT
    id_pedido,
    valor_pedido > 5000 AS acima_valor_extremo,
    (qtd_identicos_janela >= 2 AND valor_pedido >= 1000 AND linhas >= 2) AS em_grupo_identico,
    (valor_pedido > 5000
     OR (qtd_identicos_janela >= 2 AND valor_pedido >= 1000 AND linhas >= 2)) AS eh_extremo
  FROM pedidos_flag
)
SELECT
  DATE_TRUNC(b.pedido_date, WEEK(MONDAY)) AS semana,
  IF(b.canal_tem_nutri, 'COM_NUTRICIONISTA', 'SEM_NUTRICIONISTA') AS canal,
  COALESCE(b.origem_produto, 'SEM_INFO') AS origem,
  COALESCE(b.uf_pedido, 'SEM_INFO') AS uf,
  CASE
    WHEN b.uf_pedido IN ('AC', 'AM', 'AP', 'PA', 'RO', 'RR', 'TO') THEN 'Norte'
    WHEN b.uf_pedido IN ('AL', 'BA', 'CE', 'MA', 'PB', 'PE', 'PI', 'RN', 'SE') THEN 'Nordeste'
    WHEN b.uf_pedido IN ('DF', 'GO', 'MS', 'MT') THEN 'Centro-Oeste'
    WHEN b.uf_pedido IN ('ES', 'MG', 'RJ', 'SP') THEN 'Sudeste'
    WHEN b.uf_pedido IN ('PR', 'RS', 'SC') THEN 'Sul'
    ELSE 'SEM_INFO' END AS regiao,
  COALESCE(e.eh_extremo, FALSE) AS eh_extremo,
  COUNT(DISTINCT b.id_pedido)                                       AS pedidos,
  SUM(b.unidades)                                                   AS unidades,
  ROUND(SUM(b.preco_total), 2)                                      AS faturamento,
  ROUND(SUM(COALESCE(b.preco_cheio, b.preco_unitario) * b.unidades), 2) AS faturamento_preco_cheio
FROM base b
LEFT JOIN extremos e USING (id_pedido)
WHERE b.pedido_date < CURRENT_DATE('America/Sao_Paulo')
GROUP BY 1, 2, 3, 4, 5, 6;


-- =====================================================================
-- QUERY 11 — CESTA: PRODUTOS COMPRADOS JUNTOS (cesta_pares)  [CANIBALIZAÇÃO]
-- lift = pedidos_juntos × total_pedidos ÷ (pedidos_a × pedidos_b).
-- lift < 1: raramente juntos (substitutos?) | lift > 1,5: complementares.
-- =====================================================================
WITH pedidos_validos AS (
  SELECT
    id AS id_pedido,
    LOGICAL_OR(id_nutricionista IS NOT NULL) AS canal_tem_nutri,
    ANY_VALUE(UPPER(TRIM(CAST(uf AS STRING)))) AS uf_pedido
  FROM `prj-data-lakehouse-prd-9459.silver_b2c.silver_b2c_pedidos`
  WHERE (status IS NULL OR status NOT IN ('CANCELADO', 'CANCELED', 'DECLINED', 'PIX_GERADO', 'WAITING', 'PENDING'))
    AND (id_nutricionista IS NULL OR id_nutricionista NOT IN (18444, 299919, 30511))
    AND (id_paciente IS NULL OR id_paciente NOT IN (5789260))
    AND (email IS NULL OR email NOT IN ('teste@teste.com.br'))
  GROUP BY id
),
base AS (
  SELECT i.*, v.* EXCEPT (id_pedido)
  FROM `prj-data-lakehouse-prd-9459.gold_core.fct_liah_pedidos_itens` AS i
  INNER JOIN pedidos_validos AS v USING (id_pedido)
  WHERE i.pedido_date IS NOT NULL
),
itens AS (
  SELECT *, MIN(pedido_date) OVER (PARTITION BY id_produto) AS primeira_venda_produto
  FROM base
),
pedidos AS (
  SELECT
    id_pedido,
    MIN(pedido_date)                                       AS pedido_date,
    SUM(preco_total)                                       AS valor_pedido,
    SUM(COALESCE(preco_cheio, preco_unitario) * unidades)  AS valor_cheio,
    SUM(unidades)                                          AS unidades,
    COUNT(*)                                               AS linhas,
    COUNTIF(preco_unitario < preco_cheio)                  AS linhas_com_desconto,
    SUM(IF(DATE_DIFF(pedido_date, primeira_venda_produto, DAY) <= 90, preco_total, 0)) AS valor_produtos_novos
  FROM itens
  GROUP BY id_pedido
),
pedidos_flag AS (
  SELECT p.*,
    COUNT(*) OVER (
      PARTITION BY ROUND(p.valor_pedido, 2), p.linhas, p.unidades
      ORDER BY UNIX_DATE(p.pedido_date)
      RANGE BETWEEN 7 PRECEDING AND 7 FOLLOWING
    ) AS qtd_identicos_janela
  FROM pedidos p
),
extremos AS (
  SELECT
    id_pedido,
    valor_pedido > 5000 AS acima_valor_extremo,
    (qtd_identicos_janela >= 2 AND valor_pedido >= 1000 AND linhas >= 2) AS em_grupo_identico,
    (valor_pedido > 5000
     OR (qtd_identicos_janela >= 2 AND valor_pedido >= 1000 AND linhas >= 2)) AS eh_extremo
  FROM pedidos_flag
)
, janela AS (
  SELECT DISTINCT b.id_pedido, CAST(b.id_produto AS STRING) AS id_produto
  FROM base b LEFT JOIN extremos e USING (id_pedido)
  WHERE b.pedido_date >= DATE_SUB(CURRENT_DATE('America/Sao_Paulo'), INTERVAL 365 DAY)
    AND NOT COALESCE(e.eh_extremo, FALSE)
),
top AS (
  SELECT CAST(id_produto AS STRING) AS id_produto
  FROM base WHERE pedido_date >= DATE_SUB(CURRENT_DATE('America/Sao_Paulo'), INTERVAL 365 DAY)
  GROUP BY 1 ORDER BY SUM(preco_total) DESC LIMIT 150
),
ip AS (SELECT j.* FROM janela j JOIN top USING (id_produto)),
total AS (SELECT COUNT(DISTINCT id_pedido) AS n FROM janela),
ind AS (SELECT id_produto, COUNT(*) AS n FROM ip GROUP BY id_produto)
SELECT a.id_produto AS produto_a, b.id_produto AS produto_b,
       COUNT(*) AS pedidos_juntos, ia.n AS pedidos_a, ib.n AS pedidos_b, ANY_VALUE(t.n) AS total_pedidos
FROM ip a
JOIN ip b ON a.id_pedido = b.id_pedido AND a.id_produto < b.id_produto
JOIN ind ia ON ia.id_produto = a.id_produto
JOIN ind ib ON ib.id_produto = b.id_produto
CROSS JOIN total t
GROUP BY produto_a, produto_b, ia.n, ib.n;


-- #####################################################################
-- CONSULTAS PARA O METABASE (sobre as tabelas gravadas pelos notebooks)
-- Troque prj-sbx-data-liah-4956.SEU_DATASET pelo DATASET_SAIDA usado nos notebooks.
-- #####################################################################

-- =====================================================================
-- M1 — PREVISÃO DO MÊS (cartão principal)
-- Última rodada. Use "explicacao" como texto do cartão.
-- =====================================================================
SELECT mes, pedidos_previstos, ticket_base, v1_sem_extremos, v1_min, v1_max,
       extremos_esperado, v2_com_extremos, v2_min, v2_max, modelo_pedidos, explicacao
FROM `prj-sbx-data-liah-4956.SEU_DATASET.liah_previsao_mensal`
WHERE data_base = (SELECT MAX(data_base) FROM `prj-sbx-data-liah-4956.SEU_DATASET.liah_previsao_mensal`)
ORDER BY mes;


-- =====================================================================
-- M2 — REAL × PREVISTO POR CORTE (filtros Metabase: {{nivel}}, {{serie}})

-- =====================================================================
SELECT r.nivel, r.serie, r.mes, r.realizado, CAST(NULL AS FLOAT64) AS previsto,
       CAST(NULL AS FLOAT64) AS previsto_min, CAST(NULL AS FLOAT64) AS previsto_max
FROM `prj-sbx-data-liah-4956.SEU_DATASET.liah_realizado_granular` r
WHERE r.nivel = {{nivel}} [[AND r.serie = {{serie}}]]
UNION ALL
SELECT p.nivel, p.serie, p.mes, NULL, p.previsto_reconciliado, p.previsto_min, p.previsto_max
FROM `prj-sbx-data-liah-4956.SEU_DATASET.liah_previsao_granular` p
WHERE p.data_base = (SELECT MAX(data_base) FROM `prj-sbx-data-liah-4956.SEU_DATASET.liah_previsao_granular`)
  AND p.nivel = {{nivel}} [[AND p.serie = {{serie}}]]
ORDER BY serie, mes;


-- =====================================================================
-- M3 — ACURÁCIA DAS PREVISÕES PASSADAS
-- Compara cada previsão registrada (V1, sem extremos) com o realizado do mês fechado.
-- "realizado" vem da Query 1 (faturamento_total - faturamento_extremos); salve a Query 1 como pergunta do
-- Metabase ou materialize em `prj-sbx-data-liah-4956.SEU_DATASET.liah_realizado_mensal`.
-- =====================================================================
SELECT h.mes_alvo, h.horizonte, h.data_base, h.modelo_pedidos,
       h.fat_prev_final AS previsto, r.fat_base AS realizado,
       ROUND((h.fat_prev_final / NULLIF(r.fat_base, 0) - 1) * 100, 1) AS erro_pct
FROM `prj-sbx-data-liah-4956.SEU_DATASET.liah_historico_previsoes` h
JOIN `prj-sbx-data-liah-4956.SEU_DATASET.liah_realizado_mensal` r ON r.ds = h.mes_alvo
WHERE r.mes_completo
ORDER BY h.mes_alvo, h.horizonte;


-- =====================================================================
-- M4 — POR QUE ESTE MODELO? (benchmark por nível)

-- =====================================================================
SELECT nivel, modelo, complexidade, wape_pct AS erro_mensal_pct, vies_pct, escolhido, justificativa
FROM `prj-sbx-data-liah-4956.SEU_DATASET.liah_benchmark_modelos`
WHERE data_base = (SELECT MAX(data_base) FROM `prj-sbx-data-liah-4956.SEU_DATASET.liah_benchmark_modelos`)
ORDER BY nivel, wape_pct;


-- =====================================================================
-- M5 — QUEM CONCORRE COM QUEM (canibalização)

-- =====================================================================
SELECT produto_a, marca_a, produto_b, marca_b, classificacao, explicacao,
       taxa_canibalizacao_par, correlacao, lift_cesta
FROM `prj-sbx-data-liah-4956.SEU_DATASET.liah_canibalizacao_pares`
WHERE data_base = (SELECT MAX(data_base) FROM `prj-sbx-data-liah-4956.SEU_DATASET.liah_canibalizacao_pares`)
  AND classificacao <> 'Sem relação clara'
ORDER BY classificacao, forca DESC;

