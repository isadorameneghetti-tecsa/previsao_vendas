"""
liah_dados.py — acesso aos dados da Liah no BigQuery via BigQuery DataFrames (bigframes).

Fonte única para:
  * a REGRA DE LIMPEZA de pedidos (cancelados, PIX não pago, pendentes, testes);
  * a regra de PEDIDOS EXTREMOS (> R$ 5 mil ou grupos de pedidos idênticos);
  * a DETECÇÃO DE COLUNAS das tabelas (dim_liah_produtos, UF etc.), para o código
    se ajustar ao schema real sem editar SQL na mão;
  * as consultas usadas pelos notebooks (mesmas do arquivo queries_previsao_vendas_v5.sql).

Os resultados só trazem números agregados e ids de pedido/produto. Não acrescente
e-mail, nome, endereço ou qualquer dado de paciente/nutricionista.

Uso:
    import liah_dados as ld
    fonte = ld.Fonte(modo='bigquery')            # ou modo='offline' (lê cache_dados/*.csv)
    cols = ld.detectar_colunas(fonte)
    mensal = fonte.df(ld.sql_mensal(cols), 'mensal')          # pandas
    cubo = fonte.bf(ld.sql_cubo_semanal(cols))                 # bigframes (fica no BigQuery)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================
PROJETO = 'prj-data-lakehouse-prd-9459'           # onde estão as TABELAS (só leitura)
PROJETO_EXECUCAO = 'prj-sbx-data-liah-4956'       # onde as consultas RODAM e são FATURADAS
T_PEDIDOS = f'{PROJETO}.silver_b2c.silver_b2c_pedidos'
T_ITENS = f'{PROJETO}.gold_core.fct_liah_pedidos_itens'
T_PRODUTOS = f'{PROJETO}.gold_core.dim_liah_produtos'

# Regra de limpeza (antes repetida em cada query; agora só aqui)
STATUS_EXCLUIR = ['CANCELADO', 'CANCELED', 'DECLINED', 'PIX_GERADO', 'WAITING', 'PENDING']
NUTRI_TESTE = [18444, 299919, 30511]      # usuários de teste
PACIENTE_TESTE = [5789260]                # usuário de teste
EMAIL_TESTE = ['teste@teste.com.br']      # conta de teste

# Regra de pedido EXTREMO (tratado à parte na previsão)
EXTREMO = dict(valor_extremo=5000, valor_min_grupo=1000, linhas_min_grupo=2, janela_dias_grupo=7)

# Nomes candidatos para cada campo lógico. A detecção usa o 1º que existir na tabela.
# Se o nome real for outro, acrescente aqui ou passe em `manual=` para detectar_colunas().
CANDIDATOS = {
    'produto_id': ['id_produto', 'produto_id', 'id', 'sku_id', 'id_sku'],
    'produto_nome': ['nome_produto', 'produto', 'nome', 'descricao', 'titulo', 'nome_sku'],
    'marca': ['marca', 'nome_marca', 'brand', 'fabricante'],
    'grupo': ['grupo', 'grupo_produto', 'categoria', 'nome_categoria', 'departamento', 'categoria_principal'],
    'subgrupo': ['subgrupo', 'sub_grupo', 'subcategoria', 'sub_categoria', 'nome_subcategoria', 'tipo_produto', 'linha'],
    'produto_atualizacao': ['data_atualizacao', 'updated_at', 'dt_atualizacao', 'data_alteracao'],
    'uf': ['uf', 'sigla_uf', 'estado', 'uf_entrega', 'estado_entrega', 'uf_cliente', 'endereco_uf'],
}

REGIAO_POR_UF = {
    'Norte': ['AC', 'AM', 'AP', 'PA', 'RO', 'RR', 'TO'],
    'Nordeste': ['AL', 'BA', 'CE', 'MA', 'PB', 'PE', 'PI', 'RN', 'SE'],
    'Centro-Oeste': ['DF', 'GO', 'MS', 'MT'],
    'Sudeste': ['ES', 'MG', 'RJ', 'SP'],
    'Sul': ['PR', 'RS', 'SC'],
}


def _sem_fuso(s):
    try:
        s = pd.to_datetime(s)
    except (TypeError, ValueError):
        s = pd.to_datetime(s.astype(str))
    return s.dt.tz_localize(None) if getattr(s.dt, 'tz', None) is not None else s


def _tipos_simples(d: pd.DataFrame) -> pd.DataFrame:
    """bigframes devolve tipos 'nullable'/pyarrow (Int64, Float64, date32...). Converte para tipos numpy simples."""
    for c in d.columns:
        t = str(d[c].dtype)
        if t in ('Int64', 'Float64', 'Int32') or ('int' in t and 'pyarrow' in t) or ('double' in t) or ('decimal' in t) or ('numeric' in t):
            # to_numpy em vez de pd.to_numeric: no pandas 2.2 do Colab, to_numeric em colunas pyarrow com
            # valores nulos devolve menos linhas que o índice ("Length of values ... does not match")
            try:
                d[c] = d[c].to_numpy(dtype='float64', na_value=np.nan)
            except (TypeError, ValueError):
                d[c] = pd.to_numeric(d[c].astype('object'), errors='coerce').astype('float64').to_numpy()
        elif t == 'boolean' or 'bool' in t:
            d[c] = d[c].astype('object').fillna(False).astype(bool)
        elif 'date' in t or 'timestamp' in t:
            d[c] = _sem_fuso(d[c])
    return d


def para_pandas(x, nome=None, fonte=None):
    """Converte resultado bigframes -> pandas (no offline já é pandas). Grava no cache se `nome`."""
    d = _tipos_simples(x.to_pandas()) if hasattr(x, 'to_pandas') and not isinstance(x, pd.DataFrame) else x
    if isinstance(d, pd.DataFrame):
        if not isinstance(d.index, pd.RangeIndex):
            # groupby(as_index=False) no bigframes devolve índice sem nome: descarta; índice nomeado vira coluna
            d = d.reset_index(drop=all(n is None for n in d.index.names))
        if nome and fonte is not None and fonte.modo == 'bigquery':
            d.to_csv(os.path.join(fonte.pasta_cache, f'{nome_seguro(nome)}.csv'), index=False)
    return d


def nome_seguro(txt):
    """Texto -> nome válido para arquivo/coluna do BigQuery (sem acento, espaço ou símbolo)."""
    import re
    import unicodedata
    t = unicodedata.normalize('NFKD', str(txt)).encode('ascii', 'ignore').decode()
    t = t.replace('%', 'pct')
    t = re.sub(r'[^A-Za-z0-9_]+', '_', t).strip('_')
    return ('c_' + t) if t[:1].isdigit() else (t or 'col')


def _lista(v):
    return ', '.join(repr(x) if isinstance(x, str) else str(x) for x in v)


# ============================================================================
# CONEXÃO (bigframes) + CACHE
# ============================================================================
class Fonte:
    """Executa SQL no BigQuery com bigframes e guarda cópia local (cache) de cada resultado.

    modo='bigquery': consulta o BigQuery e grava cache_dados/<nome>.csv
    modo='offline' : lê cache_dados/<nome>.csv (reprocessar sem custo, ou testar sem acesso)
    """

    def __init__(self, modo='bigquery', pasta_cache='cache_dados', projeto_execucao=None, location=None,
                 reusar_cache_horas=12):
        self.modo = modo
        self.pasta_cache = pasta_cache
        # Economia: se a MESMA consulta já rodou há menos de N horas, lê a cópia local em vez de consultar de novo.
        # 0 = sempre consultar o BigQuery.
        self.reusar_cache_horas = reusar_cache_horas
        self.consultas_bq, self.consultas_cache = [], []
        os.makedirs(pasta_cache, exist_ok=True)
        self.bpd = None
        if modo == 'bigquery':
            import bigframes.pandas as bpd
            # fecha sessão anterior: permite trocar projeto/região e rodar a célula de novo sem reiniciar o Colab
            try:
                bpd.close_session()
            except Exception:
                pass
            bpd.options.bigquery.project = projeto_execucao or PROJETO_EXECUCAO
            if location:
                bpd.options.bigquery.location = location
            self.bpd = bpd

    # -- bigframes DataFrame (processamento continua no BigQuery)
    def bf(self, sql):
        if self.bpd is None:
            raise RuntimeError('fonte.bf() precisa de modo="bigquery".')
        return self.bpd.read_gbq(sql)

    # -- pandas DataFrame (resultado pequeno, já agregado)
    @staticmethod
    def _assinatura(sql):
        """Consulta + data de hoje (São Paulo): as queries usam CURRENT_DATE, então o cache nunca atravessa a meia-noite."""
        import hashlib
        hoje = pd.Timestamp.now(tz='America/Sao_Paulo').strftime('%Y-%m-%d')
        return hashlib.md5((hoje + '|' + sql).encode()).hexdigest()

    def _cache_valido(self, arq, sql):
        import hashlib
        import time
        assinatura = arq + '.sql.md5'
        if not (self.reusar_cache_horas and os.path.exists(arq) and os.path.exists(assinatura)):
            return False
        mesma_consulta = open(assinatura).read().strip() == self._assinatura(sql)
        recente = (time.time() - os.path.getmtime(arq)) / 3600 < self.reusar_cache_horas
        return mesma_consulta and recente

    def df(self, sql, nome, datas=('ds',)):
        import hashlib
        arq = os.path.join(self.pasta_cache, f'{nome}.csv')
        if self.modo == 'bigquery' and self._cache_valido(arq, sql):
            d = pd.read_csv(arq)
            self.consultas_cache.append(nome)
        elif self.modo == 'bigquery':
            d = _tipos_simples(self.bf(sql).to_pandas())
            d.to_csv(arq, index=False)
            with open(arq + '.sql.md5', 'w') as f:
                f.write(self._assinatura(sql))
            self.consultas_bq.append(nome)
        else:
            if not os.path.exists(arq):
                raise FileNotFoundError(f'{arq} não existe no cache (rode antes em modo bigquery).')
            d = pd.read_csv(arq)
        for c in datas:
            if c in d.columns:
                d[c] = _sem_fuso(d[c])
        return d

    def resumo_custo(self):
        """Quantas consultas foram ao BigQuery nesta execução e quantas vieram do cache (custo zero)."""
        print(f'Consultas ao BigQuery: {len(self.consultas_bq)} {self.consultas_bq} | '
              f'reaproveitadas do cache (sem custo): {len(self.consultas_cache)} {self.consultas_cache}')

    def bf_ou_df(self, sql, nome, datas=('ds',)):
        """bigframes no modo bigquery (dados ficam no BigQuery); pandas do cache no offline.
        O código de agregação é o mesmo nos dois casos (API estilo pandas)."""
        if self.modo == 'bigquery':
            return self.bf(sql)
        return self.df(sql, nome, datas)

    def salvar_tabela(self, df: pd.DataFrame, destino: str | None, modo='replace'):
        """Grava um resultado no BigQuery para o Metabase. destino=None -> não grava.
        ATENÇÃO: use um dataset de sandbox/analytics, nunca gold_core/silver de produção."""
        if not destino or self.bpd is None:
            return False
        if any(s in destino for s in ('.gold_core.', '.silver_b2c.', '.silver_', '.bronze_')):
            raise ValueError(f'Destino {destino} parece ser camada de produção. Use um dataset de sandbox revisado.')
        d = df.reset_index(drop=True).copy()
        d.columns = [nome_seguro(c) for c in d.columns]
        for c in d.columns:   # colunas com tipos misturados viram texto (evita erro de schema)
            if d[c].dtype == object:
                d[c] = d[c].map(lambda v: None if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))
        self.bpd.read_pandas(d).to_gbq(destino, if_exists=modo)
        return True


# ============================================================================
# DETECÇÃO DE COLUNAS
# ============================================================================
@dataclass
class Colunas:
    produto_id: str | None = None
    produto_nome: str | None = None
    marca: str | None = None
    grupo: str | None = None
    subgrupo: str | None = None
    produto_atualizacao: str | None = None
    uf: str | None = None
    uf_tabela: str | None = None          # 'pedidos' ou 'itens'
    detalhes: dict = field(default_factory=dict)

    @property
    def tem_produto(self):
        return self.produto_id is not None

    @property
    def tem_uf(self):
        return self.uf is not None

    def resumo(self):
        linhas = []
        for k in ['produto_id', 'produto_nome', 'marca', 'grupo', 'subgrupo', 'produto_atualizacao', 'uf']:
            v = getattr(self, k)
            onde = {'uf': self.uf_tabela}.get(k, 'dim_liah_produtos') if v else ''
            linhas.append({'campo': k, 'coluna encontrada': v or '— não encontrada —', 'tabela': onde})
        return pd.DataFrame(linhas)


def sql_schema():
    return f"""
SELECT 'produtos' AS tabela, column_name, data_type
FROM `{PROJETO}.gold_core.INFORMATION_SCHEMA.COLUMNS` WHERE table_name = '{T_PRODUTOS.split('.')[-1]}'
UNION ALL
SELECT 'itens', column_name, data_type
FROM `{PROJETO}.gold_core.INFORMATION_SCHEMA.COLUMNS` WHERE table_name = '{T_ITENS.split('.')[-1]}'
UNION ALL
SELECT 'pedidos', column_name, data_type
FROM `{PROJETO}.silver_b2c.INFORMATION_SCHEMA.COLUMNS` WHERE table_name = '{T_PEDIDOS.split('.')[-1]}'
"""


def detectar_colunas(fonte: Fonte, manual: dict | None = None) -> Colunas:
    """Lê o schema das 3 tabelas e escolhe as colunas de produto/UF. `manual` sobrepõe a detecção,
    ex.: manual={'grupo': 'categoria_nivel1', 'subgrupo': 'categoria_nivel2'}."""
    sch = fonte.df(sql_schema(), 'schema', datas=())
    sch['column_name'] = sch['column_name'].str.lower()
    por_tab = {t: list(g['column_name']) for t, g in sch.groupby('tabela')}
    prod = por_tab.get('produtos', [])
    c = Colunas(detalhes=por_tab)

    def achar(campo, cols):
        return next((x for x in CANDIDATOS[campo] if x in cols), None)

    for campo in ['produto_id', 'produto_nome', 'marca', 'grupo', 'subgrupo', 'produto_atualizacao']:
        setattr(c, campo, achar(campo, prod))
    # UF: prefere a tabela de pedidos (1 por pedido); senão, itens
    for tab in ('pedidos', 'itens'):
        uf = achar('uf', por_tab.get(tab, []))
        if uf:
            c.uf, c.uf_tabela = uf, tab
            break
    for k, v in (manual or {}).items():
        setattr(c, k, v)
        if k == 'uf' and not c.uf_tabela:
            c.uf_tabela = 'pedidos'
    if c.produto_id and 'id_produto' not in por_tab.get('itens', ['id_produto']):
        print('AVISO: fct_liah_pedidos_itens sem id_produto — confira o nome da chave de produto.')
    return c


# ============================================================================
# BLOCOS SQL
# ============================================================================
def _filtro_validos():
    return f"""(status IS NULL OR status NOT IN ({_lista(STATUS_EXCLUIR)}))
    AND (id_nutricionista IS NULL OR id_nutricionista NOT IN ({_lista(NUTRI_TESTE)}))
    AND (id_paciente IS NULL OR id_paciente NOT IN ({_lista(PACIENTE_TESTE)}))
    AND (email IS NULL OR email NOT IN ({_lista(EMAIL_TESTE)}))"""


def cte_pedidos_validos(cols: Colunas | None = None):
    uf = ''
    if cols is not None and cols.uf and cols.uf_tabela == 'pedidos':
        uf = f",\n    ANY_VALUE(UPPER(TRIM(CAST({cols.uf} AS STRING)))) AS uf_pedido"
    return f"""pedidos_validos AS (
  SELECT
    id AS id_pedido,
    LOGICAL_OR(id_nutricionista IS NOT NULL) AS canal_tem_nutri{uf}
  FROM `{T_PEDIDOS}`
  WHERE {_filtro_validos()}
  GROUP BY id
)"""


def cte_base(dedup=False):
    q = "\n  QUALIFY ROW_NUMBER() OVER (PARTITION BY i.id_pedido, i.id_produto ORDER BY i.data_atualizacao DESC) = 1" if dedup else ''
    return f"""base AS (
  SELECT i.*, v.* EXCEPT (id_pedido)
  FROM `{T_ITENS}` AS i
  INNER JOIN pedidos_validos AS v USING (id_pedido)
  WHERE i.pedido_date IS NOT NULL{q}
)"""


def ctes_extremos():
    """CTEs pedidos (nível pedido) + pedidos_flag + extremos(id_pedido, eh_extremo). Requer 'base'."""
    x = EXTREMO
    return f"""itens AS (
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
      RANGE BETWEEN {x['janela_dias_grupo']} PRECEDING AND {x['janela_dias_grupo']} FOLLOWING
    ) AS qtd_identicos_janela
  FROM pedidos p
),
extremos AS (
  SELECT
    id_pedido,
    valor_pedido > {x['valor_extremo']} AS acima_valor_extremo,
    (qtd_identicos_janela >= 2 AND valor_pedido >= {x['valor_min_grupo']} AND linhas >= {x['linhas_min_grupo']}) AS em_grupo_identico,
    (valor_pedido > {x['valor_extremo']}
     OR (qtd_identicos_janela >= 2 AND valor_pedido >= {x['valor_min_grupo']} AND linhas >= {x['linhas_min_grupo']})) AS eh_extremo
  FROM pedidos_flag
)"""


def cte_produtos(cols: Colunas):
    """Dimensão de produtos com nomes padronizados (id_produto STRING, nome, marca, grupo, subgrupo)."""
    if not cols.tem_produto:
        return """produtos AS (SELECT CAST(NULL AS STRING) AS id_produto, CAST(NULL AS STRING) AS produto_nome,
  CAST(NULL AS STRING) AS marca, CAST(NULL AS STRING) AS grupo, CAST(NULL AS STRING) AS subgrupo LIMIT 0)"""
    def campo(c, alias):
        return f"CAST({c} AS STRING) AS {alias}" if c else f"CAST(NULL AS STRING) AS {alias}"
    sel = ',\n    '.join([f"CAST({cols.produto_id} AS STRING) AS id_produto",
                          campo(cols.produto_nome, 'produto_nome'), campo(cols.marca, 'marca'),
                          campo(cols.grupo, 'grupo'), campo(cols.subgrupo, 'subgrupo')])
    ordem = f"{cols.produto_atualizacao} DESC" if cols.produto_atualizacao else "1"
    return f"""produtos AS (
  SELECT
    {sel}
  FROM `{T_PRODUTOS}`
  WHERE {cols.produto_id} IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (PARTITION BY CAST({cols.produto_id} AS STRING) ORDER BY {ordem}) = 1
)"""


def _expr_uf(cols: Colunas, alias='b'):
    if not cols or not cols.uf:
        return "CAST(NULL AS STRING)"
    if cols.uf_tabela == 'pedidos':
        return f"{alias}.uf_pedido"
    return f"UPPER(TRIM(CAST({alias}.{cols.uf} AS STRING)))"


def _expr_regiao(expr_uf):
    casos = '\n'.join(f"    WHEN {expr_uf} IN ({_lista(ufs)}) THEN '{reg}'" for reg, ufs in REGIAO_POR_UF.items())
    return f"CASE\n{casos}\n    ELSE 'SEM_INFO' END"


def _with(*ctes):
    return 'WITH ' + ',\n'.join(ctes)


# ============================================================================
# CONSULTAS
# ============================================================================
def sql_mensal(cols=None, dedup=False):
    """Query 1 — mensal (mesmas colunas da v3/v4)."""
    return _with(cte_pedidos_validos(cols), cte_base(dedup), ctes_extremos(), """limiar AS (
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
ORDER BY m.ds""")


def sql_diaria(cols=None, dedup=False):
    """Query 2 — diária (dias sem venda = 0), até ontem. Inclui separação base × extremos."""
    return _with(cte_pedidos_validos(cols), cte_base(dedup), ctes_extremos(), """dias AS (
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
ORDER BY dias.ds""")


def sql_divisao(cols=None):
    """Query 4 — origem 1P/3P/SEM_INFO por mês."""
    return _with(cte_pedidos_validos(cols), cte_base()) + """
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
ORDER BY 1, 2"""


def sql_impacto_limpeza():
    """Query 7 — impacto da limpeza por mês."""
    return f"""WITH {cte_pedidos_validos()},
todos_b2c AS (SELECT DISTINCT id AS id_pedido FROM `{T_PEDIDOS}`),
ped AS (
  SELECT id_pedido, MIN(pedido_date) AS pedido_date, SUM(preco_total) AS valor
  FROM `{T_ITENS}` WHERE pedido_date IS NOT NULL GROUP BY id_pedido
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
FROM classif GROUP BY ds, situacao ORDER BY ds, situacao"""


def sql_canal(cols=None):
    """Query 9 — com × sem nutricionista (hipótese: sem = público aberto)."""
    return _with(cte_pedidos_validos(cols), cte_base(), ctes_extremos()) + """
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
ORDER BY ds, canal"""


def sql_maiores_pedidos(dias=60, limite=50):
    """Query 5 — maiores pedidos recentes (diagnóstico)."""
    return _with(cte_pedidos_validos(), cte_base()) + f"""
SELECT id_pedido, MIN(pedido_date) AS pedido_date, COUNT(*) AS linhas, SUM(unidades) AS unidades,
       ROUND(SUM(preco_total), 2) AS valor_pedido,
       ROUND(SAFE_DIVIDE(SUM(preco_total), SUM(unidades)), 2) AS valor_por_unidade,
       STRING_AGG(DISTINCT COALESCE(origem_produto, 'SEM_INFO')) AS origem
FROM base
WHERE pedido_date >= DATE_SUB(CURRENT_DATE('America/Sao_Paulo'), INTERVAL {dias} DAY)
GROUP BY id_pedido ORDER BY valor_pedido DESC LIMIT {limite}"""


def sql_cubo_semanal(cols: Colunas, dedup=False, tipo='completo'):
    """Query 10 — CUBO SEMANAL, base de todas as granularidades.
    tipo='completo': semana × produto × marca/grupo/subgrupo × canal × origem × UF/região × extremo (para conferência/SQL)
    tipo='produto' : semana × produto × marca/grupo/subgrupo × canal × extremo   (níveis de produto e canibalização)
    tipo='geo'     : semana × canal × origem × UF/região × extremo               (níveis de canal, origem, região, UF)
    Os notebooks usam 'produto' + 'geo': juntos são muito menores que o completo (que cruza produto × UF e pode
    ter milhões de linhas e estourar a memória do Colab). Faturamento e unidades somam certo em qualquer corte;
    'pedidos' não (um pedido tem vários produtos)."""
    uf = _expr_uf(cols)
    campos = {
        'semana': "DATE_TRUNC(b.pedido_date, WEEK(MONDAY))",
        'id_produto': "CAST(b.id_produto AS STRING)",
        'produto_nome': "COALESCE(pr.produto_nome, CONCAT('produto ', CAST(b.id_produto AS STRING)))",
        'marca': "COALESCE(pr.marca, 'SEM_MARCA')",
        'grupo': "COALESCE(pr.grupo, 'SEM_GRUPO')",
        'subgrupo': "COALESCE(pr.subgrupo, 'SEM_SUBGRUPO')",
        'canal': "IF(b.canal_tem_nutri, 'COM_NUTRICIONISTA', 'SEM_NUTRICIONISTA')",
        'origem': "COALESCE(b.origem_produto, 'SEM_INFO')",
        'uf': f"COALESCE({uf}, 'SEM_INFO')",
        'regiao': _expr_regiao(uf),
        'eh_extremo': "COALESCE(e.eh_extremo, FALSE)",
    }
    dims = {'completo': list(campos),
            'produto': ['semana', 'id_produto', 'produto_nome', 'marca', 'grupo', 'subgrupo', 'canal', 'eh_extremo'],
            'geo': ['semana', 'canal', 'origem', 'uf', 'regiao', 'eh_extremo']}[tipo]
    sel = ',\n  '.join(f'{campos[d]} AS {d}' for d in dims)
    join_prod = "\nLEFT JOIN produtos pr ON pr.id_produto = CAST(b.id_produto AS STRING)" if tipo != 'geo' else ''
    ctes = [cte_pedidos_validos(cols), cte_base(dedup), ctes_extremos()] + ([cte_produtos(cols)] if tipo != 'geo' else [])
    return _with(*ctes) + f"""
SELECT
  {sel},
  COUNT(DISTINCT b.id_pedido)                                       AS pedidos,
  SUM(b.unidades)                                                   AS unidades,
  ROUND(SUM(b.preco_total), 2)                                      AS faturamento,
  ROUND(SUM(COALESCE(b.preco_cheio, b.preco_unitario) * b.unidades), 2) AS faturamento_preco_cheio
FROM base b
LEFT JOIN extremos e USING (id_pedido){join_prod}
WHERE b.pedido_date < CURRENT_DATE('America/Sao_Paulo')
GROUP BY {', '.join(str(k + 1) for k in range(len(dims)))}"""


class Cubos:
    """Cubo de produto + cubo geográfico, baixados uma vez. agregar() escolhe sozinho o cubo que tem as colunas
    pedidas (o menor possível) e faz o groupby localmente, sem nova consulta ao BigQuery."""

    def __init__(self, fonte, cols, dedup=False, geo=True, produto=True):
        self.c = {}
        if geo:
            self.c['geo'] = preparar_cubo(fonte.df(sql_cubo_semanal(cols, dedup, 'geo'), 'cubo_geo', datas=('semana',)))
        if produto:
            self.c['produto'] = preparar_cubo(fonte.df(sql_cubo_semanal(cols, dedup, 'produto'), 'cubo_produto', datas=('semana',)))
        for k, d in self.c.items():
            linhas = f'{len(d):,}'.replace(',', '.')
            print(f'cubo {k}: {linhas} linhas, {d.memory_usage(deep=True).sum() / 1e6:.1f} MB em memória')

    @property
    def columns(self):
        return sorted(set().union(*[d.columns for d in self.c.values()]))

    def agregar(self, dims, metricas, excluir_extremos=True, com_semana=True):
        chaves = (['semana'] if com_semana else []) + list(dims)
        for nome in ('geo', 'produto'):   # o geo é menor: tenta primeiro
            d = self.c.get(nome)
            if d is not None and set(chaves) <= set(d.columns):
                if excluir_extremos:
                    d = d[~d['eh_extremo']]
                g = d.groupby(chaves, as_index=False, observed=True)[list(metricas)].sum()
                for c in dims:
                    if isinstance(g[c].dtype, pd.CategoricalDtype):
                        g[c] = g[c].astype(str)
                return g
        raise KeyError(f'Nenhum cubo tem as colunas {chaves}')


def sql_cesta_pares(cols: Colunas, dias=365, top_produtos=150, dedup=False):
    """Query 11 — CESTA: com que frequência dois produtos saem no MESMO pedido (sem extremos).
    Base do 'lift' (complementares × substitutos) no estudo de canibalização."""
    return _with(cte_pedidos_validos(cols), cte_base(dedup), ctes_extremos()) + f"""
, janela AS (
  SELECT DISTINCT b.id_pedido, CAST(b.id_produto AS STRING) AS id_produto
  FROM base b LEFT JOIN extremos e USING (id_pedido)
  WHERE b.pedido_date >= DATE_SUB(CURRENT_DATE('America/Sao_Paulo'), INTERVAL {dias} DAY)
    AND NOT COALESCE(e.eh_extremo, FALSE)
),
top AS (
  SELECT CAST(id_produto AS STRING) AS id_produto
  FROM base WHERE pedido_date >= DATE_SUB(CURRENT_DATE('America/Sao_Paulo'), INTERVAL {dias} DAY)
  GROUP BY 1 ORDER BY SUM(preco_total) DESC LIMIT {top_produtos}
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
GROUP BY produto_a, produto_b, ia.n, ib.n"""


def sql_produtos(cols: Colunas):
    """Dimensão de produtos padronizada (para conferência)."""
    return _with(cte_produtos(cols)) + "\nSELECT * FROM produtos"


# ============================================================================
# UTILITÁRIOS
# ============================================================================
def atualizar_saida(df: pd.DataFrame, nome: str, pasta='saida', chave='data_base', substituir_tudo=False) -> pd.DataFrame:
    """Grava saida/<nome>.csv MANTENDO as rodadas anteriores: lê o arquivo que já existe, tira as linhas da mesma
    `data_base` desta rodada (rodar de novo no mesmo dia substitui, não duplica) e acrescenta as linhas novas.
    substituir_tudo=True: grava só a versão nova (para tabelas que já são a foto completa, ex.: realizado).
    Devolve a tabela completa (usada também para gravar no BigQuery)."""
    os.makedirs(pasta, exist_ok=True)
    arq = os.path.join(pasta, f'{nome}.csv')
    novo = df.copy()
    msg = 'substituído pela versão desta rodada'
    if not substituir_tudo and chave in novo.columns and os.path.exists(arq):
        try:
            antigo = pd.read_csv(arq)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            antigo = pd.DataFrame()
        if chave in antigo.columns:
            k_novo = set(pd.to_datetime(novo[chave], errors='coerce').dt.strftime('%Y-%m-%d'))
            k_ant = pd.to_datetime(antigo[chave], errors='coerce').dt.strftime('%Y-%m-%d')
            manter = ~k_ant.isin(k_novo)
            substituidas = antigo[~manter].shape[0]
            msg = (f'{k_ant[manter].nunique()} rodada(s) anterior(es) mantida(s)'
                   + (f'; {substituidas} linha(s) da mesma data substituída(s)' if substituidas else ''))
            datas_novas = [c for c in novo.columns if pd.api.types.is_datetime64_any_dtype(novo[c])]
            novo = pd.concat([antigo[manter], novo], ignore_index=True)
            for c in datas_novas:   # o CSV antigo traz datas como texto: padroniza a coluna inteira
                novo[c] = pd.to_datetime(novo[c], errors='coerce')
    novo.round(4).to_csv(arq, index=False, date_format='%Y-%m-%d')
    print(f'  {nome}.csv: {len(novo)} linhas — {msg}')
    return novo


def preparar_cubo(cubo: pd.DataFrame) -> pd.DataFrame:
    cubo = cubo.copy()
    cubo['semana'] = pd.to_datetime(cubo['semana'])
    cubo['eh_extremo'] = cubo['eh_extremo'].astype(str).str.lower().isin(['true', '1'])
    for c in ('pedidos', 'unidades', 'faturamento', 'faturamento_preco_cheio'):
        cubo[c] = pd.to_numeric(cubo[c], errors='coerce').fillna(0).astype('float32' if c != 'faturamento' else 'float64')
    for c in cubo.columns:   # textos repetidos -> category: usa uma fração da memória
        if c not in ('semana', 'eh_extremo', 'pedidos', 'unidades', 'faturamento', 'faturamento_preco_cheio'):
            cubo[c] = cubo[c].astype(str).astype('category')
    return cubo


def ultima_semana_completa(ultimo_dia: pd.Timestamp) -> pd.Timestamp:
    """Segunda-feira da última semana inteira (seg–dom) até ultimo_dia."""
    seg = ultimo_dia.normalize() - pd.Timedelta(days=ultimo_dia.dayofweek)
    return seg if ultimo_dia.dayofweek == 6 else seg - pd.Timedelta(days=7)
