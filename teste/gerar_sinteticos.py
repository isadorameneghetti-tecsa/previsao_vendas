"""Gera dados SINTÉTICOS (fictícios) no formato das consultas, para testar os notebooks em modo offline.
Nenhum dado real. Marcas e produtos inventados."""
import os, sys
import numpy as np, pandas as pd
_raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(_raiz, 'modulos'), _raiz]
import previsao_modelos as pm

rng = np.random.default_rng(7)
pasta = sys.argv[1] if len(sys.argv) > 1 else 'dados/cache_dados'
os.makedirs(pasta, exist_ok=True)
HOJE = pd.Timestamp('2026-10-06')
dias = pd.date_range('2024-03-01', HOJE - pd.Timedelta(days=1))
fer = pm.feriados_br()

# ---------- produtos
estrutura = {('Proteínas', 'Whey concentrado'): 8, ('Proteínas', 'Whey isolado'): 5, ('Proteínas', 'Proteína vegana'): 3,
             ('Creatina', 'Creatina'): 6, ('Vitaminas', 'Multivitamínico'): 5, ('Vitaminas', 'Vitamina D'): 4,
             ('Vitaminas', 'Ômega 3'): 4, ('Pré-treino', 'Pré-treino'): 4, ('Snacks', 'Barra de proteína'): 5}
marcas = [f'Marca {c}' for c in 'ABCDEFGH']
prods = []
pid = 1000
for (g, sg), n in estrutura.items():
    for k in range(n):
        preco = {'Proteínas': 150, 'Creatina': 90, 'Vitaminas': 60, 'Pré-treino': 110, 'Snacks': 12}[g] * rng.uniform(.7, 1.5)
        prods.append(dict(id_produto=pid, nome_produto=f'{sg} {marcas[k % 8]} {k}', marca=marcas[k % 8], categoria=g,
                          subcategoria=sg, preco=round(preco, 2), origem='1P' if k % 3 else '3P',
                          peso=rng.lognormal(0, .8)))
        pid += 1
P = pd.DataFrame(prods)

# ---------- pedidos diários
t = np.arange(len(dias))
nivel = np.where(dias < '2025-01-01', 5 + 25 * (t / 300).clip(0, 1), 30 + 0.02 * (dias - pd.Timestamp('2025-01-01')).days)
dow = np.array([1.15, 1.1, 1.05, 1.0, .95, .75, .8])[dias.dayofweek]
saz = 1 + .12 * np.sin(2 * np.pi * (dias.dayofyear - 60) / 365)
bf = np.array([any(abs((d - pm.black_friday(a)).days) <= 3 for a in (2024, 2025)) for d in dias]) * 1.2 + 1
natal = np.where(((dias.month == 12) & (dias.day >= 20)) | ((dias.month == 1) & (dias.day <= 2)), .6, 1)
feriado = np.array([.7 if d.date() in fer else 1 for d in dias])
lam = nivel * dow * saz * bf * natal * feriado
n_ped = rng.poisson(lam)

# promoções da Marca A no whey concentrado (para gerar canibalização)
promo = set()
for s in pd.date_range('2025-03-03', '2026-09-01', freq='9W-MON'):
    promo.update(pd.date_range(s, periods=14))

ufs = ['SP', 'RJ', 'MG', 'PR', 'RS', 'SC', 'BA', 'PE', 'GO', 'DF', 'CE', 'ES', 'AM']
uf_p = np.array([30, 12, 11, 7, 6, 5, 6, 4, 4, 4, 4, 3, 2], float); uf_p /= uf_p.sum()
linhas = []
idp = 1
w_base = P['peso'].values / P['peso'].sum()
wc = P.index[(P.subcategoria == 'Whey concentrado')]
idx_A = P.index[(P.subcategoria == 'Whey concentrado') & (P.marca == 'Marca A')]
for d, n in zip(dias, n_ped):
    w = w_base.copy()
    if d in promo:  # Marca A ganha participação dentro do subgrupo (o subgrupo cresce pouco)
        tot = w[wc].sum()
        w[idx_A] *= 4
        w[wc] *= (tot * 1.1) / w[wc].sum()
        w /= w.sum()
    p_nutri = .75 if d < pd.Timestamp('2026-05-01') else .5
    for _ in range(n):
        k = rng.choice([1, 2, 3], p=[.55, .3, .15])
        itens = rng.choice(len(P), size=k, replace=False, p=w)
        nutri, uf = rng.random() < p_nutri, rng.choice(ufs, p=uf_p)
        for i in itens:
            u = rng.choice([1, 2], p=[.8, .2])
            pc = P.preco[i]; pu = pc * (0.85 if (d in promo and i in idx_A) else 1)
            linhas.append((idp, d, P.id_produto[i], u, pu, pc, pu * u, P.origem[i], nutri, uf))
        idp += 1
# pedidos extremos depois da abertura
for d in pd.to_datetime(['2026-05-14', '2026-06-20', '2026-08-05', '2026-08-06', '2026-09-18']):
    for _ in range(3 if d.month == 8 else 1):
        i = rng.choice(wc)
        linhas.append((idp, d, P.id_produto[i], 60, P.preco[i] * .9, P.preco[i], P.preco[i] * .9 * 60, P.origem[i], False, 'SP'))
        idp += 1
I = pd.DataFrame(linhas, columns=['id_pedido', 'pedido_date', 'id_produto', 'unidades', 'preco_unitario', 'preco_cheio',
                                  'preco_total', 'origem_produto', 'tem_nutri', 'uf'])
ped = I.groupby('id_pedido').agg(pedido_date=('pedido_date', 'min'), valor=('preco_total', 'sum'),
                                 cheio=('preco_cheio', lambda s: 0), unid=('unidades', 'sum'), linhas=('id_produto', 'size'),
                                 tem_nutri=('tem_nutri', 'first'), uf=('uf', 'first'))
ped['cheio'] = (I.preco_cheio * I.unidades).groupby(I.id_pedido).sum()
ped['extremo'] = ped.valor > 5000
I = I.merge(ped[['extremo']], left_on='id_pedido', right_index=True)

# ---------- schema
sch = [('produtos', c, 'STRING') for c in ['id_produto', 'nome_produto', 'marca', 'categoria', 'subcategoria', 'preco', 'data_atualizacao']]
sch += [('itens', c, 'X') for c in ['id_pedido', 'id_produto', 'pedido_date', 'unidades', 'preco_total', 'origem_produto']]
sch += [('pedidos', c, 'X') for c in ['id', 'status', 'id_nutricionista', 'id_paciente', 'email', 'uf']]
pd.DataFrame(sch, columns=['tabela', 'column_name', 'data_type']).to_csv(f'{pasta}/schema.csv', index=False)

# ---------- mensal
ped['ds'] = ped.pedido_date.dt.to_period('M').dt.to_timestamp()
m = ped.groupby('ds').agg(y_total_pedidos=('valor', 'size'), faturamento_total=('valor', 'sum'), cheio=('cheio', 'sum'),
                          total_unidades_vendidas=('unid', 'sum'), ticket_mediano=('valor', 'median'),
                          ticket_p90=('valor', lambda s: s.quantile(.9)), maior_pedido=('valor', 'max'),
                          pedidos_extremos=('extremo', 'sum'), ultima_data=('pedido_date', 'max'),
                          dias_com_venda=('pedido_date', 'nunique'))
m['faturamento_extremos'] = ped[ped.extremo].groupby('ds')['valor'].sum().reindex(m.index).fillna(0)
p99 = ped.valor.quantile(.99)
m['pedidos_grandes'] = (ped.valor > p99).groupby(ped.ds).sum()
m['pct_fat_pedidos_grandes'] = ped.valor.where(ped.valor > p99, 0).groupby(ped.ds).sum() / m.faturamento_total * 100
m['ticket_medio'] = m.faturamento_total / m.y_total_pedidos
m['media_itens_por_pedido'] = m.total_unidades_vendidas / m.y_total_pedidos
m['preco_medio_unidade'] = m.faturamento_total / m.total_unidades_vendidas
m['desconto_medio_pct'] = (1 - m.faturamento_total / m.cheio) * 100
m['pedidos_grupos_identicos'] = 0
m['ticket_medio_sem_extremos'] = (m.faturamento_total - m.faturamento_extremos) / (m.y_total_pedidos - m.pedidos_extremos)
m['mes_completo'] = m.index < HOJE.to_period('M').to_timestamp()
m = m.drop(columns='cheio').round(2).reset_index()
m.to_csv(f'{pasta}/mensal.csv', index=False)

# ---------- diária
dd = ped.groupby('pedido_date').agg(pedidos=('valor', 'size'), faturamento=('valor', 'sum'), unidades=('unid', 'sum'),
                                    pedidos_extremos=('extremo', 'sum'))
dd['faturamento_extremos'] = ped[ped.extremo].groupby('pedido_date')['valor'].sum()
dd = dd.reindex(dias).fillna(0)
dd['ticket_medio'] = (dd.faturamento / dd.pedidos.replace(0, np.nan)).round(2)
dd.index.name = 'ds'
dd.reset_index().round(2).to_csv(f'{pasta}/diaria.csv', index=False)

# ---------- divisão, canal, impacto
I['ds'] = I.pedido_date.dt.to_period('M').dt.to_timestamp()
I.groupby(['ds', 'origem_produto']).agg(pedidos=('id_pedido', 'nunique'), faturamento=('preco_total', 'sum')) \
 .reset_index().rename(columns={'origem_produto': 'origem'}).to_csv(f'{pasta}/divisao.csv', index=False)
ped.assign(canal=np.where(ped.tem_nutri, 'COM_NUTRICIONISTA', 'SEM_NUTRICIONISTA')).groupby(['ds', 'canal']) \
   .agg(pedidos=('valor', 'size'), faturamento=('valor', 'sum'), ticket_medio=('valor', 'mean'),
        pedidos_extremos=('extremo', 'sum')).reset_index().round(2).to_csv(f'{pasta}/canal.csv', index=False)
imp = ped.groupby('ds').agg(pedidos=('valor', 'size'), faturamento=('valor', 'sum')).reset_index().assign(situacao='VALIDO')
imp2 = imp.assign(situacao='REMOVIDO_STATUS_OU_TESTE', pedidos=(imp.pedidos * .08).round(), faturamento=imp.faturamento * .07)
pd.concat([imp, imp2]).to_csv(f'{pasta}/impacto_limpeza.csv', index=False)

# ---------- cubo semanal
I = I.merge(P[['id_produto', 'nome_produto', 'marca', 'categoria', 'subcategoria']], on='id_produto')
I['semana'] = I.pedido_date - pd.to_timedelta(I.pedido_date.dt.dayofweek, unit='D')
I['canal'] = np.where(I.tem_nutri, 'COM_NUTRICIONISTA', 'SEM_NUTRICIONISTA')
reg = {u: r for r, us in {'Norte': ['AM'], 'Nordeste': ['BA', 'PE', 'CE'], 'Centro-Oeste': ['GO', 'DF'],
                          'Sudeste': ['SP', 'RJ', 'MG', 'ES'], 'Sul': ['PR', 'RS', 'SC']}.items() for u in us}
I['regiao'] = I.uf.map(reg)
I['fat_cheio'] = I.preco_cheio * I.unidades
cubo = I.groupby(['semana', 'id_produto', 'nome_produto', 'marca', 'categoria', 'subcategoria', 'canal', 'origem_produto',
                  'uf', 'regiao', 'extremo']).agg(pedidos=('id_pedido', 'nunique'), unidades=('unidades', 'sum'),
                                                  faturamento=('preco_total', 'sum'), faturamento_preco_cheio=('fat_cheio', 'sum')).reset_index()
cubo = cubo.rename(columns={'nome_produto': 'produto_nome', 'categoria': 'grupo', 'subcategoria': 'subgrupo',
                            'origem_produto': 'origem', 'extremo': 'eh_extremo'})
cubo.round(2).to_csv(f'{pasta}/cubo_semanal.csv', index=False)
m_ = ['pedidos', 'unidades', 'faturamento', 'faturamento_preco_cheio']
cubo.groupby(['semana', 'id_produto', 'produto_nome', 'marca', 'grupo', 'subgrupo', 'canal', 'eh_extremo'], as_index=False)[m_].sum() \
    .round(2).to_csv(f'{pasta}/cubo_produto.csv', index=False)
cubo.groupby(['semana', 'canal', 'origem', 'uf', 'regiao', 'eh_extremo'], as_index=False)[m_].sum().round(2).to_csv(f'{pasta}/cubo_geo.csv', index=False)

# ---------- cesta
j = I[(I.pedido_date >= HOJE - pd.Timedelta(days=365)) & ~I.extremo][['id_pedido', 'id_produto']].drop_duplicates()
tot = j.id_pedido.nunique()
ind = j.groupby('id_produto').size()
pr = j.merge(j, on='id_pedido')
pr = pr[pr.id_produto_x < pr.id_produto_y].groupby(['id_produto_x', 'id_produto_y']).size().rename('pedidos_juntos').reset_index()
pr.columns = ['produto_a', 'produto_b', 'pedidos_juntos']
pr['pedidos_a'] = pr.produto_a.map(ind); pr['pedidos_b'] = pr.produto_b.map(ind); pr['total_pedidos'] = tot
pr.to_csv(f'{pasta}/cesta_pares.csv', index=False)
P.rename(columns={'nome_produto': 'produto_nome', 'categoria': 'grupo', 'subcategoria': 'subgrupo'})[
    ['id_produto', 'produto_nome', 'marca', 'grupo', 'subgrupo']].to_csv(f'{pasta}/produtos.csv', index=False)
print('ok', len(I), 'itens', len(ped), 'pedidos', len(cubo), 'linhas cubo')
