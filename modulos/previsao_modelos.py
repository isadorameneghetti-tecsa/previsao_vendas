"""
previsao_modelos.py — catálogo de modelos de previsão de demanda + benchmark ("caminho feliz").

Ideia central: começar pelo modelo MAIS SIMPLES e só subir de complexidade quando o ganho de
precisão compensa. Cada modelo tem um nível de complexidade; a escolha final é o modelo mais simples
cujo erro fica a no máximo `tolerancia` (ex.: 5%) do melhor erro observado no backtest.

Escada de complexidade (0 = mais simples):
  0  Ingênuo, Média móvel, Sazonal ingênuo ("repete o ano passado")
  1  Sazonal × crescimento ("ano passado × ritmo recente")
  2  ETS / Holt-Winters, ARIMA
  3  SARIMA(X) com calendário, Prophet, ARIMA_PLUS (BigQuery ML, via bigframes)
  4  ML de calendário (GradientBoosting + XGBoost), TimesFM (BigQuery, opcional)
  5  Híbrido Prophet + XGBoost nos resíduos
  6  Combinação dos 3 melhores (pesos pelo erro), Top-down (pai × participação)

Para acrescentar um modelo novo: escreva fn(y, futuro, ctx, params) -> np.ndarray e registre
um Modelo(...) em catalogo_padrao(). Nada mais precisa mudar.

Projeto-agnóstico: serve para qualquer série de demanda diária ('D') ou semanal ('W').
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
for _lg in ('cmdstanpy', 'prophet', 'py.warnings'):
    logging.getLogger(_lg).setLevel(logging.ERROR)
    logging.getLogger(_lg).propagate = False


# ============================================================================
# CALENDÁRIO
# ============================================================================
def black_friday(ano):
    """Sexta-feira seguinte à 4ª quinta-feira de novembro."""
    nov = pd.date_range(f'{ano}-11-01', f'{ano}-11-30')
    return nov[nov.dayofweek == 3][3] + pd.Timedelta(days=1)


def feriados_br(anos=range(2023, 2031)):
    import holidays
    return holidays.Brazil(years=list(anos))


def _eh_feriado(datas, feriados):
    return np.array([d.date() in feriados for d in datas], dtype=int)


def features_calendario(idx: pd.DatetimeIndex, freq: str, feriados, t0='2024-01-01', tendencia=True):
    """Variáveis de calendário. D = diário (mesmas da v4); W = semanal (semana começando na segunda)."""
    t0 = pd.Timestamp(t0)
    X = pd.DataFrame(index=idx)
    bfs = [black_friday(a) for a in range(idx.min().year - 1, idx.max().year + 2)]
    if freq == 'D':
        if tendencia:
            X['t'] = (idx - t0).days
        for k in range(7):
            X[f'dow{k}'] = (idx.dayofweek == k).astype(int)
        X['dia'] = idx.day
        X['inicio_mes'] = (idx.day <= 5).astype(int)
        X['fim_mes'] = (idx.day >= idx.days_in_month - 2).astype(int)
        X['doy_sin'] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
        X['doy_cos'] = np.cos(2 * np.pi * idx.dayofyear / 365.25)
        X['feriado'] = _eh_feriado(idx, feriados)
        X['vespera'] = _eh_feriado(idx + pd.Timedelta(days=1), feriados)
        X['pos_feriado'] = _eh_feriado(idx - pd.Timedelta(days=1), feriados)
        X['black_friday'] = [int(any(abs((d - b).days) <= 3 for b in bfs)) for d in idx]
        X['fim_de_ano'] = (((idx.month == 12) & (idx.day >= 20)) | ((idx.month == 1) & (idx.day <= 2))).astype(int)
    else:
        if tendencia:
            X['t'] = (idx - t0).days / 7
        doy = idx.dayofyear + 3  # meio da semana
        for k in (1, 2):
            X[f'ano_sin{k}'] = np.sin(2 * np.pi * k * doy / 365.25)
            X[f'ano_cos{k}'] = np.cos(2 * np.pi * k * doy / 365.25)
        X['mes'] = idx.month
        dias = [pd.date_range(d, periods=7) for d in idx]
        X['n_feriados'] = [sum(x.date() in feriados for x in s) for s in dias]
        X['black_friday'] = [int(any(b in s for b in bfs)) for s in dias]
        X['fim_de_ano'] = [int(any((x.month == 12 and x.day >= 20) or (x.month == 1 and x.day <= 2) for x in s)) for s in dias]
        X['dias_inicio_mes'] = [sum(x.day <= 5 for x in s) for s in dias]
    return X


def _fourier(idx, freq, K):
    doy = idx.dayofyear + (3 if freq == 'W' else 0)
    cols = {}
    for k in range(1, K + 1):
        cols[f's{k}'] = np.sin(2 * np.pi * k * doy / 365.25)
        cols[f'c{k}'] = np.cos(2 * np.pi * k * doy / 365.25)
    return pd.DataFrame(cols, index=idx)


def _exog(idx, freq, ctx, K):
    X = _fourier(idx, freq, K)
    cal = features_calendario(idx, freq, ctx['feriados'], tendencia=False)
    extra = ['feriado', 'black_friday', 'fim_de_ano'] if freq == 'D' else ['n_feriados', 'black_friday', 'fim_de_ano']
    return pd.concat([X, cal[extra]], axis=1).astype(float)


def _passo(freq):
    return pd.Timedelta(days=1 if freq == 'D' else 7)


# ============================================================================
# CATÁLOGO
# ============================================================================
@dataclass
class Modelo:
    nome: str
    complexidade: int
    familia: str
    explicacao: str                        # frase simples, para o negócio
    params: dict = field(default_factory=dict)
    fn: Callable | None = None             # fn(y, futuro, ctx, params) -> array
    fn_lote: Callable | None = None        # fn_lote(tarefas, ctx, params) -> {chave: array}
    ativo: bool = True

    @property
    def lote(self):
        return self.fn_lote is not None


def _l(y):
    return np.log1p(np.clip(np.asarray(y, dtype=float), 0, None))


def _e(v):
    return np.clip(np.expm1(np.asarray(v, dtype=float)), 0, None)


# ---------------------------------------------------------------- nível 0-1
def m_ingenuo(y, futuro, ctx, p):
    return np.repeat(y.iloc[-p.get('janela', 1):].mean(), len(futuro))


def m_media_movel(y, futuro, ctx, p):
    return np.repeat(y.iloc[-p['janela']:].mean(), len(futuro))


def m_sazonal_ingenuo(y, futuro, ctx, p):
    lag = p['lag'] * _passo(ctx['freq'])
    reserva = m_ingenuo(y, futuro, ctx, {'janela': p.get('janela_reserva', 1)})
    v = np.array([y.get(d - lag, np.nan) for d in futuro], dtype=float)
    return np.where(np.isnan(v), reserva, v)


def m_sazonal_crescimento(y, futuro, ctx, p):
    base = m_sazonal_ingenuo(y, futuro, ctx, p)
    k, lag = p['janela_crescimento'], p['lag']
    if len(y) < lag + k:
        return base
    atual, ano_ant = y.iloc[-k:].sum(), y.iloc[-k - lag:-lag].sum()
    fator = np.clip(atual / ano_ant, *p.get('limites', (0.5, 2.0))) if ano_ant > 0 else 1.0
    return base * fator


# ---------------------------------------------------------------- nível 2
def m_ets(y, futuro, ctx, p):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    sazonal = p.get('periodo') if p.get('sazonal') and len(y) >= 2 * p.get('periodo', 1) + 2 else None
    m = ExponentialSmoothing(_l(y), trend='add', damped_trend=True,
                             seasonal='add' if sazonal else None, seasonal_periods=sazonal,
                             initialization_method='estimated').fit()
    return _e(m.forecast(len(futuro)))


def m_arima(y, futuro, ctx, p):
    from statsmodels.tsa.arima.model import ARIMA
    melhor, aic = None, np.inf
    for ordem in p['grade']:
        try:
            r = ARIMA(_l(y), order=tuple(ordem), trend='n' if ordem[1] > 0 else 'c').fit()
            if r.aic < aic:
                melhor, aic = r, r.aic
        except Exception:
            continue
    if melhor is None:
        raise RuntimeError('nenhuma ordem ARIMA convergiu')
    ctx.setdefault('ordens_arima', []).append(str(melhor.model.order))
    return _e(melhor.forecast(len(futuro)))


# ---------------------------------------------------------------- nível 3
def m_sarimax(y, futuro, ctx, p):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    freq = ctx['freq']
    Xh, Xf = _exog(y.index, freq, ctx, p['fourier']), _exog(futuro, freq, ctx, p['fourier'])
    sazonal = tuple(p['sazonal']) if freq == 'D' else (0, 0, 0, 0)
    r = SARIMAX(_l(y), exog=Xh.values, order=tuple(p['ordem']), seasonal_order=sazonal,
                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=200)
    return _e(r.forecast(len(futuro), exog=Xf.values))


def _feriados_prophet(freq, anos):
    linhas = []
    for a in anos:
        bf = black_friday(a)
        if freq == 'D':
            linhas.append(dict(holiday='black_friday', ds=bf, lower_window=-3, upper_window=3))
            linhas.append(dict(holiday='fim_de_ano', ds=pd.Timestamp(f'{a}-12-20'), lower_window=0, upper_window=13))
        else:
            linhas.append(dict(holiday='black_friday', ds=bf - pd.Timedelta(days=bf.dayofweek), lower_window=0, upper_window=0))
            for d in (pd.Timestamp(f'{a}-12-25'), pd.Timestamp(f'{a + 1}-01-01')):
                linhas.append(dict(holiday='fim_de_ano', ds=d - pd.Timedelta(days=d.dayofweek), lower_window=0, upper_window=0))
    return pd.DataFrame(linhas)


def _silenciar_prophet():
    for nome in ('cmdstanpy', 'prophet', 'prophet.plot'):
        lg = logging.getLogger(nome)
        lg.setLevel(logging.ERROR)
        lg.propagate = False


def _prophet_fit(y, ctx, p):
    from prophet import Prophet
    _silenciar_prophet()
    freq = ctx['freq']
    anos = range(y.index.min().year - 1, y.index.max().year + 3)
    # sazonalidade anual só com histórico suficiente: com menos de ~1,5 ano o Prophet extrapola a curva anual e explode
    dias_hist = (y.index.max() - y.index.min()).days
    anual = p['sazonalidade_anual'] if dias_hist >= p.get('min_dias_sazonalidade_anual', 547) else False
    m = Prophet(growth='linear', changepoint_prior_scale=p['changepoint_prior_scale'],
                seasonality_prior_scale=p['seasonality_prior_scale'],
                holidays_prior_scale=p.get('holidays_prior_scale', 10.0),
                yearly_seasonality=anual, weekly_seasonality=(freq == 'D'),
                daily_seasonality=False, holidays=_feriados_prophet(freq, anos),
                interval_width=0.8, uncertainty_samples=0)
    if freq == 'D' and p.get('feriados_pais'):
        m.add_country_holidays(country_name=p['feriados_pais'])
    m.fit(pd.DataFrame({'ds': y.index, 'y': _l(y)}))
    return m


def m_prophet(y, futuro, ctx, p):
    m = _prophet_fit(y, ctx, p)
    return _e(m.predict(pd.DataFrame({'ds': futuro}))['yhat'].values)


# ---------------------------------------------------------------- nível 4
def m_ml_calendario(y, futuro, ctx, p):
    from sklearn.ensemble import GradientBoostingRegressor
    from xgboost import XGBRegressor
    freq = ctx['freq']
    X = features_calendario(y.index, freq, ctx['feriados'], tendencia=p['tendencia'])
    Xf = features_calendario(futuro, freq, ctx['feriados'], tendencia=p['tendencia'])
    passo = 1 if freq == 'D' else 7
    peso = None
    if p.get('meia_vida_dias'):
        peso = 0.5 ** (((y.index.max() - y.index).days) / p['meia_vida_dias'])
    preds = []
    for mdl in (GradientBoostingRegressor(random_state=p['seed'], **p['gb']),
                XGBRegressor(random_state=p['seed'], **p['xgb'])):
        mdl.fit(X, _l(y), sample_weight=peso)
        preds.append(mdl.predict(Xf))
    return _e(np.mean(preds, axis=0))


# ---------------------------------------------------------------- nível 5
def m_hibrido_prophet_xgb(y, futuro, ctx, p):
    """Prophet explica tendência + sazonalidade; XGBoost aprende o que sobrou (resíduo) com o calendário."""
    from xgboost import XGBRegressor
    m = _prophet_fit(y, ctx, p['prophet'])
    hist = m.predict(pd.DataFrame({'ds': y.index}))['yhat'].values
    fut = m.predict(pd.DataFrame({'ds': futuro}))['yhat'].values
    resid = _l(y) - hist
    X = features_calendario(y.index, ctx['freq'], ctx['feriados'], tendencia=False)
    Xf = features_calendario(futuro, ctx['freq'], ctx['feriados'], tendencia=False)
    xgb = XGBRegressor(random_state=p['seed'], **p['xgb']).fit(X, resid)
    return _e(fut + p.get('peso_residuo', 1.0) * xgb.predict(Xf))


# ---------------------------------------------------------------- BigQuery (bigframes), em lote
def _tabela_lote(tarefas, log):
    linhas, hmax = [], 1
    for chave, y, fut in tarefas:
        v = _l(y) if log else np.asarray(y, dtype=float)
        linhas.append(pd.DataFrame({'id': str(chave), 'ds': pd.to_datetime(y.index), 'y': v}))
        hmax = max(hmax, len(fut))
    return pd.concat(linhas, ignore_index=True), hmax


def _ler_previsao_lote(pr, tarefas, log, col_id='id'):
    pr = pr.copy()
    ts = pd.to_datetime(pr['forecast_timestamp'])
    pr['forecast_timestamp'] = ts.dt.tz_localize(None) if getattr(ts.dt, 'tz', None) is not None else ts
    saida = {}
    for chave, y, fut in tarefas:
        x = pr[pr[col_id].astype(str) == str(chave)].sort_values('forecast_timestamp')
        v = x['forecast_value'].values[:len(fut)].astype(float)
        if len(v) < len(fut):
            v = np.concatenate([v, np.repeat(np.nan, len(fut) - len(v))])
        saida[chave] = _e(v) if log else np.clip(v, 0, None)
    return saida


def como_texto(df: pd.DataFrame) -> pd.DataFrame:
    """Converte todas as colunas para texto, inclusive colunas de lista/array (ex.: seasonal_periods do ARIMA_PLUS),
    que quebram no .astype(str) quando vêm do BigQuery em formato pyarrow."""
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        vals = []
        for v in df[c].astype(object).tolist():
            if isinstance(v, (list, tuple, np.ndarray)):
                vals.append(', '.join(str(x) for x in list(v)))
            elif v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)):
                vals.append('')
            else:
                vals.append(str(v))
        out[c] = vals
    return out


def lote_arima_plus(tarefas, ctx, p):
    """ARIMA_PLUS do BigQuery ML (auto-ARIMA + sazonalidades + feriados + limpeza de picos), via bigframes.
    Treina TODAS as séries (e todas as origens do backtest) num único modelo multi-série (id_col)."""
    import bigframes.pandas as bpd
    from bigframes.ml.forecasting import ARIMAPlus
    df, hmax = _tabela_lote(tarefas, p['log'])
    bdf = bpd.read_pandas(df)
    freq = {'D': 'DAILY', 'W': 'WEEKLY'}[ctx['freq']]
    mdl = ARIMAPlus(horizon=max(hmax, 2), data_frequency=freq,
                    holiday_region=p['holiday_region'] if ctx['freq'] == 'D' else None,
                    auto_arima_max_order=p['auto_arima_max_order'], include_drift=p['include_drift'],
                    clean_spikes_and_dips=p['clean_spikes_and_dips'], adjust_step_changes=p['adjust_step_changes'],
                    decompose_time_series=True)
    mdl.fit(bdf[['ds']], bdf[['y']], id_col=bdf[['id']])
    pr = mdl.predict(horizon=hmax, confidence_level=p['nivel_confianca']).to_pandas()
    try:  # hiperparâmetros escolhidos pelo auto-ARIMA para cada série (p, d, q, drift, sazonalidades...)
        ctx.setdefault('diagnostico', {}).setdefault('ARIMA_PLUS', []).append(como_texto(mdl.summary().to_pandas()))
    except Exception as ex:
        print('ARIMA_PLUS: não foi possível ler summary():', ex)
    return _ler_previsao_lote(pr, tarefas, p['log'])


def lote_timesfm(tarefas, ctx, p):
    """TimesFM (modelo fundacional pré-treinado do Google) via AI.FORECAST no BigQuery. Opcional."""
    import bigframes.bigquery as bbq
    import bigframes.pandas as bpd
    df, hmax = _tabela_lote(tarefas, p['log'])
    pr = bbq.ai.forecast(bpd.read_pandas(df), data_col='y', timestamp_col='ds', id_cols=['id'],
                         horizon=hmax, confidence_level=p['nivel_confianca']).to_pandas()
    return _ler_previsao_lote(pr, tarefas, p['log'])


# ============================================================================
# CATÁLOGO PADRÃO (todos os hiperparâmetros ficam aqui, visíveis)
# ============================================================================
def catalogo_padrao(freq: str, seed=42, usar_bigquery=True, usar_timesfm=False):
    D = freq == 'D'
    lag = 364 if D else 52       # 364 dias = mesmo dia da semana do ano anterior
    mods = [
        Modelo('Ingênuo', 0, 'Base',
               'Repete o último período (última semana).',
               {'janela': 7 if D else 1}, fn=m_ingenuo),
        Modelo('Média móvel', 0, 'Base',
               'Média dos últimos períodos.',
               {'janela': 28 if D else 8}, fn=m_media_movel),
        Modelo('Sazonal ingênuo', 0, 'Base',
               'Repete o mesmo período do ano anterior.',
               {'lag': lag, 'janela_reserva': 7 if D else 1, 'historico_completo': True}, fn=m_sazonal_ingenuo),
        Modelo('Sazonal × crescimento', 1, 'Base',
               'Ano anterior × ritmo de crescimento das últimas semanas.',
               {'lag': lag, 'janela_reserva': 7 if D else 1, 'janela_crescimento': 56 if D else 8, 'limites': (0.5, 2.0),
                'historico_completo': True},
               fn=m_sazonal_crescimento),
        Modelo('ETS (Holt-Winters)', 2, 'Estatístico',
               'Suavização exponencial: nível + tendência amortecida' + (' + dia da semana.' if D else '.'),
               {'sazonal': D, 'periodo': 7}, fn=m_ets),
        Modelo('ARIMA', 2, 'Estatístico',
               'Autorregressivo clássico; ordem (p,d,q) escolhida pelo menor AIC.',
               {'grade': [(1, 1, 1), (0, 1, 1), (1, 1, 0), (2, 1, 2), (1, 0, 1), (2, 0, 1)]}, fn=m_arima),
        Modelo('SARIMAX calendário', 3, 'Estatístico',
               'ARIMA com sazonalidade' + (' semanal' if D else '') + ' anual (Fourier), feriados e Black Friday.',
               {'ordem': (1, 1, 1), 'sazonal': (1, 0, 1, 7), 'fourier': 3}, fn=m_sarimax),
        Modelo('Prophet', 3, 'Estatístico',
               'Prophet (Meta): tendência com mudanças + sazonalidades + feriados BR.',
               {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 5.0, 'holidays_prior_scale': 10.0,
                'sazonalidade_anual': 10 if D else 6, 'min_dias_sazonalidade_anual': 547, 'feriados_pais': 'BR'}, fn=m_prophet),
        Modelo('ML calendário (GB+XGB)', 4, 'ML',
               'Média de GradientBoosting e XGBoost com variáveis de calendário (modelo da v4).',
               {'tendencia': True, 'meia_vida_dias': 180, 'seed': seed,
                'gb': dict(n_estimators=300, max_depth=3, learning_rate=0.03),
                'xgb': dict(n_estimators=400, max_depth=3, learning_rate=0.03)}, fn=m_ml_calendario),
        Modelo('Híbrido Prophet+XGB', 5, 'Híbrido',
               'Prophet faz a base; XGBoost corrige o que o Prophet não captou (resíduo).',
               {'seed': seed, 'peso_residuo': 1.0,
                'prophet': {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 5.0,
                            'sazonalidade_anual': 10 if D else 6, 'min_dias_sazonalidade_anual': 547, 'feriados_pais': 'BR'},
                'xgb': dict(n_estimators=200, max_depth=2, learning_rate=0.03, subsample=0.8)},
               fn=m_hibrido_prophet_xgb),
    ]
    if usar_bigquery:
        mods.append(Modelo('ARIMA_PLUS (BigQuery)', 3, 'Estatístico',
                           'Auto-ARIMA do BigQuery ML: escolhe a ordem sozinho, trata picos, feriados e mudanças de patamar.',
                           {'log': True, 'holiday_region': 'BR', 'auto_arima_max_order': 5, 'include_drift': False,
                            'clean_spikes_and_dips': True, 'adjust_step_changes': True, 'nivel_confianca': 0.8},
                           fn_lote=lote_arima_plus))
    if usar_bigquery and usar_timesfm:
        mods.append(Modelo('TimesFM (BigQuery)', 4, 'Fundacional',
                           'Modelo pré-treinado em bilhões de séries (Google), sem ajuste local.',
                           {'log': True, 'nivel_confianca': 0.8}, fn_lote=lote_timesfm))
    return mods


def tabela_hiperparametros(modelos) -> pd.DataFrame:
    linhas = []
    for m in modelos:
        def achatar(d, pref=''):
            for k, v in d.items():
                if isinstance(v, dict):
                    yield from achatar(v, f'{pref}{k}.')
                else:
                    yield f'{pref}{k}', v
        itens = list(achatar(m.params)) or [('—', '—')]
        for k, v in itens:
            linhas.append(dict(modelo=m.nome, complexidade=m.complexidade, familia=m.familia,
                               parametro=k, valor=str(v), explicacao=m.explicacao))
    return pd.DataFrame(linhas)


# ============================================================================
# BACKTEST
# ============================================================================
def montar_tarefas(series: dict, origens, horizonte: pd.DateOffset, fim=None, min_hist=30):
    tarefas = []
    for nome, s in series.items():
        for o in origens:
            y = s[s.index <= o]
            lim = o + horizonte
            if fim is not None:
                lim = min(lim, fim)
            fut = s.index[(s.index > o) & (s.index <= lim)]
            if len(y) >= min_hist and len(fut) and y.sum() > 0:
                tarefas.append(((nome, o), y, fut, s.loc[fut].values))
    return tarefas


def _janela_treino(y, m, ctx):
    """Modelos sazonais simples olham o histórico todo (precisam do ano anterior);
    os demais treinam a partir de ctx['inicio_treino'] (descarta a fase de implantação)."""
    if m.params.get('historico_completo') or not ctx.get('inicio_treino'):
        return y
    return y[y.index >= pd.Timestamp(ctx['inicio_treino'])]


def rodar_modelos(tarefas, modelos, ctx, verbose=True):
    """tarefas: [(chave, y_treino, datas_futuras, real|None)]. Devolve formato longo."""
    import time
    linhas, falhas = [], []
    for m in [m for m in modelos if m.ativo]:
        t0 = time.time()
        prevs = {}
        if m.lote:
            try:
                prevs = m.fn_lote([(i, _janela_treino(t[1], m, ctx), t[2]) for i, t in enumerate(tarefas)], ctx, m.params)
                prevs = {tarefas[i][0]: v for i, v in prevs.items()}
            except Exception as ex:
                falhas.append((m.nome, 'todas', repr(ex)[:200]))
        else:
            for chave, y, fut, _ in tarefas:
                y = _janela_treino(y, m, ctx)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        prevs[chave] = np.clip(np.asarray(m.fn(y, fut, ctx, m.params), dtype=float), 0, None)
                except Exception as ex:
                    falhas.append((m.nome, chave, repr(ex)[:200]))
        for chave, y, fut, real in tarefas:
            if chave not in prevs:
                continue
            serie, origem = chave if isinstance(chave, tuple) else (chave, y.index.max())
            linhas.append(pd.DataFrame({'serie': serie, 'origem': origem, 'ds': fut, 'modelo': m.nome,
                                        'previsto': prevs[chave],
                                        'real': real if real is not None else np.nan}))
        if verbose:
            print(f'  {m.nome:<26} {time.time() - t0:6.1f}s  ({len(prevs)} previsões)')
    if falhas and verbose:
        print(f'  {len(falhas)} falha(s) — ver ctx["falhas"]')
    ctx.setdefault('falhas', []).extend(falhas)
    return pd.concat(linhas, ignore_index=True) if linhas else pd.DataFrame()


def backtest(series, modelos, origens, horizonte, ctx, fim=None, min_hist=30, verbose=True):
    tarefas = montar_tarefas(series, origens, horizonte, fim, min_hist)
    if verbose:
        print(f'Backtest: {len(series)} série(s) × {len(origens)} origem(ns) = {len(tarefas)} tarefa(s)')
    bt = rodar_modelos(tarefas, modelos, ctx, verbose)
    if not bt.empty:
        bt['horizonte_mes'] = ((bt['ds'].dt.year - bt['origem'].dt.year) * 12 + bt['ds'].dt.month - bt['origem'].dt.month)
    return bt


def adicionar_combinacao(bt, top=3, nome='Combinação top-3', excluir=()):
    """Média ponderada (1/erro) dos `top` melhores modelos de cada série. Os pesos de uma origem são
    calculados SÓ com as outras origens (sem 'colar' do próprio teste)."""
    base = bt[~bt['modelo'].isin(list(excluir) + [nome])]
    origens = sorted(base['origem'].unique())
    if len(origens) < 2:
        return bt
    erro = (base.assign(ae=(base.previsto - base.real).abs())
                .groupby(['serie', 'origem', 'modelo']).agg(ae=('ae', 'sum'), r=('real', 'sum')).reset_index())
    novas = []
    for (serie, o), g in base.groupby(['serie', 'origem']):
        outras = erro[(erro.serie == serie) & (erro.origem != o)].groupby('modelo')[['ae', 'r']].sum()
        if outras.empty:
            continue
        w = 1 / (outras['ae'] / outras['r'].clip(lower=1e-9)).clip(lower=1e-6)
        w = w.sort_values(ascending=False).head(top)
        w = w / w.sum()
        piv = g.pivot_table(index='ds', columns='modelo', values='previsto')
        cols = [c for c in w.index if c in piv.columns]
        if not cols:
            continue
        prev = (piv[cols] * (w[cols] / w[cols].sum())).sum(axis=1)
        real = g.drop_duplicates('ds').set_index('ds')['real']
        novas.append(pd.DataFrame({'serie': serie, 'origem': o, 'ds': prev.index, 'modelo': nome,
                                   'previsto': prev.values, 'real': real.loc[prev.index].values,
                                   'horizonte_mes': g.drop_duplicates('ds').set_index('ds').loc[prev.index, 'horizonte_mes'].values}))
    return pd.concat([bt] + novas, ignore_index=True) if novas else bt


def adicionar_top_down(bt_filho, bt_pai, series_filho, series_pai, pai_de: dict, modelo_pai: dict,
                       semanas=13, nome='Top-down (pai × part.)'):
    """Previsão do filho = previsão do pai (modelo escolhido no nível pai) × participação recente do filho
    no pai (últimas `semanas`). Útil quando há canibalização: o total do grupo é mais estável e o que
    oscila é a divisão entre as marcas/produtos."""
    novas = []
    for filho, pai in pai_de.items():
        if filho not in series_filho or pai not in series_pai or pai not in modelo_pai:
            continue
        s_f, s_p = series_filho[filho], series_pai[pai]
        bp = bt_pai[(bt_pai.serie == pai) & (bt_pai.modelo == modelo_pai[pai])]
        for o, g in bp.groupby('origem'):
            share = participacao(s_f, s_p, o, semanas)
            gf = bt_filho[(bt_filho.serie == filho) & (bt_filho.origem == o)].drop_duplicates('ds').set_index('ds')
            g = g.set_index('ds')
            g = g[g.index.isin(gf.index)]
            if g.empty:
                continue
            novas.append(pd.DataFrame({'serie': filho, 'origem': o, 'ds': g.index, 'modelo': nome,
                                       'previsto': g['previsto'].values * share, 'real': gf.loc[g.index, 'real'].values,
                                       'horizonte_mes': gf.loc[g.index, 'horizonte_mes'].values}))
    return pd.concat([bt_filho] + novas, ignore_index=True) if novas else bt_filho


def participacao(s_filho, s_pai, ate, semanas=13):
    jan = (s_pai.index <= ate) & (s_pai.index > ate - pd.Timedelta(weeks=semanas))
    tot = s_pai[jan].sum()
    return float(s_filho.reindex(s_pai.index[jan]).fillna(0).sum() / tot) if tot > 0 else 0.0


# ============================================================================
# MÉTRICAS E ESCOLHA
# ============================================================================
def mensalizar(bt):
    """Soma previsto/real por mês calendário (meses da semana = mês da segunda-feira)."""
    b = bt.assign(mes=bt['ds'].dt.to_period('M').dt.to_timestamp())
    return (b.groupby(['serie', 'origem', 'modelo', 'mes'], as_index=False)
             .agg(previsto=('previsto', 'sum'), real=('real', 'sum'), horizonte_mes=('horizonte_mes', 'max')))


def metricas(b, por=('modelo',)):
    """WAPE = erro absoluto total ÷ volume real total (%). Viés = quanto previu acima (+) ou abaixo (−)."""
    def f(g):
        r, p = g['real'].sum(), g['previsto'].sum()
        ape = (g['previsto'] - g['real']).abs() / g['real'].where(g['real'] > 0)
        return pd.Series({'WAPE %': (g['previsto'] - g['real']).abs().sum() / r * 100 if r else np.nan,
                          'MAPE %': ape.mean() * 100, 'Viés %': (p / r - 1) * 100 if r else np.nan,
                          'n': len(g)})
    return b.groupby(list(por)).apply(f).reset_index()


def escolher(tab, modelos, tolerancia=0.05, referencia='Sazonal ingênuo', col='WAPE %'):
    """Regra do caminho feliz: entre os modelos com erro até (1+tolerancia) × melhor erro,
    fica o MAIS SIMPLES. Devolve (nome_escolhido, tabela com ranking e justificativa)."""
    comp = {m.nome: m.complexidade for m in modelos}
    comp.setdefault('Combinação top-3', 6)
    comp.setdefault('Top-down (pai × part.)', 6)
    t = tab.dropna(subset=[col]).copy()
    t['complexidade'] = t['modelo'].map(comp).fillna(9).astype(int)
    melhor = t[col].min()
    t['elegível'] = t[col] <= melhor * (1 + tolerancia)
    t = t.sort_values(['elegível', 'complexidade', col], ascending=[False, True, True])
    esc = t.iloc[0]['modelo']
    ref = t.loc[t.modelo == referencia, col]
    t['ganho vs ' + referencia + ' (p.p.)'] = (ref.iloc[0] - t[col]) if len(ref) else np.nan
    t['escolhido'] = t['modelo'].eq(esc)
    t = t.sort_values(col).reset_index(drop=True)
    return esc, t


def justificar(t, esc, col='WAPE %', referencia='Sazonal ingênuo'):
    best = t.iloc[0]
    e = t[t.modelo == esc].iloc[0]
    ref = t[t.modelo == referencia]
    txt = f'Escolhido: {esc} (erro {e[col]:.1f}%).'
    if best['modelo'] != esc:
        txt += f' O melhor foi {best["modelo"]} ({best[col]:.1f}%), mas a diferença é pequena e {esc} é mais simples.'
    if len(ref) and referencia != esc:
        g = ref.iloc[0][col] - e[col]
        txt += (f' Erra {g:.1f} p.p. menos que "{referencia}".' if g > 0 else
                f' ATENÇÃO: não supera "{referencia}" — considere usar o modelo simples.')
    return txt


def faixa_empirica(bt_mensal, modelo, quantil=0.8):
    """Erro relativo |prev/real−1| no percentil `quantil`, por horizonte (meses à frente)."""
    b = bt_mensal[bt_mensal.modelo == modelo]
    return (b.assign(e=(b.previsto / b.real - 1).abs()).groupby('horizonte_mes')['e'].quantile(quantil))


def prever(series: dict, modelo: Modelo, futuro: dict, ctx) -> dict:
    """Previsão final: futuro = {serie: DatetimeIndex}. Devolve {serie: pd.Series}."""
    tarefas = [(k, s, futuro[k], None) for k, s in series.items() if k in futuro and len(futuro[k])]
    out = rodar_modelos(tarefas, [modelo], ctx, verbose=False)
    if out.empty:
        return {}
    return {k: g.set_index('ds')['previsto'] for k, g in out.groupby('serie')}


def prever_escolhido(escolhido, series, futuro, catalogo, ctx, bt=None, top=3,
                     prev_pai=None, series_pai=None, pai_de=None, semanas=13):
    """Previsão final usando o modelo escolhido no benchmark, inclusive os 'compostos':
    - 'Combinação top-3': roda os 3 melhores modelos de cada série (pelo backtest) e pondera por 1/erro;
    - 'Top-down': previsão do pai × participação recente."""
    cat = {m.nome: m for m in catalogo}
    if escolhido.startswith('Combinação'):
        e = (bt[~bt.modelo.isin([escolhido, 'Top-down (pai × part.)'])]
             .assign(ae=lambda x: (x.previsto - x.real).abs())
             .groupby(['serie', 'modelo'])[['ae', 'real']].sum())
        e['wape'] = e['ae'] / e['real'].clip(lower=1e-9)
        saida, pesos = {}, {}
        for serie in series:
            if serie not in e.index.get_level_values(0):
                continue
            w = (1 / e.loc[serie, 'wape'].clip(lower=1e-6)).sort_values(ascending=False).head(top)
            w = w[w.index.isin(cat)]
            w = w / w.sum()
            prevs = [prever({serie: series[serie]}, cat[mn], {serie: futuro[serie]}, ctx).get(serie) for mn in w.index]
            ok = [(p, w.iloc[i]) for i, p in enumerate(prevs) if p is not None]
            if ok:
                tot_w = sum(x[1] for x in ok)
                saida[serie] = sum(p * (wi / tot_w) for p, wi in ok)
                pesos[serie] = {mn: round(float(v), 3) for mn, v in w.items()}
        ctx.setdefault('pesos_combinacao', {}).update(pesos)
        return saida
    if escolhido.startswith('Top-down'):
        saida = {}
        for serie, pai in (pai_de or {}).items():
            if serie in series and pai in (prev_pai or {}):
                saida[serie] = prev_pai[pai] * participacao(series[serie], series_pai[pai], series[serie].index.max(), semanas)
        return saida
    return prever(series, cat[escolhido], futuro, ctx)
