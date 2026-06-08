# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 17:31:54 2026

@author: bsjhw
"""

"""
USD/KRW FX 리스크 분석 — 코어 라이브러리

    import coremodel_fxrisk as main
    df = main.load_fx_data()
    df = main.calc_return(df)
    df = main.calc_volatility(df)
    df, model = main.detect_regime_hmm(df)
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False


REGIME_COLORS = {0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c'}
REGIME_LABELS = {0: 'Stable', 1: 'Transition', 2: 'Crisis'}

# 헤지 정책 기초값 ; 정책으로 고정한 값
HEDGE_BASE = {'Stable': 0.30, 'Transition': 0.60, 'Crisis': 0.90}


def set_style():
    """
    그래프 전역 설정
    
    """
    plt.rcParams.update({
        'figure.dpi': 150,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'font.size': 10,
    })

# 데이터 로드

def load_fx_data(start='2000-01-01', cache='krw_data.csv'):
    """
    Yahoo Finance에서 USD/KRW 환율 데이터를 다운로드, CSV 캐시가 존재하면 재다운로드 없이 로드


    Parameters
    ----------
    start : str
        데이터 시작일. 기본값 '2000-01-01'
    cache : str
        CSV 캐시 파일 경로. 기본값 'krw_data.csv'

    Returns
    -------
    df : DataFrame
        Columns: date (datetime64), Price (float)

    """
    if os.path.exists(cache):
        df = pd.read_csv(cache, parse_dates=['date'])
    else:
        raw = yf.download("KRW=X", start=start, auto_adjust=False)
        close = raw['Close']
        if isinstance(close, pd.DataFrame):      # MultiIndex 방어
            close = close.iloc[:, 0]
        df = close.to_frame(name='Price').rename_axis('date').reset_index()
        df.to_csv(cache, index=False, encoding='utf-8-sig')

    df['date'] = pd.to_datetime(df['date'])
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df = df.dropna(subset=['Price']).sort_values('date').reset_index(drop=True)
    return df


def load_signals(start='2000-01-01'):
    """
    글로벌 금융 지표를 Yahoo Finance에서 다운로드

    Parameters
    ----------
    start : str
        데이터 시작일 기본값: '2000-01-01'

    Returns
    -------
    signals : DataFrame
        Columns: VIX, DXY, USDCNY
        
    """
    
    vix = yf.download("^VIX",  start=start, auto_adjust=False)['Close']
    dxy = yf.download("UUP",   start=start, auto_adjust=False)['Close']
    cny = yf.download("CNY=X", start=start, auto_adjust=False)['Close']
    signals = pd.concat([vix, dxy, cny], axis=1)
    signals.columns = ['VIX', 'DXY', 'USDCNY']
    return signals


def load_foreign_flow(start='2000-01-01', market='KOSPI', cache='foreign_flow.csv'):
    """
    KRX 외국인 순매수대금 시계열

    CSV 캐시 반환, 없다면 pykrx로 다운로드

    Parameters
    ----------
    start  : str   — 시작일 'YYYY-MM-DD'
    market : str   — 'KOSPI' 또는 'KOSDAQ'
    cache  : str   — CSV 캐시 경로

    Returns
    -------
    flow : DataFrame
        Columns: date (datetime64), foreign_net (float, 원 단위)

    
    """
    if os.path.exists(cache):
        return pd.read_csv(cache, parse_dates=['date'])

    from pykrx import stock
    s = start.replace('-', '')
    e = pd.Timestamp.today().strftime('%Y%m%d')
    raw = stock.get_market_trading_value_by_date(s, e, market)

    flow = raw[['외국인']].copy()
    flow.columns = ['foreign_net']
    flow = flow.rename_axis('date').reset_index()
    flow['date'] = pd.to_datetime(flow['date'])
    flow.to_csv(cache, index=False, encoding='utf-8-sig')
    return flow


# 수익률 · 변동성 · VaR 

def calc_return(df):
    """
    로그수익률 계산: r_t = ln(P_t / P_{t-1})

    Parameters
    ----------
    df : DataFrame — 'Price' 열 필요

    Returns
    -------
    df : DataFrame — 'log_return' 열 추가
    
    """
    df = df.sort_values('date').copy()
    df['log_return'] = np.log(df['Price'] / df['Price'].shift(1))
    return df.dropna()


def calc_volatility(df, window=30):
    """
    Rolling 표준편차로 일별 변동성을 계산

    Parameters
    ----------
    df     : DataFrame — 'log_return' 열 필요
    window : int       — rolling 기간 (거래일). 기본값 30.

    Returns
    -------
    df : DataFrame — 'volatility' 열 추가
    
    """
    df['volatility'] = df['log_return'].rolling(window).std()
    return df.dropna()


def calc_realized_vol(df, window=30):
    """
    일별 변동성, σ_annual = σ_daily × √252

    Parameters
    ----------
    df     : DataFrame — 'log_return' 열 필요
    window : int       — rolling 기간 (거래일). 기본값 30.

    Returns
    -------
    df : DataFrame — 'realized_vol' 열 추가 
    
    """
    df = df.copy()
    df['realized_vol'] = df['log_return'].rolling(window, min_periods=window).std() * np.sqrt(252)
    return df.dropna(subset=['realized_vol']).reset_index(drop=True)


def calc_var(df, confidence_levels=(0.95, 0.99)):
    """
    Historical Simulation 방식으로 Value-at-Risk를 계산

    Parameters
    ----------
    df                : DataFrame — 'log_return' 열 필요
    confidence_levels : tuple     — 신뢰 수준. 기본값 (0.95, 0.99).

    Returns
    -------
    var_result : dict
        {'daily': {'VaR_95': float, 'VaR_99': float},
         'annual': {'VaR_95': float, 'VaR_99': float}}

    Notes
    -----
    연환산 VaR = 일별 VaR × √252 (i.i.d. 정규분포 가정 하의 근사값)
    실제 수익률은 fat tail을 가지므로 연환산 값은 과소추정될 수 있음
    
    """
    returns = df['log_return'].dropna().values
    out = {'daily': {}, 'annual': {}}
    for c in confidence_levels:
        key = f"VaR_{int(c * 100)}"
        daily = float(np.percentile(returns, (1 - c) * 100))
        out['daily'][key] = daily
        out['annual'][key] = daily * np.sqrt(252)
    return out


def calc_var_by_regime(df, confidence_levels=(0.95, 0.99), state_col='hmm_state_label'):
    """
    레짐별 Historical VaR, 전체 표본 단일 VaR이 가리는 분포 차이를 명시

    상태에 따라 일일 예상 손실이 약 1.9배 차이 발생
    
    Returns
    -------
    {regime: {'VaR_95': float, 'VaR_99': float}}
    
    """
    out = {}
    for label in ['Stable', 'Transition', 'Crisis']:
        r = df.loc[df[state_col] == label, 'log_return'].dropna().values
        if len(r) == 0:
            continue
        out[label] = {f"VaR_{int(c * 100)}": float(np.percentile(r, (1 - c) * 100))
                      for c in confidence_levels}
    return out


# 레짐 탐지

def detect_regime(df, k=3):
    """
    변동성 순서대로 0=Stable, 1=Transition, 2=Crisis 로 지정

    Parameters
    ----------
    df : DataFrame — 'volatility' 열 필요
    k  : int       — 클러스터 수. 기본값 3

    Returns
    -------
    df : DataFrame — 'state' 열 추가

    """
    data = df[['volatility']].dropna()
    model = KMeans(n_clusters=k, random_state=0, n_init=10)
    df.loc[data.index, 'state'] = model.fit_predict(data)

    # 클러스터 번호는 매 실행 랜덤이라 변동성 크기 순으로 0/1/2 고정
    centers = model.cluster_centers_.flatten()
    rank = {old: new for new, old in enumerate(np.argsort(centers))}
    df['state'] = df['state'].map(rank)
    return df


def detect_regime_hmm(df, n_states=3, n_iter=200, random_state=42):
    """
    시장 레짐을 탐지

    관측 변수: log_return, volatility (StandardScaler 표준화 후 학습)

    레이블 재정렬 기준:
    1차: 평균 변동성 크기 (σ_Stable < σ_Transition < σ_Crisis)
    2차: 변동성이 유사한 경우(< 5% 차이) 이론 전이 자기지속 확률로 보정

    Parameters
    ----------
    df           : DataFrame — 'log_return', 'volatility' 열 필요
    n_states     : int — 상태 수. 기본값 3
    n_iter       : int — EM 최대 반복 횟수. 기본값 200
    random_state : int — 재현성 시드. 기본값 42

    Returns
    -------
    df    : DataFrame — 'hmm_state' (int), 'hmm_state_label' (str) 열 추가
    model : GaussianHMM — 학습 완료 HMM 객체

    References
    ----------
    Hamilton (1989); Ang & Bekaert (2002); Guidolin & Timmermann (2007)
    
    """
    df = df.copy()
    obs = df[['log_return', 'volatility']].dropna()
    X = StandardScaler().fit_transform(obs.values)   # GaussianHMM은 스케일 민감 → 표준화 필수

    model = GaussianHMM(n_components=n_states, covariance_type='full',
                        n_iter=n_iter, random_state=random_state, verbose=False)
    model.fit(X)
    df.loc[obs.index, 'hmm_state'] = model.predict(X)

    # 1차: 평균 변동성 오름차순
    mean_vol = {s: df.loc[df['hmm_state'] == s, 'volatility'].mean() for s in range(n_states)}
    order = sorted(mean_vol, key=mean_vol.get)

    # 2차: 인접 비위기 두 상태가 5% 이내로 붙으면 자기지속 확률로 보정
    VOL_TIE = 0.05
    for i in range(len(order) - 2):                  # 3-state면 i=0 (Stable vs Transition)만
        a, b = order[i], order[i + 1]
        if abs(mean_vol[b] - mean_vol[a]) / max(mean_vol[a], mean_vol[b]) < VOL_TIE:
            if model.transmat_[a, a] < model.transmat_[b, b]:
                order[i], order[i + 1] = b, a

    remap = {old: new for new, old in enumerate(order)}
    df['hmm_state'] = df['hmm_state'].map(remap)
    model.transmat_ = model.transmat_[np.ix_(order, order)]   # 전이행렬도 같은 순서로
    df['hmm_state_label'] = df['hmm_state'].map(REGIME_LABELS)
    return df, model


def get_transition_matrix(model, df=None):
    """
    HMM으로부터 이론/ 실증 전이 확률 행렬을 추출

    Parameters
    ----------
    model : GaussianHMM — detect_regime_hmm 반환 객체
    df    : DataFrame or None — 'hmm_state' 열 포함 시 실증 행렬도 계산

    Returns
    -------
    trans_df     : DataFrame — 이론 전이 확률 행렬 (행 합계 = 1.0)
    emp_trans_df : DataFrame or None — 실증 전이 빈도 행렬

    Notes
    -----
    P[i][j]: 상태 i에서 다음 날 상태 j로 전이할 확률
    """
    n = model.transmat_.shape[0]
    labels = [REGIME_LABELS[i] for i in range(n)]
    trans_df = pd.DataFrame(model.transmat_, index=labels, columns=labels)

    emp_trans_df = None
    if df is not None and 'hmm_state' in df.columns:
        states = df['hmm_state'].dropna().astype(int).values
        m = np.zeros((n, n), dtype=int)
        for t in range(len(states) - 1):
            i, j = states[t], states[t + 1]
            if 0 <= i < n and 0 <= j < n:
                m[i][j] += 1
        row_sums = m.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1                  # 0 나눗셈 방지
        emp_trans_df = pd.DataFrame(m / row_sums, index=labels, columns=labels)

    return trans_df, emp_trans_df


# 헤지 비율 · 예산

def calc_hedge_ratio(regime, firm_type, size, skew, base=None):
    """  
    Regime 기반 가중 헤지 비율을 산출


    Parameters
    ----------
    regime    : str   — 'Stable' / 'Transition' / 'Crisis'
    firm_type : str   — '수출' / '수입'
    size      : str   — '대기업' / '중소기업'
    skew      : float — 해당 레짐의 수익률 왜도

    Returns
    -------
    float — 헤지 비율 [0, 1]

    Notes
    -----
    Δexposure 설계 근거:
        Crisis   : delta_exp = 0   — 위기 시 기업유형 무관 최대 방어 (base 90% 고정)
        Transition: 수출 -10%, 수입 +10%  — 변동성 확대 구간 방어 강화
        Stable   : 수출 +10%, 수입 -5%   — 수입 -5%는 Stable 양의 왜도(KRW 약세 꼬리) 반영
        
    """
    base_map = base or HEDGE_BASE
    b = base_map[regime]

    if regime == 'Crisis':
        d_exp = 0.0
    elif regime == 'Transition':
        d_exp = -0.10 if firm_type == '수출' else +0.10
    else:  # Stable
        d_exp = +0.10 if firm_type == '수출' else -0.05

    d_size = -0.05 if size == '대기업' else +0.05
    d_skew = float(np.clip(0.02 * max(-skew, 0), 0, 0.05))
    return float(np.clip(b + d_exp + d_size + d_skew, 0, 1))


def calc_hedge_budget(regime_var, self_persistence, lam=1.0):
    """ 
    레짐별 헤지 예산

    Parameters
    ----------
    regime_var : float
    self_persistence : float
        자기지속 확률 P(ii) — 실증 전이행렬 대각값
    lam : float, optional
        헤지 예산 허용 비율 λ. 기업이 설정하는 정책 파라미터. 기본값 1.0

    Returns
    -------
    dict
        E_duration : float — 기대 지속 일수, 1 / (1 - P(ii))
        daily      : float — 일일 예산, λ × VaR_t
        cumulative : float — 누적 기대비, 일일 × 기대 지속

    Notes
    -----
    Crisis는 P(C→C) ≈ 0.98 → 기대지속 50일 이상 → 누적비용이 폭증한다.
    선제 헤지가 사후 헤지보다 구조적으로 유리한 근거다 (슬라이드 31).
    P(ii) = 1.0 인 흡수상태(표본 내 이탈 없음)는 0.9999로 상한을 둔다.
    
    """
    # P(ii)=1이면 표본 내 흡수상태(빠져나온 적 없음) → 기대지속 발산. 보수적으로 상한 처리.
    self_persistence = min(self_persistence, 0.9999)
    duration = 1.0 / (1.0 - self_persistence)
    daily = lam * regime_var
    return {'E_duration': duration, 'daily': daily, 'cumulative': daily * duration}


# 글로벌 신호 상관

def calc_signal_correlation(df, signals, lag_days=5):
    """
    글로벌 지표와 KRW 변동성의 리드-래그 상관 비교

    Parameters
    ----------
    df       : DataFrame — 'date', 'volatility' 열 필요
    signals  : DataFrame — load_signals() 반환값
    lag_days : int       — 최대 시차 (거래일). 기본값 5

    Returns
    -------
    corr_result : dict
        'contemporaneous' : Series  — 동시 Pearson 상관계수
        'lead_lag'        : DataFrame — lag별 상관계수 (index: -lag_days ~ +lag_days)

    Notes
    -----
    양수 lag = 글로벌 지표가 KRW 변동성에 선행
    
    """
    merged = df.set_index('date')[['volatility']].join(signals, how='inner').dropna()
    cols = [c for c in ['VIX', 'DXY', 'USDCNY'] if c in merged.columns]

    contemporaneous = merged[cols].corrwith(merged['volatility'])

    records = {}
    for lag in range(-lag_days, lag_days + 1):
        row = {}
        for c in cols:
            pair = pd.concat([merged['volatility'], merged[c].shift(lag)], axis=1).dropna()
            row[c] = stats.pearsonr(pair.iloc[:, 0], pair.iloc[:, 1])[0] if len(pair) > 10 else np.nan
        records[lag] = row
    lead_lag = pd.DataFrame(records).T
    lead_lag.index.name = 'lag_days'

    return {'contemporaneous': contemporaneous, 'lead_lag': lead_lag}


# Granger 인과 (확장 분석)

def build_causality_frame(df, flow, signals=None):
    """
    Granger 검정용 데이터프레임

    Parameters
    ----------
    df      : DataFrame — calc_return 처리 완료 (date, log_return)
    flow    : DataFrame — load_foreign_flow 반환값
    signals : DataFrame or None — load_signals 반환값 (VIX 교란 통제용)

    Returns
    -------
    frame : DataFrame
        fx_ret      : 원화 로그수익률
        foreign_net : 외국인 순매수대금
        vix         : signals 제공 시 포함
        
        """
    frame = df[['date', 'log_return']].rename(columns={'log_return': 'fx_ret'}).merge(flow, on='date')
    if signals is not None:
        vix = signals[['VIX']].reset_index()
        vix.columns = ['date', 'vix']
        vix['date'] = pd.to_datetime(vix['date'])
        frame = frame.merge(vix, on='date')
    return frame.set_index('date').dropna()


def test_stationarity(frame, cols, signif=0.05):
    """
    ADF 단위근 검정으로 각 시계열의 정상성을 확인
    비정상 시계열에 Granger를 적용하면 허위 결과가 발생

    Returns
    -------
    DataFrame : series별 ADF 통계량, p값, 정상성 여부
    
    """
    from statsmodels.tsa.stattools import adfuller
    rows = []
    for c in cols:
        stat, p, *_ = adfuller(frame[c].dropna(), autolag='AIC')
        rows.append({'series': c, 'ADF_stat': round(stat, 3),
                     'p_value': round(p, 4), 'stationary': p < signif})
    return pd.DataFrame(rows).set_index('series')


def granger_causality(frame, x='fx_ret', y='foreign_net', control=None, maxlag=10, signif=0.05):
    """
    양방향 Granger 인과를 검정

    process
    ----
    1. ADF 정상성 확인 → 비정상 변수 1차 차분
    2. VAR 적합 후 AIC로 최적 시차 자동 선택
    3. VAR 안정성 확인
    4. 양방향 F검정
    5. 잔차 백색성 검정

    Returns
    -------
    dict : lag, 안정성, 양방향 p값, 잔차백색성 p값, 통제변수
    
    """
    from statsmodels.tsa.api import VAR

    cols = [x, y] + ([control] if control else [])
    data = frame[cols].copy()

    st = test_stationarity(data, cols, signif)
    for c in cols:
        if not st.loc[c, 'stationary']:
            data[c] = np.log(data[c]).diff() if (data[c] > 0).all() else data[c].diff()
    data = data.dropna()

    model = VAR(data)
    lag = max(int(model.select_order(maxlag).aic), 1)
    res = model.fit(lag)
    return {
        'lag(AIC)': lag,
        'stable': res.is_stable(verbose=False),
        f'{x}->{y}_p': round(res.test_causality(y, [x], kind='f').pvalue, 6),
        f'{y}->{x}_p': round(res.test_causality(x, [y], kind='f').pvalue, 6),
        'residual_white_p': round(res.test_whiteness(nlags=lag + 5, adjusted=True).pvalue, 4),
        'controlled_for': control,
    }


# 시각화

def plot_regime_timeline(df, state_col='hmm_state', label_col='hmm_state_label'):
    """
    레짐 타임라인 산점도

    Parameters
    ----------
    df : DataFrame — 'date', 'volatility', 'state' 열 필요
    
    """
    colors = ['green', 'orange', 'red']
    plt.figure(figsize=(12, 4))
    for s in sorted(df[state_col].dropna().unique()):
        sub = df[df[state_col] == s]
        plt.scatter(sub['date'], sub['volatility'], color=colors[int(s)],
                    label=sub[label_col].iloc[0], s=4, alpha=0.7)
    plt.title("USD/KRW HMM Volatility Regime Timeline")
    plt.xlabel("Date"); plt.ylabel("Volatility"); plt.legend()
    plt.show()


def plot_transition_matrix(trans_df, title='HMM Regime Transition Probability'):
    """
    전이 확률 행렬의 시각화

    Parameters
    ----------
    trans_df : DataFrame — get_transition_matrix() 반환값
    title    : str       — 그래프 제목

    Returns
    -------
    fig : matplotlib.figure.Figure
    
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(trans_df.values, cmap=plt.cm.RdYlGn, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Transition Probability')
    for i in range(len(trans_df)):
        for j in range(len(trans_df)):
            v = trans_df.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha='center', va='center', fontsize=11,
                    color='white' if (v > 0.7 or v < 0.15) else 'black', fontweight='bold')
    ax.set_xticks(range(len(trans_df))); ax.set_yticks(range(len(trans_df)))
    ax.set_xticklabels(trans_df.columns); ax.set_yticklabels(trans_df.index)
    ax.set_xlabel("To State"); ax.set_ylabel("From State"); ax.set_title(title, fontsize=11)
    plt.tight_layout(); plt.show()
    return fig


def plot_signal_corr(corr_result):
    """
    VIX / DXY / USDCNY와 KRW 변동성의 리드-래그 상관을 시각화

    Parameters
    ----------
    corr_result : dict — calc_signal_correlation() 반환값

    Returns
    -------
    fig : matplotlib.figure.Figure

    Notes
    -----
    x축 lag > 0: 글로벌 지표가 KRW 변동성에 선행
    x축 lag < 0: KRW 변동성이 글로벌 지표에 선행
    
    """
    lead_lag = corr_result['lead_lag']
    colors = {'VIX': '#e74c3c', 'DXY': '#3498db', 'USDCNY': '#2ecc71'}
    fig, ax = plt.subplots(figsize=(10, 5))
    for c in lead_lag.columns:
        ax.plot(lead_lag.index, lead_lag[c], marker='o', markersize=3, linewidth=1.5,
                color=colors.get(c, 'gray'), label=c)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_title("Lead-Lag Correlation: Global Signals vs KRW Volatility\n"
                 "(positive lag = signal leads KRW volatility)", fontsize=11)
    ax.set_xlabel("Lag (trading days)"); ax.set_ylabel("Pearson Correlation"); ax.legend()
    plt.tight_layout(); plt.show()
    return fig