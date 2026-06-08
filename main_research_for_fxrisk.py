# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 14:48:42 2026

@author: bsjhw
"""
"""
USD/KRW FX RISK 분석

    1. 시장 구조      : 환율 추이 · 연환산 변동성
    2. 리스크 정량화  : 전체표본 VaR · Jarque-Bera
    3. 변동성 지속성  : ACF · 위기 이벤트 오버레이
    4. HMM 레짐 탐지  : Stable / Transition / Crisis
    5. 전이행렬       : 이론 · 실증
    6. 레짐별 VaR     : 상태별 손실 차이 + 시나리오
    7. 글로벌 신호    : VIX / DXY / USDCNY 리드-래그
    8. 탐지 지연      : 충격 유형별 대응 가능 창
    9. 헤지 비율      : 레짐 기반 가중 헤지표
    10. 헤지 예산     : VaR × 기대지속

    장효원 (JangHyowon-biz)
"""

import coremodel_fxrisk as main
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf

plt.style.use("ggplot")
main.set_style()


# 분석 정책 파라미터 - 역사적 변동성이 투입된 가상의 시나리오
POSITION_USD    = 100_000_000     # 시나리오 노출 규모 ($100M)
USDKRW          = 1450            # 시나리오 환율 (원)
VIX_LEAD_DAYS   = 5               # VIX/CNY 선행 신호
EXEC_DAYS       = 3               # CFO 승인 1~2일 + 결제 D+2
LAMBDA          = 1.0             # 헤지 예산 허용 비율 λ (기업 설정)

CRISIS_EVENTS = {
    'GFC':     '2008-09-15',      # 리먼 파산
    'EU_Debt': '2011-08-05',      # 유럽 재정위기 S&P 강등
    'COVID':   '2020-03-11',      # WHO 팬데믹 선언
    'Fed75bp': '2022-06-15',      # Fed 75bp 인상
}


# 데이터 · 시장 구조
df = main.load_fx_data()
df = main.calc_return(df)
df = main.calc_volatility(df)
df = main.detect_regime(df)
df = df.dropna()

plt.figure(figsize=(10, 4))
plt.plot(df['date'], df['Price'])
plt.title("USD/KRW Exchange Rate (2000–present)")
plt.xlabel("Date"); plt.ylabel("KRW per USD")
plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.YearLocator(1))
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
plt.gcf().autofmt_xdate()
plt.show()

df = main.calc_realized_vol(df, window=30)
plt.figure(figsize=(10, 4))
plt.plot(df['date'], df['realized_vol'])
plt.title("USD/KRW Annualized Volatility (30d Rolling)")
plt.xlabel("Date"); plt.ylabel("Annualized Volatility")
plt.show()


# 리스크 정량화 (전체 표본 VaR)
var_result = main.calc_var(df)
print("[ 전체 표본 Daily VaR ]")
print(f"  95% : {var_result['daily']['VaR_95']:.4%}")
print(f"  99% : {var_result['daily']['VaR_99']:.4%}")
print("[ 연환산 VaR — i.i.d. 근사, fat tail로 과소추정 가능 ]")
print(f"  95% : {var_result['annual']['VaR_95']:.4%}")
print(f"  99% : {var_result['annual']['VaR_99']:.4%}")

plt.figure(figsize=(8, 4))
plt.hist(df['log_return'], bins=100)
plt.axvline(var_result['daily']['VaR_95'], color='red',     linestyle='--', label='95% VaR')
plt.axvline(var_result['daily']['VaR_99'], color='darkred', linestyle='--', label='99% VaR')
plt.legend(); plt.title("USD/KRW Return Distribution with VaR")
plt.show()

jb_stat, jb_p = stats.jarque_bera(df['log_return'].dropna())
print(f"Jarque-Bera p-value: {jb_p:.4e} → 정규분포 기각, fat tail 확인")


# 변동성 지속성
fig, ax = plt.subplots(figsize=(10, 4))
plot_acf(df['volatility'].dropna(), lags=150, alpha=0.05, zero=False, ax=ax)
ax.set_title("Volatility Persistence (ACF, lag 1–150 days)")
ax.set_xlabel("Lag (trading days)"); ax.set_ylabel("Autocorrelation")
ax.axhline(0, color='black', linewidth=0.8)
plt.tight_layout(); plt.show()

plt.figure(figsize=(12, 4))
plt.plot(df['date'], df['volatility'], label='Volatility')
for d, name in {'2008-09-15': 'Lehman/GFC', '2011-08-05': 'EU Debt',
                '2020-03-11': 'COVID', '2022-02-24': 'Russia–Ukraine',
                '2022-06-15': 'Fed 75bp'}.items():
    plt.axvline(pd.to_datetime(d), linestyle='--', alpha=0.6)
    plt.text(pd.to_datetime(d), df['volatility'].max() * 0.9, name, rotation=90, fontsize=8)
plt.title("USD/KRW Volatility with Major Events")
plt.xlabel("Date"); plt.ylabel("Volatility"); plt.legend()
plt.show()


# HMM 레짐 탐지 3-State 
df, hmm_model = main.detect_regime_hmm(df, n_states=3, n_iter=200)
assert hmm_model.monitor_.converged, "HMM이 수렴하지 않음 — n_iter를 늘리거나 데이터 확인"

main.plot_regime_timeline(df)

# 레짐 요약 : 발생비중 · 평균변동성 · 자기지속
trans_df, emp_trans_df = main.get_transition_matrix(hmm_model, df)
summary = df.groupby('hmm_state_label')['volatility'].agg(['count', 'mean']).round(4)
summary['share'] = (summary['count'] / summary['count'].sum() * 100).round(1)
summary['self_persist'] = [emp_trans_df.loc[r, r] for r in summary.index]
print("\n[ 레짐 요약 ]")
print(summary.loc[['Stable', 'Transition', 'Crisis']].round(4))

# 자체검증 ,정식 테스트 대신 inline 가드
#   Stable<Transition 변동성은 단언하지 아니함
#   상태 0과 1은 σ가 통계적으로 거의 같고(0.0052 vs 0.0053) 진짜 구분축은 방향성·지속성으로 구분한다
vol_by = summary['mean']
assert vol_by['Crisis'] == vol_by.max(), "Crisis가 최고 변동성이 아님 — 레이블 정렬 확인"

# 보정 후 Stable(자기지속 높음) ≥ Transition(낮음)
p_stable, p_trans = hmm_model.transmat_[0, 0], hmm_model.transmat_[1, 1]
assert p_stable >= p_trans, (
    f"레이블 경고: P(S→S)={p_stable:.3f} < P(T→T)={p_trans:.3f} "
    "→ detect_regime_hmm 2차 보정 확인")

# 전이행렬 각 행 합 = 1
assert np.allclose(hmm_model.transmat_.sum(axis=1), 1.0), "전이행렬 행 합 ≠ 1"

# 표준화 공간 평균 · 레짐별 수익률 분포
print("\n[ HMM means_ (표준화 공간) ]")
print(pd.DataFrame(hmm_model.means_, index=['Stable', 'Transition', 'Crisis'],
                   columns=['log_return(std)', 'volatility(std)']).round(4))

print("\n[ 레짐별 수익률 분포 — 방향성·꼬리 ]")
dist = []
for lbl in ['Stable', 'Transition', 'Crisis']:
    r = df[df['hmm_state_label'] == lbl]['log_return'].dropna()
    dist.append({'Regime': lbl, 'count': len(r), 'mean_return': round(r.mean(), 6),
                 'skewness': round(r.skew(), 4), 'kurtosis': round(r.kurt(), 4)})
print(pd.DataFrame(dist).set_index('Regime'))


# 전이행렬 (이론 · 실증)
print("\n[ 이론 전이행렬 ]")
print(trans_df.round(4))
print("\n[ 실증 전이행렬 ]")
print(emp_trans_df.round(4))
main.plot_transition_matrix(trans_df)


# 레짐별 VaR + 시나리오 손실
regime_var = main.calc_var_by_regime(df)
print("\n[ 레짐별 95% / 99% VaR ]")
for lbl in ['Stable', 'Transition', 'Crisis']:
    if lbl in regime_var:
        v = regime_var[lbl]
        print(f"  {lbl:11} 95% {v['VaR_95']:.4%}   99% {v['VaR_99']:.4%}")

# $100M 노출 기준 일일 예상 손실 (Stable vs Crisis)
exposure_krw = POSITION_USD * USDKRW
for lbl in ['Stable', 'Crisis']:
    loss = abs(regime_var[lbl]['VaR_95']) * exposure_krw
    print(f"  {lbl} 95% 일일 예상 손실 ≈ {loss/1e8:.1f}억원")


# 글로벌 신호 리드-래그
try:
    signals = main.load_signals(start='2000-01-01')
    corr = main.calc_signal_correlation(df, signals, lag_days=VIX_LEAD_DAYS)
    print("\n[ 리드-래그 최대 상관 시차 ]")
    for c in corr['lead_lag'].columns:
        s = corr['lead_lag'][c]
        best = s.abs().idxmax()
        where = (f"{c}가 KRW에 {best}일 선행" if best > 0 else
                 f"KRW가 {c}에 {abs(best)}일 선행" if best < 0 else "동시")
        print(f"  {c}: lag {best:+d}, 상관 {s[best]:.3f} → {where}")
    main.plot_signal_corr(corr)
except Exception as e:
    print(f"[경고] 글로벌 신호 분석 건너뜀: {e}")


# 탐지 지연 · 행동 가능 창
print("\n[ Crisis 탐지 지연 ]")
df_sorted = df.sort_values('date').reset_index(drop=True)
lags = []
for name, date in CRISIS_EVENTS.items():
    dt = pd.to_datetime(date)
    after = df_sorted[(df_sorted['date'] >= dt) & (df_sorted['hmm_state_label'] == 'Crisis')]
    if len(after):
        lag = (after['date'].iloc[0] - dt).days
        lags.append(lag)
        print(f"  {name:9} {date}  →  {after['date'].iloc[0].date()}  ({lag}일)")
    else:
        print(f"  {name:9} {date}  →  탐지 없음")

if lags:
    avg_lag = np.mean(lags)
    window = VIX_LEAD_DAYS - avg_lag - EXEC_DAYS
    print(f"  평균 지연 {avg_lag:.1f}일 (최소 {min(lags)} / 최대 {max(lags)})")
    print(f"  행동 가능 창 = 선행 {VIX_LEAD_DAYS}일 − 지연 {avg_lag:.1f}일 − 실행 {EXEC_DAYS}일 = {window:.1f}일")
    if window <= 0:
        print("  → 선제 헤지가 구조적으로 불가능 → Pre-committed Forward 필요")
    elif window <= 2:
        print("  → 창이 매우 협소 → 사전 약정 헤지 권장")
    else:
        print("  → 제한적이나 선제 행동 가능")


# 헤지 비율표
skew_by = {lbl: df[df['hmm_state_label'] == lbl]['log_return'].skew()
           for lbl in ['Stable', 'Transition', 'Crisis']}

print("\n[ 레짐 기반 헤지 비율 ]")
rows = []
for regime in ['Stable', 'Transition', 'Crisis']:
    for firm in ['수출', '수입']:
        for size in ['대기업', '중소기업']:
            hr = main.calc_hedge_ratio(regime, firm, size, skew_by[regime])
            rows.append({'Regime': regime, '유형': firm, '규모': size,
                         'skew': round(skew_by[regime], 4), 'HR': f"{hr*100:.1f}%"})
print(pd.DataFrame(rows).to_string(index=False))


# 헤지 예산 (Transition은 VaR 산출에서 배제하고 Stable VaR를 보수적으로 차용)
print("\n[ 헤지 예산 (λ 배수) ]")
var_for_budget = {
    'Stable':     abs(regime_var['Stable']['VaR_95']),
    'Transition': abs(regime_var['Stable']['VaR_95']),
    'Crisis':     abs(regime_var['Crisis']['VaR_95']),
}
for regime in ['Stable', 'Transition', 'Crisis']:
    b = main.calc_hedge_budget(var_for_budget[regime], emp_trans_df.loc[regime, regime], lam=LAMBDA)
    print(f"  {regime:11} VaR {var_for_budget[regime]:.2%}  "
          f"기대지속 {b['E_duration']:.1f}일  누적 λ×{b['cumulative']:.2%}")


# 확장 분석 (WIP) — 원화 약세 ↔ 외국인 순매도 Granger( pykrx 설치 후 주석 해제)
# flow  = main.load_foreign_flow(start='2000-01-01', market='KOSPI')
# frame = main.build_causality_frame(df, flow, signals=signals)
# print(main.test_stationarity(frame, ['fx_ret', 'foreign_net']))
# print(main.granger_causality(frame, 'fx_ret', 'foreign_net', control=None))   # 2변량
# print(main.granger_causality(frame, 'fx_ret', 'foreign_net', control='vix'))  # VIX 통제