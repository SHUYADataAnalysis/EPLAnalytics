"""
EPL Analytics — Custom Metrics Builder
データ: vaastav/Fantasy-Premier-League (GitHub)
ライセンス: FPL data © Premier League, non-commercial personal use
"""

import io, time, warnings
import numpy as np
import pandas as pd
import matplotlib
try:
    matplotlib.use("Agg")
except Exception:
    pass
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import requests
import streamlit as st
from scipy.stats import pearsonr, rankdata
from scipy.stats import zscore as sp_zscore

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
VAASTAV = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
SEASON  = "2024-25"
FULL_MIN = 3420.0
POS_MAP  = {1:"GK", 2:"DEF", 3:"MID", 4:"FWD"}

C = dict(
    pitch   = "#1a5c36",   # ピッチグリーン
    pitch_l = "#27834e",
    chalk   = "#1a1a2e",   # 本文テキスト（濃紺）← 白背景に対して高コントラスト
    amber   = "#c45c00",   # アンバー（濃いめ）
    sky     = "#0077aa",   # スカイブルー（濃いめ）
    card    = "#ffffff",
    muted   = "#444444",   # サブテキスト
    dark    = "#0f172a",   # ヘッダー背景
    neg     = "#cc2200",
    pos     = "#1a7a3a",
    bg      = "#f5f7fa",   # ページ背景（薄いグレー）
    sidebar = "#1e2d3d",   # サイドバー背景（濃紺）
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="EPL Analytics", layout="wide", page_icon="⚽")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&family=Bebas+Neue&display=swap');

html, body { background: #f5f7fa !important; }
[data-testid="stAppViewContainer"] { background: #f5f7fa !important; }
[data-testid="stAppViewContainer"] > .main { background: #f5f7fa !important; }
.main .block-container { background: #f5f7fa !important; }

p, span, li, td, th { color: #1a1a2e !important; }
.stMarkdown, .stMarkdown * { color: #1a1a2e !important; }
[data-testid="stMarkdownContainer"] * { color: #1a1a2e !important; }
h1, h2, h3, h4 { color: #1a1a2e !important; }
label { color: #1a1a2e !important; }
[data-testid="stWidgetLabel"] { color: #1a1a2e !important; }
[data-testid="stWidgetLabel"] * { color: #1a1a2e !important; }

[data-testid="stSidebar"] { background: #1e2d3d !important; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #ffffff !important; }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] * { color: #b0c4d8 !important; }

.stSelectbox > div > div > div { background: #ffffff !important; color: #1a1a2e !important; }
.stMultiSelect > div > div > div { background: #ffffff !important; color: #1a1a2e !important; }
[data-baseweb="select"] * { color: #1a1a2e !important; }
[data-baseweb="popover"] { background: #ffffff !important; }
[data-baseweb="popover"] * { color: #1a1a2e !important; background: #ffffff !important; }
[data-baseweb="option"] { color: #1a1a2e !important; background: #ffffff !important; }
[data-baseweb="option"]:hover { background: #e8f4fd !important; }
[data-baseweb="tag"] { background: #1a5c36 !important; }
[data-baseweb="tag"] span { color: #ffffff !important; }
input { background: #ffffff !important; color: #1a1a2e !important; }
input[type="password"] { color: #1a1a2e !important; }

.stTabs [data-baseweb="tab-list"] { border-bottom: 2px solid #1a5c36; }
.stTabs [data-baseweb="tab"] {
  background: #e8f0eb; border-radius: 6px 6px 0 0;
  color: #1a1a2e !important; font-weight: 600; font-size: .82rem; padding: 6px 14px;
}
.stTabs [aria-selected="true"] { background: #1a5c36 !important; }
.stTabs [aria-selected="true"] * { color: #ffffff !important; }

.section-bar {
  height: 4px;
  background: linear-gradient(90deg, #1a5c36, #c45c00, #0077aa, transparent);
  margin: .5rem 0 1rem; border-radius: 2px;
}
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
[data-testid="stDataFrame"] * { color: #1a1a2e !important; }

[data-testid="stNotification"] { background: #e8f4fd !important; }
[data-testid="stNotification"] * { color: #1a1a2e !important; }
.stAlert * { color: #1a1a2e !important; }
[data-testid="stSuccess"] { background: #dcfce7 !important; }
[data-testid="stSuccess"] * { color: #166534 !important; }
[data-testid="stError"] { background: #fee2e2 !important; }
[data-testid="stError"] * { color: #991b1b !important; }

.stRadio * { color: #1a1a2e !important; }
.stSlider * { color: #1a1a2e !important; }
.stMultiSelect * { color: #1a1a2e !important; }
.stCaptionContainer { color: #444444 !important; }
</style>
""", unsafe_allow_html=True)

# ── Data fetch ────────────────────────────────────────────────────────────────

# =========================================================
# API-Football 統合（シュート・ポゼッション・コーナー等）
# =========================================================
APF_BASE   = "https://v3.football.api-sports.io"
EPL_LEAGUE = 39
SEASON_TO_APF = {"2025-26": 2025, "2024-25": 2024, "2023-24": 2023, "2022-23": 2022}

def _apf_get(endpoint: str, params: dict, api_key: str) -> dict | None:
    """API-Football への単一リクエスト"""
    hdrs = {"x-apisports-key": api_key, "Accept": "application/json"}
    try:
        r = requests.get(f"{APF_BASE}/{endpoint}", headers=hdrs,
                         params=params, timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def check_apf_key(api_key: str) -> tuple:
    """APIキー確認と残リクエスト数を返す"""
    data = _apf_get("status", {}, api_key)
    if data and "response" in data:
        req = data["response"].get("requests", {})
        used = int(req.get("current", 0))
        limit = int(req.get("limit_day", 100))
        return True, used, limit - used
    return False, 0, 0


def fetch_finished_fixture_ids(season_str: str, api_key: str) -> list:
    """
    全試合IDを取得し終了済み(FT/AET/PEN)のみ返す（1リクエスト消費）。
    session_stateで簡易キャッシュ（ページ内で重複呼び出しを防ぐ）。
    """
    # セッションキャッシュから返す（同一セッション内で再利用）
    _fid_key = f"fixture_ids_{season_str}"
    if _fid_key in st.session_state and st.session_state[_fid_key]:
        return st.session_state[_fid_key]

    apf_season = SEASON_TO_APF.get(season_str)
    if not apf_season:
        return []

    data = _apf_get("fixtures",
                    {"league": EPL_LEAGUE, "season": apf_season},
                    api_key)
    if not data:
        return []

    finished_statuses = {"FT", "AET", "PEN"}
    ids = [
        f["fixture"]["id"]
        for f in data.get("response", [])
        if f.get("fixture", {}).get("status", {}).get("short") in finished_statuses
    ]

    if ids:  # 空リストはキャッシュしない
        st.session_state[_fid_key] = ids
    return ids


def _parse_stats(response: list) -> dict:
    """fixture/statistics レスポンスを {home:{...}, away:{...}} に変換"""
    result = {}
    for i, team_data in enumerate(response[:2]):
        side = "home" if i == 0 else "away"
        tname = team_data.get("team", {}).get("name", "")
        stats = {"team_name": tname}
        for stat in team_data.get("statistics", []):
            val = stat.get("value")
            if isinstance(val, str) and val.endswith("%"):
                try: val = float(val.rstrip("%"))
                except: val = None
            elif val is not None:
                try: val = float(val)
                except: pass
            stats[stat["type"]] = val
        result[side] = stats
    return result


# セッションステートが消えても残るようにキャッシュキーを管理
_APF_CACHE_KEY = "apf_stats_v1"

def _load_apf_cache() -> dict:
    """セッションから統計キャッシュを読む"""
    import json as _j
    raw = st.session_state.get(_APF_CACHE_KEY, "{}")
    try:
        return _j.loads(raw) if raw else {}
    except Exception:
        return {}

def _save_apf_cache(data: dict) -> None:
    """統計キャッシュをセッションに保存"""
    import json as _j
    st.session_state[_APF_CACHE_KEY] = _j.dumps(data)


def fetch_and_cache_stats(
    fixture_ids: list, api_key: str, max_per_run: int = 80
) -> dict:
    """
    未取得の試合だけAPIを叩いて session_state に保存。
    ボタン押下 → ページ再実行 → session_state は維持される。
    ページを閉じると消えるが、毎回ボタン押下で追加取得できる。
    """
    import time as _time

    cached  = _load_apf_cache()
    missing = [fid for fid in fixture_ids if str(fid) not in cached]
    to_fetch = missing[:max_per_run]

    if not to_fetch:
        return {int(k): v for k, v in cached.items() if int(k) in fixture_ids}

    prog = st.progress(0)
    msg  = st.empty()
    newly_fetched = 0

    for i, fid in enumerate(to_fetch):
        msg.caption(f"📡 取得中: {i+1}/{len(to_fetch)} 試合...")
        prog.progress((i+1) / len(to_fetch))
        data = _apf_get("fixtures/statistics", {"fixture": fid}, api_key)
        if data and data.get("response"):
            cached[str(fid)] = _parse_stats(data["response"])
            newly_fetched += 1
        _time.sleep(0.35)

    prog.empty()
    msg.empty()

    _save_apf_cache(cached)

    if newly_fetched > 0:
        st.sidebar.success(f"✅ {newly_fetched}試合を新たに取得しました")

    return {int(k): v for k, v in cached.items() if int(k) in fixture_ids}


def build_apf_team_stats(fixture_cache: dict, df_teams: "pd.DataFrame") -> "pd.DataFrame":
    """
    fixture_cache からチームごとにシュート・ポゼッション等を集計し
    df_teams に列として追加する。
    """
    rows = []
    for fid, sides in fixture_cache.items():
        for side in ["home", "away"]:
            s = sides.get(side, {})
            if not s or not s.get("team_name"):
                continue
            rows.append({
                "team_name":        s["team_name"],
                "total_shots":      s.get("Total Shots"),
                "shots_on_target":  s.get("Shots on Goal"),
                "shots_inside_box": s.get("Shots insidebox"),
                "possession_pct":   s.get("Ball Possession"),
                "corners":          s.get("Corner Kicks"),
                "fouls":            s.get("Fouls"),
                "offsides":         s.get("Offsides"),
                "passes_total":     s.get("Total passes"),
                "pass_accuracy":    s.get("Passes %"),
            })

    if not rows:
        return df_teams

    df_apf = pd.DataFrame(rows)
    for c in df_apf.columns:
        if c != "team_name":
            df_apf[c] = pd.to_numeric(df_apf[c], errors="coerce")

    agg = df_apf.groupby("team_name").agg(
        shots_pm          = ("total_shots",      "mean"),
        shots_on_tgt_pm   = ("shots_on_target",  "mean"),
        shots_inbox_pm    = ("shots_inside_box", "mean"),
        possession        = ("possession_pct",   "mean"),
        corners_pm        = ("corners",          "mean"),
        fouls_pm          = ("fouls",            "mean"),
        offsides_pm       = ("offsides",         "mean"),
        pass_acc          = ("pass_accuracy",    "mean"),
        passes_pm         = ("passes_total",     "mean"),
        n_fixtures        = ("total_shots",      "count"),
    ).reset_index()

    # 決定率
    merged = df_teams.merge(agg, on="team_name", how="left")
    merged["shot_conversion"] = (
        merged["gf_per_match"] / merged["shots_pm"].replace(0, np.nan) * 100
    ).round(1)
    return merged

def _get(url, timeout=20):
    for _ in range(3):
        try:
            r = requests.get(url, headers={"User-Agent":"Mozilla/5.0 Chrome/124"}, timeout=timeout)
            if r.status_code == 200:
                return r
        except Exception:
            pass
        time.sleep(2)
    return None

@st.cache_data(ttl=1800, show_spinner=False)
def load_season(season):
    r_p = _get(f"{VAASTAV}/{season}/players_raw.csv")
    r_g = _get(f"{VAASTAV}/{season}/gws/merged_gw.csv")
    r_t = _get(f"{VAASTAV}/{season}/teams.csv")
    return (
        pd.read_csv(io.StringIO(r_p.text)) if r_p else None,
        pd.read_csv(io.StringIO(r_g.text)) if r_g else None,
        pd.read_csv(io.StringIO(r_t.text)) if r_t else None,
    )

# ── Data preparation ──────────────────────────────────────────────────────────
NUM_COLS_PLAYER = [
    "minutes","goals_scored","assists","clean_sheets","goals_conceded",
    "saves","yellow_cards","red_cards","bonus","bps","total_points","now_cost",
    "expected_goals","expected_assists","expected_goal_involvements",
    "expected_goals_conceded","influence","creativity","threat","ict_index",
    "own_goals","penalties_saved","penalties_missed","starts",
    "clean_sheets_per_90","expected_goals_per_90","expected_assists_per_90",
    "expected_goal_involvements_per_90","expected_goals_conceded_per_90",
    "goals_conceded_per_90","saves_per_90","starts_per_90",
]

NUM_COLS_GW = [
    "goals_scored","assists","expected_goals","expected_assists",
    "expected_goal_involvements","expected_goals_conceded","goals_conceded",
    "saves","clean_sheets","yellow_cards","red_cards","bonus","minutes",
    "creativity","threat","influence","ict_index","team_a_score","team_h_score",
]

def prep_players(df_raw, team_id_map):
    df = df_raw.copy()
    df["player_name"] = df["web_name"] if "web_name" in df.columns else df.index.astype(str)
    df["position"]    = (df["element_type"] if "element_type" in df.columns
                         else pd.Series(0, index=df.index)).map(POS_MAP).fillna("UNK")
    df["team_name"]   = (df["team"] if "team" in df.columns
                         else pd.Series(0, index=df.index)).map(team_id_map).fillna("Unknown")
    for c in NUM_COLS_PLAYER:
        df[c] = pd.to_numeric(df[c] if c in df.columns
                              else pd.Series(0, index=df.index), errors="coerce").fillna(0)
    df["price_m"]      = df["now_cost"] / 10.0
    df["goal_luck"]    = df["goals_scored"] - df["expected_goals"]
    df["def_luck"]     = df["expected_goals_conceded"] - df["goals_conceded"]
    df["mins_p90"]     = (df["minutes"] / 90).clip(lower=0.1)
    df["xG_p90"]       = df["expected_goals"] / df["mins_p90"]
    df["xA_p90"]       = df["expected_assists"] / df["mins_p90"]
    df["xGI_p90"]      = df["expected_goal_involvements"] / df["mins_p90"]
    df["goals_p90"]    = df["goals_scored"] / df["mins_p90"]
    df["assists_p90"]  = df["assists"] / df["mins_p90"]
    df["saves_p90"]    = df["saves"] / df["mins_p90"]
    df["tackles_p90"]  = df["tackles"] / df["mins_p90"]
    df["recoveries_p90"] = df["recoveries"] / df["mins_p90"]
    df["cbi_p90"]      = df["clearances_blocks_interceptions"] / df["mins_p90"]
    df["def_contribution_p90"] = df["defensive_contribution"] / df["mins_p90"]
    return df

def build_team_stats(dg_raw, team_id_map):
    dg = dg_raw.copy()
    for c in NUM_COLS_GW:
        dg[c] = pd.to_numeric(dg[c] if c in dg.columns
                              else pd.Series(0, index=dg.index), errors="coerce").fillna(0)
    dg["gf"] = np.where(dg["was_home"].fillna(False),
                        dg["team_h_score"], dg["team_a_score"])
    dg["ga"] = np.where(dg["was_home"].fillna(False),
                        dg["team_a_score"], dg["team_h_score"])

    # 得失点（マッチスコアから）
    ms = dg.dropna(subset=["gf","ga"]).groupby(["team","round"]).agg(
        gf=("gf","first"), ga=("ga","first")
    ).reset_index()
    scores = ms.groupby("team").agg(
        goals_scored=("gf","sum"),
        goals_conceded=("ga","sum"),
        matches=("round","nunique"),
    ).reset_index()

    # 攻撃指標（全選手集計）
    atk = dg.groupby("team").agg(
        xG=("expected_goals","sum"),
        xA=("expected_assists","sum"),
        creativity=("creativity","sum"),
        threat=("threat","sum"),
        influence=("influence","sum"),
        yellow_cards=("yellow_cards","sum"),
        red_cards=("red_cards","sum"),
        bonus=("bonus","sum"),
        assists=("assists","sum"),
        tackles=("tackles","sum"),
        recoveries=("recoveries","sum"),
        cbi=("clearances_blocks_interceptions","sum"),
        def_contribution=("defensive_contribution","sum"),
    ).reset_index()

    # GKからCS・xGC・Saves
    gk = dg[dg["position"]=="GK"].copy()
    gk_agg = gk.groupby("team").agg(
        xGC=("expected_goals_conceded","sum"),
        clean_sheets=("clean_sheets","sum"),
        saves=("saves","sum"),
    ).reset_index()

    team = scores.merge(atk, on="team").merge(gk_agg, on="team", how="left")
    team[["xGC","clean_sheets","saves"]] = team[["xGC","clean_sheets","saves"]].fillna(0)

    m = team["matches"].clip(lower=1)
    team["xG_per_match"]  = (team["xG"]  / m).round(2)
    team["xGC_per_match"] = (team["xGC"] / m).round(2)
    team["gf_per_match"]  = (team["goals_scored"] / m).round(2)
    team["ga_per_match"]  = (team["goals_conceded"] / m).round(2)
    team["xG_diff"]       = (team["xG"] - team["xGC"]).round(2)
    team["goal_diff"]     = team["goals_scored"] - team["goals_conceded"]
    team["goal_luck"]     = (team["goals_scored"] - team["xG"]).round(2)
    team["def_luck"]      = (team["xGC"] - team["goals_conceded"]).round(2)
    team["cs_per_match"]  = (team["clean_sheets"] / m).round(2)
    team["saves_per_match"] = (team["saves"] / m).round(2)

    return team.rename(columns={"team": "team_name"})

# ── Matplotlib style ──────────────────────────────────────────────────────────
def apply_dark_style(fig, axes=None):
    """白背景・濃いテキストの高コントラストスタイル"""
    fig.patch.set_facecolor("#f5f7fa")
    if axes is None:
        axes = fig.get_axes()
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    for ax in axes:
        ax.set_facecolor("#ffffff")
        ax.tick_params(colors="#333333", labelsize=8)
        ax.xaxis.label.set_color("#333333")
        ax.yaxis.label.set_color("#333333")
        ax.title.set_color("#1a1a2e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")
        ax.grid(True, color="#e5e7eb", linewidth=0.5, alpha=0.8)
    return fig

PITCH_COLORS = [
    "#48cae4","#f4a261","#22c55e","#a78bfa","#f87171",
    "#34d399","#fb923c","#60a5fa","#e879f9","#fbbf24",
    "#4ade80","#38bdf8","#c084fc","#fb7185","#a3e635",
    "#2dd4bf","#f472b6","#818cf8","#facc15","#6ee7b7",
]

def team_color_map(teams):
    return {t: PITCH_COLORS[i % len(PITCH_COLORS)] for i, t in enumerate(sorted(teams))}

# ── Chart helpers ─────────────────────────────────────────────────────────────
def radar(df_sel, metrics, labels, title, z_pool=None):
    """Dark-themed percentile radar chart"""
    if z_pool is None:
        pool = df_sel
    else:
        pool = z_pool
    df_pct = df_sel[metrics].copy()
    for col in metrics:
        pv = pool[col].dropna()
        df_pct[col] = df_pct[col].apply(
            lambda v: float((pv <= v).mean()) if len(pv) else 0.5
        )
    n      = len(labels)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#f5f7fa")
    ax.set_facecolor("#ffffff")
    ax.spines["polar"].set_edgecolor("#cccccc")
    ax.grid(color="#e5e7eb", lw=.6)
    palette = PITCH_COLORS
    patches = []
    for i, (idx, row) in enumerate(df_pct.iterrows()):
        c    = palette[i % len(palette)]
        vals = row[metrics].tolist() + [row[metrics[0]]]
        name = str(idx)[:20]
        ax.plot(angles, vals, "o-", lw=2, color=c, alpha=.9)
        ax.fill(angles, vals, alpha=.12, color=c)
        patches.append(mpatches.Patch(color=c, label=name))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=8, color="#1a1a2e", fontweight="bold")
    ax.set_ylim(0,1)
    ax.set_yticks([.25,.5,.75])
    ax.set_yticklabels(["25%","50%","75%"], size=7, color="#555555")
    ax.tick_params(pad=10)
    ax.set_title(title, size=10, color=C["pitch"], pad=20, fontweight="bold")
    ax.legend(handles=patches, loc="upper right",
              bbox_to_anchor=(1.5, 1.15), fontsize=8,
              facecolor="#f5f7fa", edgecolor="#cccccc", labelcolor="#1a1a2e")
    plt.tight_layout()
    return fig

def scatter_2d(df, x_col, y_col, label_col, title, c_map=None):
    fig, ax = plt.subplots(figsize=(8,6))
    apply_dark_style(fig, ax)
    teams = df[label_col].tolist()
    colors = [c_map.get(t, C["sky"]) for t in teams] if c_map else [C["sky"]]*len(teams)
    ax.scatter(df[x_col], df[y_col], c=colors, s=100, alpha=.9, edgecolors="#374151", lw=.5, zorder=3)
    # 平均線
    ax.axhline(df[y_col].mean(), color="#374151", ls="--", lw=.8, zorder=1)
    ax.axvline(df[x_col].mean(), color="#374151", ls="--", lw=.8, zorder=1)
    # ラベル
    for _, row in df.iterrows():
        ax.annotate(row[label_col], (row[x_col], row[y_col]),
                    xytext=(5,5), textcoords="offset points",
                    fontsize=7.5, color=C["chalk"], fontweight="600")
    ax.set_xlabel(x_col.replace("_"," ").title(), color=C["muted"])
    ax.set_ylabel(y_col.replace("_"," ").title(), color=C["muted"])
    ax.set_title(title, color=C["chalk"], fontweight="bold")
    plt.tight_layout()
    return fig

def pca_plot(df, metrics, label_col, title, c_map=None):
    from numpy.linalg import svd
    X = df[metrics].fillna(0).values.astype(float)
    X_z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    _, _, Vt = svd(X_z, full_matrices=False)
    scores = X_z @ Vt[:2].T
    loadings = Vt[:2].T

    fig = plt.figure(figsize=(10, 5))
    apply_dark_style(fig)
    gs = GridSpec(1, 2, figure=fig, wspace=.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    apply_dark_style(fig, [ax1, ax2])

    # Scatterplot
    teams = df[label_col].tolist()
    colors = [c_map.get(t, C["sky"]) for t in teams] if c_map else [C["sky"]]*len(df)
    ax1.scatter(scores[:,0], scores[:,1], c=colors, s=90, alpha=.9,
                edgecolors="#374151", lw=.5, zorder=3)
    ax1.axhline(0, color="#374151", ls="--", lw=.7)
    ax1.axvline(0, color="#374151", ls="--", lw=.7)
    for i, t in enumerate(teams):
        ax1.annotate(t, (scores[i,0], scores[i,1]),
                     xytext=(5,5), textcoords="offset points",
                     fontsize=7, color=C["chalk"], fontweight="600")
    ax1.set_xlabel("PC 1", color=C["muted"])
    ax1.set_ylabel("PC 2", color=C["muted"])
    ax1.set_title(title, color=C["chalk"], fontweight="bold")

    # Loading bar chart  ← loadings shape: (n_metrics, 2)
    load_df = pd.DataFrame(loadings, index=metrics, columns=["PC1","PC2"]).reset_index()
    load_df.columns = ["metric","PC1","PC2"]
    x_pos = np.arange(len(metrics))
    w = .35
    ax2.barh(x_pos + w/2, load_df["PC1"], w, label="PC 1",
             color=C["amber"], alpha=.85)
    ax2.barh(x_pos - w/2, load_df["PC2"], w, label="PC 2",
             color=C["sky"], alpha=.85)
    ax2.set_yticks(x_pos)
    ax2.set_yticklabels([m.replace("_"," ") for m in metrics],
                        fontsize=8, color=C["chalk"])
    ax2.axvline(0, color="#374151", lw=.7)
    ax2.set_xlabel("Loading", color=C["muted"])
    ax2.set_title("Loadings — which metrics drive each PC",
                  color=C["chalk"], fontweight="bold")
    ax2.legend(facecolor="#1f2937", edgecolor="#374151",
               labelcolor=C["chalk"], fontsize=9)
    plt.tight_layout()
    return fig, scores, loadings, Vt

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown(f"""
<div style="padding:.8rem 0 .3rem">
  <span style="font-family:'Bebas Neue',sans-serif;font-size:1.5rem;
    color:{C['amber']};letter-spacing:.06em">⚽ EPL Analytics</span>
  <div style="color:{C['muted']};font-size:.72rem;margin-top:.1rem">
    Custom Metrics Builder
  </div>
</div>
""", unsafe_allow_html=True)

season = st.sidebar.selectbox("Season", ["2025-26","2024-25","2023-24","2022-23"])
page   = st.sidebar.radio("", ["🏟️ Team Analysis","👤 Player Analysis"], label_visibility="collapsed")
st.sidebar.markdown("---")
# APIキーは st.secrets から読む（公開アプリ用・入力欄なし）
_raw_key = ""
try:
    _raw_key = st.secrets.get("APIFOOTBALL_KEY", "")
except Exception:
    pass
api_key_input = _raw_key.strip()
apf_enabled   = bool(api_key_input)
apf_remain    = 0   # デフォルト値

st.sidebar.markdown("**⚡ API-Football**")
if apf_enabled:
    apf_valid, apf_used, apf_remain = check_apf_key(api_key_input)
    if apf_valid:
        st.sidebar.success(f"✅ 接続済み  残: {apf_remain}req/日")
    else:
        st.sidebar.error("❌ APIキー無効")
        apf_enabled = False
        apf_remain  = 0
else:
    st.sidebar.caption("⚡指標: Streamlit Secrets に APIFOOTBALL_KEY を設定してください")
st.sidebar.markdown("---")

# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner("Loading data..."):
    df_p_raw, df_g_raw, df_t_raw = load_season(season)

if df_p_raw is None or df_g_raw is None:
    st.error(f"""
**データ取得失敗** — 以下のURLをブラウザで開き、手動ダウンロードしてください:
```
{VAASTAV}/{season}/players_raw.csv  →  players_raw.csv として保存
{VAASTAV}/{season}/gws/merged_gw.csv  →  merged_gw.csv として保存
```
""")
    st.stop()

team_id_map = {}
if df_t_raw is not None and "id" in df_t_raw.columns and "name" in df_t_raw.columns:
    team_id_map = dict(zip(df_t_raw["id"], df_t_raw["name"]))

df_players  = prep_players(df_p_raw, team_id_map)
df_teams    = build_team_stats(df_g_raw, team_id_map)

# API-Football データの取得・マージ
if apf_enabled:
    import json as _j
    fixture_ids = fetch_finished_fixture_ids(season, api_key_input)
    n_total     = len(fixture_ids)
    cached_dict = _load_apf_cache()
    n_cached = len([k for k in cached_dict if str(k) in [str(f) for f in fixture_ids]])

    # 取得状況をサイドバーに表示
    if n_total == 0:
        st.sidebar.warning("⚠️ 試合IDを取得できません。APIキーと接続を確認してください")
        st.sidebar.caption(f"season={season}, league={EPL_LEAGUE}")
    else:
        st.sidebar.markdown(f"⚡ **{n_cached}/{n_total}試合 取得済み**",
                            unsafe_allow_html=True)

    # 手動取得ボタン
    fetch_col1, fetch_col2 = st.sidebar.columns([2,1])
    fetch_n = fetch_col1.number_input("取得数", 10, 95, 80, 10, key="fetch_n",
                                       label_visibility="collapsed")
    do_fetch = fetch_col2.button("📡 取得", key="do_fetch",
                                  help=f"未取得の試合を最大{int(fetch_n)}件取得します")

    # キャッシュクリアボタン（試合ID再取得用）
    if st.sidebar.button("🔄 試合ID再取得", key="clear_fid",
                          help="キャッシュをクリアして試合IDを再取得します"):
        # session_state の fixture_ids キャッシュをクリア
        for k in list(st.session_state.keys()):
            if k.startswith("fixture_ids_"):
                del st.session_state[k]
        st.rerun()

    if do_fetch and n_total > 0 and apf_remain > 3:
        fixture_cache = fetch_and_cache_stats(
            fixture_ids, api_key_input,
            max_per_run=min(int(fetch_n), apf_remain - 2)
        )
    else:
        fixture_cache = {int(k): v for k, v in cached_dict.items()
                         if int(k) in fixture_ids}

    if fixture_cache:
        df_teams = build_apf_team_stats(fixture_cache, df_teams)

tcmap       = team_color_map(df_teams["team_name"].tolist())

# ── Header ────────────────────────────────────────────────────────────────────
_view_label = "Team" if "Team" in page else "Player"
_subtitle   = f"{season}  ·  Custom Metrics Builder  ·  {_view_label} View"

# SVGヘッダー（CSSに依存しないため確実に表示）
_svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="900" height="90" viewBox="0 0 900 90">
  <rect width="900" height="90" rx="10" fill="#0f172a"/>
  <rect width="6" height="90" rx="3" fill="#1a5c36"/>
  <text x="22" y="52" font-family="Arial Black,Arial,sans-serif"
        font-size="32" font-weight="900" letter-spacing="2"
        fill="#ffffff">EPL Analytics</text>
  <text x="22" y="74" font-family="Arial,sans-serif"
        font-size="11" font-weight="600" fill="#94a3b8">{_subtitle}</text>
</svg>"""
import base64 as _b64
_svg_b64 = _b64.b64encode(_svg.encode()).decode()
st.markdown(
    f'<img src="data:image/svg+xml;base64,{_svg_b64}" style="width:100%;display:block">',
    unsafe_allow_html=True
)
st.markdown("<div style='height:4px;background:linear-gradient(90deg,#1a5c36,#c45c00,#0077aa,transparent);border-radius:2px;margin-bottom:1rem'></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  TEAM ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
if "Team" in page:

    # ── 利用可能指標一覧 ──────────────────────────────────────────────────────
    TEAM_METRICS = {
        # label: (col, description, category)
        "Goals Scored":       ("goals_scored",   "Total goals scored",               "Attack"),
        "Goals Conceded":     ("goals_conceded", "Total goals conceded",              "Defense"),
        "Goal Difference":    ("goal_diff",      "Goals scored − Goals conceded",     "Attack"),
        "xG (Total)":         ("xG",             "Expected goals scored",             "Attack"),
        "xGC (Total)":        ("xGC",            "Expected goals conceded",           "Defense"),
        "xG per Match":       ("xG_per_match",   "xG / matches played",               "Attack"),
        "xGC per Match":      ("xGC_per_match",  "xGC / matches played",              "Defense"),
        "xG Difference":      ("xG_diff",        "xG − xGC",                          "Attack"),
        "Goal Luck (Attack)": ("goal_luck",      "Goals − xG (positive = clinical)",  "Luck"),
        "Def Luck":           ("def_luck",       "xGC − GA (positive = fortunate)",   "Luck"),
        "Clean Sheets":       ("clean_sheets",   "Number of clean sheets",            "Defense"),
        "CS per Match":       ("cs_per_match",   "Clean sheets / matches",            "Defense"),
        "Saves":              ("saves",          "Total saves by GK",                 "Defense"),
        "Saves per Match":    ("saves_per_match","Saves / matches",                   "Defense"),
        "GF per Match":       ("gf_per_match",   "Goals scored / matches",            "Attack"),
        "GA per Match":       ("ga_per_match",   "Goals conceded / matches",          "Defense"),
        "Assists":            ("assists",        "Total assists",                      "Attack"),
        "xA (Total)":         ("xA",             "Expected assists",                   "Attack"),
        "Creativity":         ("creativity",     "FPL Creativity (chance creation)",   "Attack"),
        "Threat":             ("threat",         "FPL Threat (goal threat)",           "Attack"),
        "Yellow Cards":       ("yellow_cards",   "Yellow cards accumulated",           "Discipline"),
        "Red Cards":          ("red_cards",      "Red cards accumulated",              "Discipline"),
    "Tackles (team)":     ("tackles",        "Total team tackles",                 "Defense"),
    "Recoveries (team)":  ("recoveries",     "Total team ball recoveries",         "Defense"),
    "CBI (team)":         ("cbi",            "Team clearances+blocks+interceptions","Defense"),
    "Def Contribution":   ("def_contribution","Team defensive contribution score", "Defense"),
    # ── API-Football 取得指標（要APIキー）──────────────────────────────────────
    "Shots/Match ⚡":     ("shots_pm",        "Total shots per match (API-Football)", "Attack⚡"),
    "Shots on Target/M ⚡":("shots_on_tgt_pm","Shots on target per match",            "Attack⚡"),
    "Shots in Box/M ⚡":  ("shots_inbox_pm",  "Shots inside box per match",           "Attack⚡"),
    "Shot Conversion % ⚡":("shot_conversion", "Goals / Shots × 100",                 "Attack⚡"),
    "Possession % ⚡":    ("possession",       "Average ball possession (%)",          "Attack⚡"),
    "Corners/Match ⚡":   ("corners_pm",       "Corners per match",                   "Attack⚡"),
    "Fouls/Match ⚡":     ("fouls_pm",         "Fouls committed per match",           "Discipline⚡"),
    "Offsides/Match ⚡":  ("offsides_pm",      "Offsides per match",                  "Attack⚡"),
    "Pass Accuracy % ⚡": ("pass_acc",         "Pass completion rate (%)",             "Attack⚡"),
    "Passes/Match ⚡":    ("passes_pm",        "Total passes per match",               "Attack⚡"),
    }
    all_metric_labels = list(TEAM_METRICS.keys())
    metric_cols = {v[0]: k for k, v in TEAM_METRICS.items()}

    tab_overview, tab_radar, tab_scatter, tab_pca, tab_custom = st.tabs([
        "📋 Available Metrics",
        "🕸️ Radar",
        "⊕ 2-Axis Plot",
        "📐 PCA",
        "🔧 Custom Metric",
    ])

    # ── Tab 0: Available Metrics ──────────────────────────────────────────────
    with tab_overview:
        st.markdown("## Available Team Metrics")
        st.markdown("<div class='section-bar'></div>", unsafe_allow_html=True)

        # 指標の読み方ガイド
        with st.expander("📖 指標の読み方・使い方ガイド", expanded=False):
            st.markdown("""
            #### FPL独自指標（ICT）の意味
            | 指標 | 意味 | 高い選手の特徴 |
            |------|------|--------------|
            | **Creativity** | チャンスメイクの量と質。キーパス・クロス・スルーパスを評価 | トップ下、クリエイティブなMF（例: デ・ブライネ） |
            | **Threat** | ゴールへの脅威。シュート数・シュート位置・PA内行動を評価 | ストライカー、得点力が高いFW（例: サラー） |
            | **Influence** | 試合全体への関与度。ボールタッチ・デュエル・守備行動を包括的に評価 | キャプテン的存在、試合を動かすMF |
            | **ICT Index** | Influence + Creativity + Threat の合成スコア。FPLの総合評価 | 全能型MF・FW（例: サラー、デ・ブライネ） |
            | **xG (Expected Goals)** | そのシュートがゴールになる統計的確率の合計。「本来の得点力」を示す | — |
            | **xA (Expected Assists)** | アシストパスのxG合計。「パスの質」を示す | — |
            | **xGI** | xG + xA の合計。攻撃関与度の総合値 | — |
            | **xGC** | 被xG。出場中に相手に与えた得点期待値。低いほど守備が良い | — |

            #### 使い方のヒント
            - **レーダーチャート**: 複数チームを同じ指標で比較。標準化済みなので公平な比較ができます
            - **2軸散布図**: 例）「xG per Match」vs「xGC per Match」→ 攻守バランスの把握
            - **PCA**: 複数指標を入れると、隠れた「プレースタイル軸」を発見できます
            - **カスタム指標**: 例）「攻撃力 = xG×1.5 + Shots×0.5 - xGC×1.0」のように自由に設計
            - **⚡マーク指標**: API-Footballから取得。Streamlit SecretsにAPIキーを設定すると使えます
            """)

        cats = sorted(set(v[2] for v in TEAM_METRICS.values()))
        for cat in cats:
            st.markdown(f"### {cat}")
            rows = [(lbl, TEAM_METRICS[lbl][0], TEAM_METRICS[lbl][1])
                    for lbl in TEAM_METRICS if TEAM_METRICS[lbl][2] == cat]
            df_cat = pd.DataFrame(rows, columns=["Metric","Column","Description"])
            df_cat["Min"]  = df_cat["Column"].apply(lambda c: f"{df_teams[c].min():.2f}" if c in df_teams else "—")
            df_cat["Max"]  = df_cat["Column"].apply(lambda c: f"{df_teams[c].max():.2f}" if c in df_teams else "—")
            df_cat["Mean"] = df_cat["Column"].apply(lambda c: f"{df_teams[c].mean():.2f}" if c in df_teams else "—")
            st.dataframe(df_cat, use_container_width=True, hide_index=True)

    # ── Tab 1: Radar ─────────────────────────────────────────────────────────
    with tab_radar:
        st.markdown("## Radar Chart — Team Comparison")
        st.markdown("<div class='section-bar'></div>", unsafe_allow_html=True)

        col_a, col_b = st.columns([1,2])
        with col_a:
            sel_teams   = st.multiselect("Select Teams (2–6)",
                                          sorted(df_teams["team_name"]),
                                          default=sorted(df_teams["team_name"])[:6])
            sel_metrics = st.multiselect("Select Metrics (3–8)",
                                          all_metric_labels,
                                          default=["xG per Match","xGC per Match",
                                                   "CS per Match","Saves per Match",
                                                   "Goal Luck (Attack)","Def Luck"])
        with col_b:
            if len(sel_teams) >= 2 and len(sel_metrics) >= 3:
                cols_sel = [TEAM_METRICS[m][0] for m in sel_metrics]
                df_r = df_teams[df_teams["team_name"].isin(sel_teams)].set_index("team_name")
                fig_r = radar(df_r, cols_sel, sel_metrics,
                              "Team Comparison Radar", z_pool=df_teams.set_index("team_name"))
                st.pyplot(fig_r, use_container_width=True)
                st.caption("Outer = higher EPL-wide percentile. Axes show position relative to all 20 clubs.")
            else:
                st.info("Select at least 2 teams and 3 metrics.")

    # ── Tab 2: 2-Axis Scatter ─────────────────────────────────────────────────
    with tab_scatter:
        st.markdown("## 2-Axis Club Positioning")
        st.markdown("<div class='section-bar'></div>", unsafe_allow_html=True)

        col_a, col_b = st.columns([1,3])
        with col_a:
            x_label = st.selectbox("X Axis", all_metric_labels,
                                   index=all_metric_labels.index("xG per Match"))
            y_label = st.selectbox("Y Axis", all_metric_labels,
                                   index=all_metric_labels.index("xGC per Match"))
        with col_b:
            x_col = TEAM_METRICS[x_label][0]
            y_col = TEAM_METRICS[y_label][0]
            fig_s = scatter_2d(df_teams, x_col, y_col, "team_name",
                               f"{x_label}  vs  {y_label}", tcmap)
            st.pyplot(fig_s, use_container_width=True)
            st.caption("Dashed lines = league average. Teams top-right lead on both axes.")

    # ── Tab 3: PCA ────────────────────────────────────────────────────────────
    with tab_pca:
        st.markdown("## Principal Component Analysis")
        st.markdown("<div class='section-bar'></div>", unsafe_allow_html=True)
        st.info("ℹ️ EPLは20チームのみのため、投入指標数は6以下を推奨します。それ以上では偶然の構造を拾いやすくなります。")

        sel_pca = st.multiselect("Metrics for PCA (3–6 recommended)",
                                  all_metric_labels,
                                  default=["xG per Match","xGC per Match","GF per Match",
                                           "GA per Match","CS per Match","Goal Luck (Attack)"])
        if len(sel_pca) >= 3:
            pca_cols = [TEAM_METRICS[m][0] for m in sel_pca]
            fig_pca, scores, loadings, Vt = pca_plot(
                df_teams, pca_cols, "team_name",
                "Club Positioning — PC1 vs PC2", tcmap
            )
            st.pyplot(fig_pca, use_container_width=True)

            # PC寄与度サマリー
            col1, col2 = st.columns(2)
            load_df = pd.DataFrame({"Metric": sel_pca,
                                    "PC1 Loading": loadings[:,0].round(3),
                                    "PC2 Loading": loadings[:,1].round(3)})
            load_df["PC1 Abs"] = load_df["PC1 Loading"].abs()
            load_df["PC2 Abs"] = load_df["PC2 Loading"].abs()

            with col1:
                st.markdown("**PC 1 — Top contributing metrics**")
                st.caption("High PC1 score → right on scatter plot")
                st.dataframe(load_df.sort_values("PC1 Abs",ascending=False)
                             [["Metric","PC1 Loading"]]
                             .style.background_gradient(cmap="RdYlGn",subset=["PC1 Loading"],vmin=-1,vmax=1)
                             .format({"PC1 Loading":"{:+.3f}"}),
                             hide_index=True, use_container_width=True)
            with col2:
                st.markdown("**PC 2 — Top contributing metrics**")
                st.caption("High PC2 score → top on scatter plot")
                st.dataframe(load_df.sort_values("PC2 Abs",ascending=False)
                             [["Metric","PC2 Loading"]]
                             .style.background_gradient(cmap="RdYlGn",subset=["PC2 Loading"],vmin=-1,vmax=1)
                             .format({"PC2 Loading":"{:+.3f}"}),
                             hide_index=True, use_container_width=True)

            # PC1スコアをランキング表示
            df_pca_out = df_teams[["team_name"] + pca_cols].copy()
            df_pca_out["PC1_score"] = scores[:,0].round(3)
            df_pca_out["PC2_score"] = scores[:,1].round(3)
            st.markdown("**Club scores on each principal component**")
            st.dataframe(
                df_pca_out[["team_name","PC1_score","PC2_score"]]
                .sort_values("PC1_score", ascending=False)
                .style.background_gradient(subset=["PC1_score","PC2_score"], cmap="RdYlGn"),
                use_container_width=True, hide_index=True
            )
        else:
            st.info("Select at least 3 metrics.")

    # ── Tab 4: Custom Metric ──────────────────────────────────────────────────
    with tab_custom:
        st.markdown("## 🔧 Build Your Own Metric")
        st.markdown("<div class='section-bar'></div>", unsafe_allow_html=True)
        st.markdown("""
        **使い方:** 指標を選んで係数を設定するだけでオリジナル指標が作れます。
        係数は手動で設定するか、「Correlation with Points」ボタンで勝ち点との相関から自動提案できます。
        """)

        metric_name = st.text_input("Metric Name", value="My Custom Metric")

        # 指標選択テーブル
        n_rows = st.number_input("Number of metrics to combine", 1, 8, 3)
        components = []
        col_h1, col_h2, col_h3, col_h4 = st.columns([3,2,2,3])
        col_h1.markdown("**Metric**")
        col_h2.markdown("**Weight**")
        col_h3.markdown("**Direction**")
        col_h4.markdown("**Formula**")

        for i in range(int(n_rows)):
            c1,c2,c3,c4 = st.columns([3,2,2,3])
            with c1:
                m = st.selectbox(f"Metric {i+1}", all_metric_labels,
                                  key=f"cm_{i}",
                                  index=min(i, len(all_metric_labels)-1))
            with c2:
                w = st.number_input("Weight", value=1.0, step=0.1, key=f"cw_{i}")
            with c3:
                d = st.selectbox("", ["+", "−"], key=f"cd_{i}")
            with c4:
                sign = 1 if d == "+" else -1
                col = TEAM_METRICS[m][0]
                st.markdown(f"<span style='color:{C['muted']};font-size:.8rem'>"
                            f"`{'+' if sign>0 else '-'}{abs(w):.1f} × {col}`</span>",
                            unsafe_allow_html=True)
            components.append((m, col, w, sign))

        # 相関ベース自動係数
        if st.button("🔄 Auto-weight by correlation with xG Difference"):
            for i, (m, col, w, sign) in enumerate(components):
                if col in df_teams.columns:
                    r, _ = pearsonr(df_teams[col].fillna(0), df_teams["xG_diff"].fillna(0))
                    st.write(f"  {m}: r = {r:+.3f} → suggested weight = {abs(r):.2f} "
                             f"({'+'  if r > 0 else '−'})")

        # 計算・表示
        if st.button("▶  Calculate & Rank", type="primary"):
            df_custom = df_teams[["team_name"] + [c[1] for c in components]].copy()
            df_custom[metric_name] = sum(
                row[2] * row[3] * df_custom[row[1]].fillna(0)
                for row in components
            )
            df_custom = df_custom.sort_values(metric_name, ascending=False).reset_index(drop=True)
            df_custom.index += 1

            col_t, col_c = st.columns([2,3])
            with col_t:
                st.markdown(f"**Rankings: {metric_name}**")
                show_cols = ["team_name", metric_name] + [c[1] for c in components]
                st.dataframe(
                    df_custom[show_cols].style
                    .background_gradient(subset=[metric_name], cmap="RdYlGn")
                    .format({metric_name: "{:.3f}"}),
                    use_container_width=True
                )
            with col_c:
                # 簡易棒グラフ
                fig_bar, ax_bar = plt.subplots(figsize=(6,6))
                apply_dark_style(fig_bar, ax_bar)
                colors_bar = [tcmap.get(t, C["sky"]) for t in df_custom["team_name"]]
                ax_bar.barh(df_custom["team_name"][::-1],
                            df_custom[metric_name][::-1],
                            color=colors_bar[::-1], edgecolor="#374151", lw=.4)
                ax_bar.axvline(df_custom[metric_name].mean(),
                               color=C["amber"], ls="--", lw=1, label="Average")
                ax_bar.set_xlabel(metric_name, color=C["muted"])
                ax_bar.set_title(f"Club Ranking: {metric_name}",
                                 color=C["chalk"], fontweight="bold")
                ax_bar.legend(facecolor="#1f2937", edgecolor="#374151",
                              labelcolor=C["chalk"])
                plt.tight_layout()
                st.pyplot(fig_bar, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PLAYER ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
else:
    PLAYER_METRICS = {
        "Goals":          ("goals_scored",                 "Total goals",                "Attack"),
        "Assists":        ("assists",                      "Total assists",              "Attack"),
        "xG":             ("expected_goals",               "Expected goals",             "Attack"),
        "xA":             ("expected_assists",             "Expected assists",           "Attack"),
        "xGI":            ("expected_goal_involvements",   "xG + xA",                   "Attack"),
        "xG p90":         ("xG_p90",                      "xG per 90 mins",             "Attack"),
        "xA p90":         ("xA_p90",                      "xA per 90 mins",             "Attack"),
        "xGI p90":        ("xGI_p90",                     "xGI per 90 mins",            "Attack"),
        "Goals p90":      ("goals_p90",                   "Goals per 90 mins",          "Attack"),
        "Assists p90":    ("assists_p90",                  "Assists per 90 mins",        "Attack"),
        "Threat":         ("threat",                       "FPL Threat score",           "Attack"),
        "Creativity":     ("creativity",                   "FPL Creativity score",       "Playmaking"),
        "Influence":      ("influence",                    "FPL Influence score",        "Playmaking"),
        "ICT Index":      ("ict_index",                    "FPL ICT combined",           "Playmaking"),
        "xGC":            ("expected_goals_conceded",      "Expected goals conceded",    "Defense"),
        "Saves":          ("saves",                        "Total saves (GK)",           "Defense"),
        "Saves p90":      ("saves_p90",                   "Saves per 90 (GK)",          "Defense"),
        "Clean Sheets":   ("clean_sheets",                 "Clean sheets",               "Defense"),
        "Goals Conceded": ("goals_conceded",               "Goals conceded while on",    "Defense"),
        "Goal Luck":      ("goal_luck",                    "Goals − xG",                 "Luck"),
        "Def Luck":       ("def_luck",                     "xGC − Goals conceded",       "Luck"),
        "Minutes":        ("minutes",                      "Total minutes played",       "Availability"),
        "Starts":         ("starts",                       "Number of starts",           "Availability"),
        "Yellow Cards":   ("yellow_cards",                 "Yellow cards",               "Discipline"),
        "Red Cards":      ("red_cards",                    "Red cards",                  "Discipline"),
        "Tackles":        ("tackles",                      "Tackles (2025-26+)",          "Defense"),
        "Tackles p90":    ("tackles_p90",                  "Tackles per 90 mins",         "Defense"),
        "Recoveries":     ("recoveries",                   "Ball recoveries (2025-26+)",  "Defense"),
        "Recoveries p90": ("recoveries_p90",               "Recoveries per 90 mins",      "Defense"),
        "CBI":            ("clearances_blocks_interceptions","Clearances+Blocks+Interceptions","Defense"),
        "CBI p90":        ("cbi_p90",                      "CBI per 90 mins",             "Defense"),
        "Def Contribution":("defensive_contribution",      "Defensive contribution score","Defense"),
        "Def Contribution p90":("def_contribution_p90",   "Def contribution per 90",     "Defense"),
        "Bonus":          ("bonus",                        "FPL Bonus points",            "FPL"),
        "FPL Points":     ("total_points",                 "Total FPL points",            "FPL"),
        "Price (£M)":     ("price_m",                     "Current FPL price",           "FPL"),
    }
    all_player_labels = list(PLAYER_METRICS.keys())

    # サイドバーフィルター
    st.sidebar.markdown("**Player Filters**")
    pos_filter = st.sidebar.multiselect("Position", ["GK","DEF","MID","FWD"],
                                         default=["GK","DEF","MID","FWD"])
    team_filter = st.sidebar.multiselect("Team", sorted(df_players["team_name"].unique()),
                                          default=sorted(df_players["team_name"].unique()))
    min_min = st.sidebar.slider("Min minutes", 90, 3000, 450, 90)

    df_filt = df_players[
        df_players["position"].isin(pos_filter)
        & df_players["team_name"].isin(team_filter)
        & (df_players["minutes"] >= min_min)
    ].copy()
    st.sidebar.markdown(f"<div style='color:{C['muted']};font-size:.8rem'>"
                        f"Filtered players: <b style='color:{C['amber']}'>{len(df_filt)}</b></div>",
                        unsafe_allow_html=True)

    tab_avail, tab_top10, tab_prad, tab_pcap, tab_custp = st.tabs([
        "📋 Available Metrics",
        "🏆 Top 10 Rankings",
        "🕸️ Player Radar",
        "📐 Play Style (PCA)",
        "🔧 Custom Metric",
    ])

    with tab_avail:
        st.markdown("## Available Player Metrics")
        st.markdown("<div class='section-bar'></div>", unsafe_allow_html=True)

        with st.expander("📖 指標の読み方・使い方ガイド", expanded=False):
            st.markdown("""
            #### FPL独自指標（ICT）の意味
            | 指標 | 意味 | 参考 |
            |------|------|------|
            | **xG** | シュートのゴール期待値の合計。得点の実力値 | 得点数と比較するとLuck(運)がわかる |
            | **xA** | アシストパスのxG合計。パスの質の指標 | — |
            | **xGI / xGI p90** | xG+xA。攻撃への総関与度 | p90は90分あたりの値（出場時間補正） |
            | **xGC** | 出場中の被xG。守備貢献の反対側の指標（低いほど良い） | — |
            | **Creativity** | キーパス・クロス・スルーパス等のチャンスメイク量 | MF・攻撃的なDFが高い |
            | **Threat** | シュートの位置・数から算出するゴール脅威スコア | FW・得点力の高い選手が高い |
            | **Influence** | ボールタッチ・デュエル・守備を含む試合関与度の総合値 | 中盤のキーマンが高い |
            | **ICT Index** | Influence+Creativity+Threatの合成。FPL的な総合評価 | — |
            | **Goal Luck** | 実得点 − xG。プラスなら「xGより多く決めた」（好調/運) | — |
            | **Def Luck** | xGC − 失点。プラスなら「xGより少ない失点」（好守/運) | — |
            | **CBI** | クリアランス+ブロック+インターセプトの合計。守備行動量 | DF・守備的MFが高い |

            #### 使い方のヒント
            - **Top 10 Rankings**: ポジションフィルターと組み合わせて使うと効果的です（例: MIDのみでxA p90ランキング）
            - **Radar**: 同ポジション・同チームの選手を比較すると分かりやすいです
            - **Play Style PCA**: MIDのみに絞り攻撃指標を入れると「ボックストゥボックス vs アンカー」の軸が出ます
            - **p90指標**: 出場時間が異なる選手を公平に比較できます（最低出場分数フィルターと併用推奨）
            - **カスタム指標例**: 「攻撃貢献 = xG p90 × 2 + xA p90 × 1.5 + Creativity × 0.01」
            """)

        cats = sorted(set(v[2] for v in PLAYER_METRICS.values()))
        for cat in cats:
            st.markdown(f"### {cat}")
            rows = [(lbl, PLAYER_METRICS[lbl][0], PLAYER_METRICS[lbl][1])
                    for lbl in PLAYER_METRICS if PLAYER_METRICS[lbl][2] == cat]
            df_cat = pd.DataFrame(rows, columns=["Metric","Column","Description"])
            for stat_col, stat_lbl in [("Min","min"),("Max","max"),("Mean","mean")]:
                df_cat[stat_col] = df_cat["Column"].apply(
                    lambda c: f"{getattr(df_filt[c], stat_lbl)():.2f}"
                    if c in df_filt.columns else "—"
                )
            st.dataframe(df_cat, use_container_width=True, hide_index=True)

    with tab_top10:
        st.markdown("## Top 10 Rankings")
        st.markdown("<div class='section-bar'></div>", unsafe_allow_html=True)

        col_a, col_b = st.columns([1,3])
        with col_a:
            rank_metric = st.selectbox("Rank by", all_player_labels,
                                       index=all_player_labels.index("xGI p90"))
            show_n = st.radio("Show", [10, 20, 30], horizontal=True)
        with col_b:
            col_r = PLAYER_METRICS[rank_metric][0]
            if col_r not in df_filt.columns:
                st.warning(f"Column '{col_r}' not available.")
            else:
                # col_r が "minutes" 等の固定列と重複する場合の対処
                _base_cols = ["player_name","team_name","position","minutes"]
                _show_cols = _base_cols if col_r in _base_cols else _base_cols + [col_r]
                df_top = (df_filt[list(dict.fromkeys(_show_cols))]  # 重複除去
                          .sort_values(col_r, ascending=False)
                          .head(int(show_n))
                          .reset_index(drop=True))
                df_top.index += 1

                # パーセンタイル追加
                pool_vals = df_filt[col_r].dropna()
                df_top["Percentile"] = df_top[col_r].apply(
                    lambda v: f"Top {100-int((pool_vals<=v).mean()*100)}%"
                )
                pos_color = {"GK":"#F59E0B","DEF":"#3B82F6","MID":"#8B5CF6","FWD":"#EF4444"}
                def pos_style(v):
                    c = pos_color.get(v,"#6b7280")
                    return f"background:{c};color:white;font-weight:700;border-radius:4px;text-align:center"
                df_top_show = df_top.rename(columns={col_r: rank_metric})
                styled = (df_top_show.style
                          .background_gradient(subset=[rank_metric], cmap="RdYlGn")
                          .format({rank_metric: "{:.3f}"}))
                # pandas 3.x は .map()、旧版は .applymap()
                try:
                    styled = styled.map(pos_style, subset=["position"])
                except AttributeError:
                    try:
                        styled = styled.applymap(pos_style, subset=["position"])
                    except Exception:
                        pass
                st.dataframe(styled, use_container_width=True)

    with tab_prad:
        st.markdown("## Player Radar")
        st.markdown("<div class='section-bar'></div>", unsafe_allow_html=True)
        col_a, col_b = st.columns([1,2])
        with col_a:
            all_p = sorted(df_filt["player_name"].tolist())
            sel_p = st.multiselect("Select Players (2–5)", all_p,
                                    default=all_p[:3] if len(all_p) >= 3 else all_p)
            sel_pm = st.multiselect("Metrics (3–7)", all_player_labels,
                                     default=["xG p90","xA p90","Creativity",
                                              "Threat","Influence","Clean Sheets"])
        with col_b:
            if len(sel_p) >= 2 and len(sel_pm) >= 3:
                pm_cols = [PLAYER_METRICS[m][0] for m in sel_pm]
                df_pr = df_filt[df_filt["player_name"].isin(sel_p)].set_index("player_name")
                fig_pr = radar(df_pr, pm_cols, sel_pm, "Player Radar",
                               z_pool=df_filt.set_index("player_name"))
                st.pyplot(fig_pr, use_container_width=True)
                st.caption("Outer = higher percentile among all filtered players.")
            else:
                st.info("Select at least 2 players and 3 metrics.")

    with tab_pcap:
        st.markdown("## Play Style Analysis (PCA)")
        st.markdown("<div class='section-bar'></div>", unsafe_allow_html=True)
        st.info("**推奨:** 同ポジションで実施すると意味のある軸が出てきます。例: MFのみでPCA → ボックストゥボックス vs アンカー")

        pca_pos = st.multiselect("Position filter for PCA", ["GK","DEF","MID","FWD"],
                                  default=["MID"])
        df_pca_p = df_filt[df_filt["position"].isin(pca_pos)].copy()

        sel_pcap = st.multiselect("Metrics for PCA (4–8 recommended)", all_player_labels,
                                   default=["xG p90","xA p90","Creativity",
                                            "Threat","Influence","Saves p90"])
        if len(df_pca_p) >= 5 and len(sel_pcap) >= 3:
            pcap_cols = [PLAYER_METRICS[m][0] for m in sel_pcap]
            # PCA実行
            X = df_pca_p[pcap_cols].fillna(0).values.astype(float)
            X_z = (X - X.mean(0)) / (X.std(0) + 1e-9)
            from numpy.linalg import svd as npsvd
            _, _, Vt_p = npsvd(X_z, full_matrices=False)
            scores_p = X_z @ Vt_p[:2].T
            loadings_p = Vt_p[:2].T

            df_pca_p = df_pca_p.reset_index(drop=True).copy()
            df_pca_p["PC1"] = scores_p[:,0]
            df_pca_p["PC2"] = scores_p[:,1]

            # 散布図（選手名オーバーレイ）
            fig_pp, ax_pp = plt.subplots(figsize=(10,7))
            apply_dark_style(fig_pp, ax_pp)
            pcmap = {p: PITCH_COLORS[i % len(PITCH_COLORS)]
                     for i, p in enumerate(df_pca_p["team_name"].unique())}
            clrs_p = [pcmap.get(t, C["sky"]) for t in df_pca_p["team_name"]]
            ax_pp.scatter(df_pca_p["PC1"], df_pca_p["PC2"],
                          c=clrs_p, s=60, alpha=.8, edgecolors="#374151", lw=.4, zorder=3)
            ax_pp.axhline(0, color="#374151", ls="--", lw=.7)
            ax_pp.axvline(0, color="#374151", ls="--", lw=.7)
            for _, row in df_pca_p.iterrows():
                ax_pp.annotate(row["player_name"][:12],
                               (row["PC1"], row["PC2"]),
                               xytext=(3,3), textcoords="offset points",
                               fontsize=6.5, color=C["chalk"], alpha=.8)
            ax_pp.set_xlabel("PC 1", color=C["muted"])
            ax_pp.set_ylabel("PC 2", color=C["muted"])
            ax_pp.set_title(f"Play Style Map — {', '.join(pca_pos)}",
                            color=C["chalk"], fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_pp, use_container_width=True)

            # 寄与度 → プレースタイルラベル提案
            col1, col2 = st.columns(2)
            load_df_p = pd.DataFrame({"Metric": sel_pcap,
                                      "PC1 Loading": loadings_p[:,0].round(3),
                                      "PC2 Loading": loadings_p[:,1].round(3)})
            top_pc1 = load_df_p.reindex(load_df_p["PC1 Loading"].abs().sort_values(ascending=False).index).iloc[:3]
            top_pc2 = load_df_p.reindex(load_df_p["PC2 Loading"].abs().sort_values(ascending=False).index).iloc[:3]

            with col1:
                st.markdown("**PC1 — Style Axis Drivers**")
                st.dataframe(top_pc1[["Metric","PC1 Loading"]]
                             .style.background_gradient(cmap="RdYlGn",
                                                        subset=["PC1 Loading"],vmin=-1,vmax=1)
                             .format({"PC1 Loading":"{:+.3f}"}),
                             hide_index=True, use_container_width=True)
            with col2:
                st.markdown("**PC2 — Style Axis Drivers**")
                st.dataframe(top_pc2[["Metric","PC2 Loading"]]
                             .style.background_gradient(cmap="RdYlGn",
                                                        subset=["PC2 Loading"],vmin=-1,vmax=1)
                             .format({"PC2 Loading":"{:+.3f}"}),
                             hide_index=True, use_container_width=True)

            # TOP/BOT10
            st.markdown("**PC1 extremes (play style contrast)**")
            c1,c2 = st.columns(2)
            with c1:
                st.markdown(f"<span class='pill pill-amber'>High PC1 →</span>", unsafe_allow_html=True)
                st.dataframe(df_pca_p.nlargest(5,"PC1")[["player_name","team_name","PC1"]]
                             .style.format({"PC1":"{:+.2f}"}), hide_index=True)
            with c2:
                st.markdown(f"<span class='pill pill-sky'>Low PC1 →</span>", unsafe_allow_html=True)
                st.dataframe(df_pca_p.nsmallest(5,"PC1")[["player_name","team_name","PC1"]]
                             .style.format({"PC1":"{:+.2f}"}), hide_index=True)
        else:
            st.info("5人以上の選手と3指標以上を選択してください。")

    with tab_custp:
        st.markdown("## 🔧 Build Your Own Player Metric")
        st.markdown("<div class='section-bar'></div>", unsafe_allow_html=True)

        pmetric_name = st.text_input("Metric Name", value="My Player Score")
        pn_rows = st.number_input("Number of metrics", 1, 8, 3, key="pn")
        pcomps = []
        st.columns([3,2,2,3])[0].markdown("**Metric**")

        for i in range(int(pn_rows)):
            c1,c2,c3,_ = st.columns([3,2,2,3])
            with c1:
                m = st.selectbox(f"Metric {i+1}", all_player_labels,
                                  key=f"pm_{i}",
                                  index=min(i, len(all_player_labels)-1))
            with c2:
                w = st.number_input("Weight", value=1.0, step=0.1, key=f"pw_{i}")
            with c3:
                d = st.selectbox("", ["+","−"], key=f"pd_{i}")
            pcomps.append((m, PLAYER_METRICS[m][0], w, 1 if d=="+" else -1))

        show_pn = st.radio("Show top N", [10,20,30], horizontal=True, key="pshow")

        if st.button("▶  Calculate & Rank", type="primary", key="pcalc"):
            df_pc = df_filt[["player_name","team_name","position","minutes"]
                            + [c[1] for c in pcomps]].copy()
            df_pc[pmetric_name] = sum(
                r[2]*r[3]*df_pc[r[1]].fillna(0) for r in pcomps
            )
            df_pc = df_pc.sort_values(pmetric_name, ascending=False).head(int(show_pn)).reset_index(drop=True)
            df_pc.index += 1
            # パーセンタイル
            full_scores = sum(r[2]*r[3]*df_filt[r[1]].fillna(0) for r in pcomps)
            df_pc["Percentile"] = df_pc[pmetric_name].apply(
                lambda v: f"Top {100-int((full_scores<=v).mean()*100)}%"
            )
            st.dataframe(df_pc.style
                         .background_gradient(subset=[pmetric_name], cmap="RdYlGn")
                         .format({pmetric_name:"{:.3f}"}),
                         use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:#0f172a;color:#94a3b8;font-size:.72rem;
     padding:.8rem 1.2rem;border-radius:8px;margin-top:2rem;text-align:center;
     border-top:2px solid #1a5c36">
  Data: <b style='color:#cbd5e1'>vaastav/Fantasy-Premier-League</b>
  (github.com/vaastav/Fantasy-Premier-League) ·
  FPL data © Premier League · Non-commercial personal use only<br>
  Built with Streamlit · Python · NumPy · Pandas
</div>
""", unsafe_allow_html=True)
