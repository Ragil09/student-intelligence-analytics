import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
#   PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Student Intelligence Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════
#   GLOBAL THEME — matches notebook exactly
# ═══════════════════════════════════════════════════════════════
PALETTE   = ["#00F5FF", "#FF6B35", "#7B2FBE", "#00D68F", "#FFD166"]
BG        = "#0A0E1A"
CARD      = "#111827"
CARD2     = "#1A2235"
TEXT      = "#E2E8F0"
ACCENT    = "#00F5FF"
BORDER    = "#2D3748"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    CARD,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   TEXT,
    "xtick.color":       TEXT,
    "ytick.color":       TEXT,
    "text.color":        TEXT,
    "grid.color":        "#1E2A3A",
    "grid.alpha":        0.6,
    "font.family":       "monospace",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "legend.facecolor":  CARD,
    "legend.edgecolor":  BORDER,
})

CLUSTER_NAMES = {
    0: "High Achievers",
    1: "Consistent Learners",
    2: "At-Risk Students",
    3: "Growth Potential"
}

# ═══════════════════════════════════════════════════════════════
#   CUSTOM CSS — full dark theme
# ═══════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

/* ── Base */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
    background-color: {BG} !important;
    color: {TEXT} !important;
    font-family: 'Rajdhani', sans-serif !important;
}}

[data-testid="stSidebar"] {{
    background-color: {CARD} !important;
    border-right: 1px solid {BORDER} !important;
}}

[data-testid="stSidebar"] * {{
    color: {TEXT} !important;
    font-family: 'Rajdhani', sans-serif !important;
}}

/* ── Header strip */
[data-testid="stHeader"] {{
    background-color: {BG} !important;
}}

/* ── Metric cards */
[data-testid="stMetric"] {{
    background: {CARD2} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 12px !important;
    padding: 18px 20px !important;
}}
[data-testid="stMetricLabel"] {{
    color: #94A3B8 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}
[data-testid="stMetricValue"] {{
    color: {ACCENT} !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
}}
[data-testid="stMetricDelta"] svg {{
    fill: #00D68F !important;
}}

/* ── Dataframe */
[data-testid="stDataFrame"], .stDataFrame {{
    background: {CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
}}
iframe {{
    background: {CARD} !important;
}}

/* ── Selectbox / widgets */
[data-testid="stSelectbox"] > div > div {{
    background: {CARD2} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    color: {TEXT} !important;
    font-family: 'Share Tech Mono', monospace !important;
}}
[data-testid="stSelectbox"] label {{
    color: {ACCENT} !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}

/* ── Multiselect */
[data-testid="stMultiSelect"] label {{
    color: {ACCENT} !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
}}
[data-testid="stMultiSelect"] > div > div {{
    background: {CARD2} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
}}

/* ── Slider */
[data-testid="stSlider"] label {{
    color: {ACCENT} !important;
    font-family: 'Share Tech Mono', monospace !important;
    text-transform: uppercase;
    font-size: 0.8rem !important;
}}
.stSlider [data-baseweb="slider"] div {{
    background: {ACCENT} !important;
}}

/* ── Tabs */
[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    background: {CARD} !important;
    border-bottom: 1px solid {BORDER} !important;
    border-radius: 10px 10px 0 0 !important;
    padding: 4px 8px 0;
    gap: 4px;
}}
[data-testid="stTabs"] [data-baseweb="tab"] {{
    background: transparent !important;
    color: #64748B !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.82rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-radius: 8px 8px 0 0 !important;
    padding: 8px 18px !important;
    border: none !important;
    transition: all 0.2s;
}}
[data-testid="stTabs"] [aria-selected="true"] {{
    background: {CARD2} !important;
    color: {ACCENT} !important;
    border-bottom: 2px solid {ACCENT} !important;
}}
[data-testid="stTabs"] [data-baseweb="tab-panel"] {{
    background: {CARD2} !important;
    border: 1px solid {BORDER} !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
    padding: 24px !important;
}}

/* ── Divider */
hr {{
    border-color: {BORDER} !important;
    margin: 32px 0 !important;
}}

/* ── Scrollbar */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {BG}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: #4A5568; }}

/* ── Section heading utility */
.section-head {{
    font-family: 'Share Tech Mono', monospace;
    color: {ACCENT};
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 4px;
}}
.card-box {{
    background: {CARD2};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#   HELPER — plotly layout preset
# ═══════════════════════════════════════════════════════════════
def plotly_dark_layout(fig, title="", height=420):
    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=CARD,
        font=dict(color=TEXT, family="Share Tech Mono, monospace", size=11),
        title=dict(text=title, x=0.5,
                   font=dict(size=15, color=ACCENT, family="Share Tech Mono, monospace")),
        height=height,
        margin=dict(l=24, r=24, t=48, b=24),
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
        legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1,
                    font=dict(color=TEXT, size=11)),
    )
    return fig


def mpl_fig(w=14, h=6):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD)
    return fig, ax


# ═══════════════════════════════════════════════════════════════
#   LOAD DATA
# ═══════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv("student_segmentation_result.csv")
    if "ClusterLabel" not in df.columns and "Cluster" in df.columns:
        df["ClusterLabel"] = df["Cluster"].map(CLUSTER_NAMES)
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("⚠️  `student_segmentation_result.csv` not found. Run the notebook first.")
    st.stop()


# ═══════════════════════════════════════════════════════════════
#   SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding: 24px 0 16px;'>
        <div style='font-family:"Share Tech Mono",monospace; font-size:1.6rem;
                    color:{ACCENT}; letter-spacing:0.15em;'>🎓 ANALYTICS</div>
        <div style='font-family:"Share Tech Mono",monospace; font-size:0.7rem;
                    color:#64748B; letter-spacing:0.1em; margin-top:4px;'>
            STUDENT INTELLIGENCE PLATFORM
        </div>
        <hr style='border-color:{BORDER}; margin: 16px 0;'>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-head">📌 Navigation</p>', unsafe_allow_html=True)
    page = st.radio("", [
        "🏠  Overview",
        "📊  Distribution EDA",
        "🤖  Cluster Analysis",
        "🔮  PCA Visualization",
        "🔍  Data Explorer",
    ], label_visibility="collapsed")

    st.markdown(f"<hr style='border-color:{BORDER}; margin:20px 0;'>", unsafe_allow_html=True)
    st.markdown('<p class="section-head">⚙️ Filters</p>', unsafe_allow_html=True)

    all_labels = sorted(df["ClusterLabel"].unique())
    selected_clusters = st.multiselect(
        "Cluster Filter",
        options=all_labels,
        default=all_labels
    )

    numeric_cols = ["StudyHours", "Attendance", "ExamScore",
                    "EngagementScore", "AcademicIndex"]
    color_by = st.selectbox("Color Axis (Scatter)", numeric_cols, index=2)

    st.markdown(f"<hr style='border-color:{BORDER}; margin:20px 0;'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-family:"Share Tech Mono",monospace; font-size:0.65rem;
                color:#475569; line-height:1.8; text-align:center;'>
        Python · Pandas · Scikit-Learn<br>
        Plotly · Seaborn · Streamlit<br>
        K-Means · PCA · Silhouette
    </div>
    """, unsafe_allow_html=True)

# Apply cluster filter
dff = df[df["ClusterLabel"].isin(selected_clusters)]

# ═══════════════════════════════════════════════════════════════
#   HEADER BANNER
# ═══════════════════════════════════════════════════════════════
st.markdown(f"""
<div style='background: linear-gradient(135deg, {CARD} 0%, #0F1D35 100%);
            border: 1px solid {BORDER}; border-radius: 16px;
            padding: 32px 36px; margin-bottom: 28px;
            box-shadow: 0 4px 32px rgba(0,245,255,0.07);'>
    <div style='display:flex; align-items:center; gap:20px;'>
        <div style='font-size:3rem;'>🎓</div>
        <div>
            <div style='font-family:"Share Tech Mono",monospace;
                        font-size:1.6rem; color:{ACCENT};
                        letter-spacing:0.1em; line-height:1.1;'>
                STUDENT INTELLIGENCE ANALYTICS
            </div>
            <div style='font-family:"Share Tech Mono",monospace;
                        font-size:0.78rem; color:#64748B;
                        letter-spacing:0.12em; margin-top:6px;'>
                UNSUPERVISED LEARNING — K-MEANS CLUSTERING + PCA
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#   PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════
if "Overview" in page:

    # KPI Row
    total   = len(dff)
    n_clust = dff["Cluster"].nunique()
    avg_score = dff["ExamScore"].mean()
    avg_eng   = dff["EngagementScore"].mean() if "EngagementScore" in dff else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Students",    f"{total:,}")
    c2.metric("Clusters",          f"{n_clust}")
    c3.metric("Avg Exam Score",    f"{avg_score:.1f}")
    c4.metric("Avg Engagement",    f"{avg_eng:.2f}")
    c5.metric("Features Used",     "11")

    st.markdown(f"<hr style='border-color:{BORDER}; margin:24px 0 20px;'>", unsafe_allow_html=True)

    # ── Cluster population + Exam score side by side
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<p class="section-head">📌 Cluster Population</p>', unsafe_allow_html=True)
        counts = dff["ClusterLabel"].value_counts().reset_index()
        counts.columns = ["ClusterLabel", "Count"]
        fig_pie = go.Figure(go.Pie(
            labels=counts["ClusterLabel"],
            values=counts["Count"],
            hole=0.45,
            marker=dict(colors=PALETTE, line=dict(color=BG, width=2)),
            textfont=dict(size=12, family="Share Tech Mono, monospace"),
        ))
        plotly_dark_layout(fig_pie, "Population Distribution", height=380)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_r:
        st.markdown('<p class="section-head">📈 Avg Exam Score by Cluster</p>', unsafe_allow_html=True)
        avg_by = dff.groupby("ClusterLabel")["ExamScore"].mean().reset_index()
        fig_bar = go.Figure(go.Bar(
            x=avg_by["ClusterLabel"], y=avg_by["ExamScore"],
            marker=dict(color=PALETTE, line=dict(color=BG, width=1)),
            text=avg_by["ExamScore"].round(2),
            textposition="outside",
            textfont=dict(color=TEXT, size=12),
        ))
        plotly_dark_layout(fig_bar, "Average Exam Score per Cluster", height=380)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown(f"<hr style='border-color:{BORDER}; margin:8px 0 20px;'>", unsafe_allow_html=True)

    # ── Cluster summary table
    st.markdown('<p class="section-head">📋 Cluster Intelligence Summary</p>', unsafe_allow_html=True)
    profile_cols = ["StudyHours", "Attendance", "ExamScore",
                    "EngagementScore", "AcademicIndex"]
    profile_cols = [c for c in profile_cols if c in dff.columns]
    summary = dff.groupby("ClusterLabel")[profile_cols].mean().round(2)
    summary["Population"] = dff.groupby("ClusterLabel").size()
    summary["Pct (%)"] = (summary["Population"] / len(dff) * 100).round(1)
    st.dataframe(summary.style
        .background_gradient(cmap="Blues", subset=profile_cols)
        .format("{:.2f}", subset=profile_cols),
        use_container_width=True, height=220)

    st.markdown(f"<hr style='border-color:{BORDER}; margin:24px 0 16px;'>", unsafe_allow_html=True)

    # ── Radar chart
    st.markdown('<p class="section-head">🕸️ Cluster DNA — Radar Profile</p>', unsafe_allow_html=True)
    radar_features = [c for c in ["StudyHours","Attendance","AssignmentCompletion",
                                   "Discussions","EngagementScore","ExamScore"] if c in dff.columns]
    cluster_means = dff.groupby("Cluster")[radar_features].mean()
    cluster_norm  = (cluster_means - cluster_means.min()) / (
        cluster_means.max() - cluster_means.min() + 1e-9)

    fig_radar = go.Figure()
    for cid in sorted(dff["Cluster"].unique()):
        if cid not in cluster_norm.index:
            continue
        vals = cluster_norm.loc[cid].tolist()
        vals += vals[:1]
        cats = radar_features + [radar_features[0]]
        lbl  = CLUSTER_NAMES.get(cid, f"Cluster {cid}")
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=cats,
            fill="toself", name=lbl,
            line=dict(color=PALETTE[cid % len(PALETTE)], width=2),
            fillcolor=PALETTE[cid % len(PALETTE)],
            opacity=0.28
        ))

    fig_radar.update_layout(
        polar=dict(
            bgcolor=CARD,
            radialaxis=dict(visible=True, range=[0,1], gridcolor=BORDER,
                            color="#64748B", tickfont=dict(size=9)),
            angularaxis=dict(gridcolor=BORDER, color=TEXT,
                             tickfont=dict(size=11, family="Share Tech Mono, monospace"))
        ),
        paper_bgcolor=BG,
        font=dict(color=TEXT, family="Share Tech Mono, monospace"),
        title=dict(text="Cluster DNA — Normalized Feature Radar",
                   x=0.5, font=dict(size=15, color=ACCENT)),
        legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1,
                    font=dict(color=TEXT)),
        height=460,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    st.plotly_chart(fig_radar, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
#   PAGE: DISTRIBUTION EDA
# ═══════════════════════════════════════════════════════════════
elif "Distribution" in page:

    st.markdown('<p class="section-head">📊 Distribution Analysis — Student Performance Metrics</p>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["  NUMERIC DISTRIBUTIONS  ", "  CATEGORICAL BREAKDOWN  ", "  CORRELATION MATRIX  "])

    with tab1:
        from scipy.stats import gaussian_kde
        num_cols = [c for c in ["StudyHours","Attendance","AssignmentCompletion",
                                 "OnlineCourses","Discussions","ExamScore"] if c in dff.columns]
        #cool_cmap = plt.cm.get_cmap("cool")
        cool_cmap = matplotlib.colormaps["cool"]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.patch.set_facecolor(BG)
        fig.suptitle("Distribution Analysis — Student Performance Metrics",
                     fontsize=14, fontweight="bold", color=ACCENT, y=1.01)

        for ax, col in zip(axes.flat, num_cols):
            ax.set_facecolor(CARD)
            data = dff[col].dropna()
            n, bins, patches = ax.hist(data, bins=25, edgecolor=BG, linewidth=0.5, alpha=0.85)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            col_vals = (bin_centers - bin_centers.min()) / (bin_centers.max() - bin_centers.min() + 1e-9)
            for c_val, p in zip(col_vals, patches):
                p.set_facecolor(cool_cmap(c_val))
            if len(data) > 1:
                kde = gaussian_kde(data)
                x_r = np.linspace(data.min(), data.max(), 200)
                ax2 = ax.twinx()
                ax2.plot(x_r, kde(x_r), color="#FF6B35", lw=2.5)
                ax2.set_yticks([])
                ax2.set_facecolor(CARD)
            ax.axvline(data.mean(), color="#FFD166", lw=1.5, linestyle="--", alpha=0.9)
            ax.set_title(col, fontsize=11, fontweight="bold", color=TEXT, pad=8)
            ax.tick_params(colors=TEXT, labelsize=8)
            ax.spines["bottom"].set_color(BORDER)
            ax.spines["left"].set_color(BORDER)

        plt.tight_layout()
        st.pyplot(fig)

    with tab2:
        cat_cols = [c for c in ["LearningStyle","Gender","Motivation","StressLevel",
                                  "Extracurricular","Internet"] if c in dff.columns]

        fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
        fig2.patch.set_facecolor(BG)
        fig2.suptitle("Categorical Breakdown — Student Profile Analysis",
                      fontsize=14, fontweight="bold", color=ACCENT, y=1.01)

        for ax, col in zip(axes2.flat, cat_cols):
            ax.set_facecolor(CARD)
            vc = dff[col].value_counts()
            bars = ax.barh(vc.index.astype(str), vc.values,
                           color=PALETTE[:len(vc)], edgecolor=BG, linewidth=1.2, height=0.65)
            for bar, val in zip(bars, vc.values):
                ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                        f" {val:,}", va="center", fontsize=9, color=TEXT, fontweight="bold")
            ax.set_title(col, fontsize=11, fontweight="bold", color=TEXT, pad=8)
            ax.tick_params(colors=TEXT, labelsize=9)
            ax.spines["bottom"].set_color(BORDER)
            ax.spines["left"].set_color(BORDER)
            ax.invert_yaxis()

        plt.tight_layout()
        st.pyplot(fig2)

    with tab3:
        numeric_df = dff.select_dtypes(include=np.number).drop(
            columns=[c for c in ["Cluster","PCA1","PCA2","PCA3"] if c in dff.columns], errors="ignore")
        corr = numeric_df.corr()

        cmap_div = LinearSegmentedColormap.from_list("custom", ["#7B2FBE", BG, "#00F5FF"], N=256)
        mask = np.triu(np.ones_like(corr, dtype=bool))

        fig3, ax3 = plt.subplots(figsize=(14, 11))
        fig3.patch.set_facecolor(BG)
        ax3.set_facecolor(CARD)
        sns.heatmap(corr, mask=mask, cmap=cmap_div, center=0,
                    annot=True, fmt=".2f", annot_kws={"size": 9, "color": TEXT},
                    linewidths=0.5, linecolor="#1E2A3A",
                    cbar_kws={"shrink": 0.8}, ax=ax3, vmin=-1, vmax=1)
        ax3.set_title("Feature Correlation Matrix", fontsize=15,
                      fontweight="bold", color=ACCENT, pad=15)
        ax3.tick_params(colors=TEXT, labelsize=9)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig3)


# ═══════════════════════════════════════════════════════════════
#   PAGE: CLUSTER ANALYSIS
# ═══════════════════════════════════════════════════════════════
elif "Cluster" in page:

    st.markdown('<p class="section-head">🤖 Cluster Analysis — K-Means Segmentation</p>', unsafe_allow_html=True)

    tab_a, tab_b, tab_c = st.tabs(["  FEATURE COMPARISON  ", "  BOX DISTRIBUTIONS  ", "  ELBOW + SILHOUETTE  "])

    with tab_a:
        compare_features = [c for c in ["ExamScore","StudyHours","Attendance",
                                          "EngagementScore","AcademicIndex"] if c in dff.columns]
        cluster_summary = dff.groupby("ClusterLabel")[compare_features].mean().reset_index()

        fig_grouped = go.Figure()
        for col, color in zip(compare_features, PALETTE):
            fig_grouped.add_trace(go.Bar(
                x=cluster_summary["ClusterLabel"],
                y=cluster_summary[col],
                name=col,
                marker_color=color,
                text=cluster_summary[col].round(2),
                textposition="outside",
                textfont=dict(size=11, color=TEXT),
            ))
        fig_grouped.update_layout(barmode="group")
        plotly_dark_layout(fig_grouped, "Cluster Feature Comparison", height=460)
        st.plotly_chart(fig_grouped, use_container_width=True)

    with tab_b:
        box_feats = [c for c in ["ExamScore","StudyHours","Attendance",
                                   "EngagementScore","AcademicIndex"] if c in dff.columns]
        fig_box, axes_box = plt.subplots(1, len(box_feats), figsize=(20, 7))
        fig_box.patch.set_facecolor(BG)
        fig_box.suptitle("Feature Distribution by Cluster",
                         fontsize=14, fontweight="bold", color=ACCENT, y=1.02)

        cluster_ids = sorted(dff["Cluster"].unique())
        for ax, feat in zip(axes_box, box_feats):
            ax.set_facecolor(CARD)
            data_groups = [dff[dff["Cluster"] == c][feat].values for c in cluster_ids]
            bp = ax.boxplot(data_groups, patch_artist=True,
                            medianprops=dict(color="white", lw=2.5),
                            whiskerprops=dict(color=BORDER),
                            capprops=dict(color=BORDER),
                            flierprops=dict(markerfacecolor=ACCENT, markersize=3, alpha=0.4))
            for patch, color in zip(bp["boxes"], PALETTE):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)
            ax.set_title(feat, fontsize=10, fontweight="bold", color=TEXT)
            ax.set_xticklabels([f"C{i}" for i in cluster_ids], color=TEXT, fontsize=9)
            ax.tick_params(colors=TEXT, labelsize=8)
            ax.spines["bottom"].set_color(BORDER)
            ax.spines["left"].set_color(BORDER)
            ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig_box)

    with tab_c:
        if "StudyHours" in df.columns:
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score

            feat_cols = [c for c in ["StudyHours","Attendance","AssignmentCompletion",
                                      "OnlineCourses","Discussions","Motivation",
                                      "StressLevel","LearningStyle","EngagementScore",
                                      "AcademicIndex","ExamScore"] if c in df.columns]
            numeric_only = df[feat_cols].select_dtypes(include=np.number)
            scaler = StandardScaler()
            scaled = scaler.fit_transform(numeric_only)

            with st.spinner("Computing Elbow & Silhouette curves…"):
                wcss, sil = [], []
                for k in range(2, 11):
                    km = KMeans(n_clusters=k, random_state=42, n_init=10)
                    lbs = km.fit_predict(scaled)
                    wcss.append(km.inertia_)
                    sil.append(silhouette_score(scaled, lbs))

            k_range = list(range(2, 11))

            col1, col2 = st.columns(2)
            with col1:
                fig_elbow = go.Figure()
                fig_elbow.add_trace(go.Scatter(
                    x=k_range, y=wcss, mode="lines+markers",
                    line=dict(color=ACCENT, width=2.5),
                    marker=dict(color="#FF6B35", size=9),
                    fill="tozeroy", fillcolor="rgba(0,245,255,0.06)",
                    name="WCSS"
                ))
                fig_elbow.add_vline(x=4, line_color="#FFD166", line_dash="dash", line_width=2,
                                    annotation_text="K=4", annotation_font_color="#FFD166")
                plotly_dark_layout(fig_elbow, "Elbow Method — WCSS", height=360)
                st.plotly_chart(fig_elbow, use_container_width=True)

            with col2:
                fig_sil = go.Figure(go.Bar(
                    x=k_range, y=sil,
                    marker_color=["#FF6B35" if k == 4 else ACCENT for k in k_range],
                    text=[f"{v:.3f}" for v in sil], textposition="outside",
                    textfont=dict(color=TEXT, size=11),
                ))
                fig_sil.add_vline(x=4, line_color="#FFD166", line_dash="dash", line_width=2,
                                   annotation_text="K=4", annotation_font_color="#FFD166")
                plotly_dark_layout(fig_sil, "Silhouette Score per K", height=360)
                st.plotly_chart(fig_sil, use_container_width=True)
        else:
            st.info("Elbow/Silhouette computation requires the original feature columns.")


# ═══════════════════════════════════════════════════════════════
#   PAGE: PCA VISUALIZATION
# ═══════════════════════════════════════════════════════════════
elif "PCA" in page:

    st.markdown('<p class="section-head">🔮 PCA Dimensionality Reduction</p>', unsafe_allow_html=True)

    has_pca = all(c in dff.columns for c in ["PCA1", "PCA2"])
    has_pca3 = "PCA3" in dff.columns

    tab_2d, tab_3d, tab_scree = st.tabs(["  2D CLUSTER MAP  ", "  3D INTERACTIVE  ", "  SCREE PLOT  "])

    with tab_2d:
        # matplotlib scatter styled like notebook
        fig_s, ax_s = plt.subplots(figsize=(14, 9))
        fig_s.patch.set_facecolor(BG)
        ax_s.set_facecolor(CARD)

        for cid, color in enumerate(PALETTE[:4]):
            mask_c = dff["Cluster"] == cid
            if not mask_c.any():
                continue
            sub = dff[mask_c]
            ax_s.scatter(sub["PCA1"], sub["PCA2"],
                         c=color, alpha=0.7, s=60,
                         edgecolors="white", linewidths=0.3,
                         label=CLUSTER_NAMES.get(cid, f"Cluster {cid}"))
            cx, cy = sub["PCA1"].mean(), sub["PCA2"].mean()
            ax_s.annotate(f"C{cid}", (cx, cy), fontsize=14, fontweight="bold",
                          color="white", ha="center", va="center",
                          bbox=dict(boxstyle="circle,pad=0.4", fc=color, alpha=0.9, ec="white"))

        ax_s.set_title("Student Cluster Map — PCA 2D Projection",
                       fontsize=15, fontweight="bold", color=ACCENT, pad=15)
        ax_s.set_xlabel("Principal Component 1", fontsize=12, color=TEXT)
        ax_s.set_ylabel("Principal Component 2", fontsize=12, color=TEXT)
        ax_s.legend(fontsize=11, facecolor=CARD, edgecolor=BORDER,
                    labelcolor=TEXT, loc="upper right")
        ax_s.grid(True, alpha=0.2)
        ax_s.spines["bottom"].set_color(BORDER)
        ax_s.spines["left"].set_color(BORDER)
        plt.tight_layout()
        st.pyplot(fig_s)

    with tab_3d:
        if has_pca3:
            fig_3d = px.scatter_3d(
                dff, x="PCA1", y="PCA2", z="PCA3",
                color="ClusterLabel",
                color_discrete_sequence=PALETTE,
                hover_data={c: True for c in ["ExamScore","StudyHours","Attendance"] if c in dff.columns},
                opacity=0.82,
                title="3D Student Segmentation Map"
            )
            fig_3d.update_layout(
                paper_bgcolor=BG,
                font=dict(color=TEXT, family="Share Tech Mono, monospace"),
                title=dict(x=0.5, font=dict(size=16, color=ACCENT)),
                scene=dict(
                    bgcolor=CARD,
                    xaxis=dict(backgroundcolor=CARD, gridcolor=BORDER),
                    yaxis=dict(backgroundcolor=CARD, gridcolor=BORDER),
                    zaxis=dict(backgroundcolor=CARD, gridcolor=BORDER),
                ),
                legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1,
                            font=dict(color=TEXT)),
                margin=dict(l=0, r=0, b=0, t=50),
                height=560
            )
            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            st.info("PCA3 column not found. Re-run the notebook with 3-component PCA.")

    with tab_scree:
        if "StudyHours" in df.columns:
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA

            feat_cols = [c for c in ["StudyHours","Attendance","AssignmentCompletion",
                                      "OnlineCourses","Discussions","Motivation",
                                      "StressLevel","LearningStyle","EngagementScore",
                                      "AcademicIndex","ExamScore"] if c in df.columns]
            numeric_only = df[feat_cols].select_dtypes(include=np.number)
            scaled = StandardScaler().fit_transform(numeric_only)
            pca_full = PCA(random_state=42)
            pca_full.fit(scaled)
            cum_var = np.cumsum(pca_full.explained_variance_ratio_)
            n_comp  = len(cum_var)

            fig_scree = go.Figure()
            fig_scree.add_trace(go.Bar(
                x=list(range(1, n_comp+1)),
                y=pca_full.explained_variance_ratio_ * 100,
                marker_color=ACCENT, opacity=0.7, name="Individual"
            ))
            fig_scree.add_trace(go.Scatter(
                x=list(range(1, n_comp+1)), y=cum_var * 100,
                mode="lines+markers",
                line=dict(color="#FF6B35", width=2.5),
                marker=dict(size=7), name="Cumulative"
            ))
            fig_scree.add_hline(y=90, line_color="#FFD166", line_dash="dot",
                                annotation_text="90% Threshold",
                                annotation_font_color="#FFD166")
            plotly_dark_layout(fig_scree, "PCA Scree Plot — Variance Explained", height=420)
            st.plotly_chart(fig_scree, use_container_width=True)
        else:
            st.info("Scree plot requires original feature columns.")


# ═══════════════════════════════════════════════════════════════
#   PAGE: DATA EXPLORER
# ═══════════════════════════════════════════════════════════════
elif "Explorer" in page:

    st.markdown('<p class="section-head">🔍 Interactive Data Explorer</p>', unsafe_allow_html=True)

    col_sel, col_filt = st.columns([2, 1])
    with col_filt:
        selected_label = st.selectbox("Filter by Cluster", ["All"] + all_labels)

    explore_df = dff if selected_label == "All" else dff[dff["ClusterLabel"] == selected_label]

    with col_sel:
        st.markdown(f"""
        <div style='background:{CARD2}; border:1px solid {BORDER}; border-radius:10px;
                    padding:14px 18px; margin-bottom:4px;'>
            <span style='font-family:"Share Tech Mono",monospace; color:{ACCENT};
                         font-size:0.8rem;'>SHOWING</span>
            <span style='font-family:"Share Tech Mono",monospace; color:{TEXT};
                         font-size:1.1rem; font-weight:700; margin-left:10px;'>
                {len(explore_df):,} students
            </span>
            <span style='font-family:"Share Tech Mono",monospace; color:#64748B;
                         font-size:0.8rem; margin-left:6px;'>
                / {len(df):,} total
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.dataframe(explore_df.reset_index(drop=True), use_container_width=True, height=340)

    st.markdown(f"<hr style='border-color:{BORDER}; margin:20px 0 16px;'>", unsafe_allow_html=True)
    st.markdown('<p class="section-head">📈 Custom Scatter Explorer</p>', unsafe_allow_html=True)

    num_available = [c for c in explore_df.select_dtypes(include=np.number).columns
                     if c not in ["Cluster","PCA1","PCA2","PCA3"]]

    sc1, sc2, sc3 = st.columns(3)
    x_axis  = sc1.selectbox("X Axis",  num_available, index=0)
    y_axis  = sc2.selectbox("Y Axis",  num_available, index=min(2, len(num_available)-1))
    size_by = sc3.selectbox("Size By", ["(none)"] + num_available, index=0)

    fig_scatter = px.scatter(
        explore_df,
        x=x_axis, y=y_axis,
        color="ClusterLabel",
        color_discrete_sequence=PALETTE,
        size=size_by if size_by != "(none)" else None,
        size_max=20,
        hover_data={"ClusterLabel": True},
        opacity=0.78,
        title=f"{x_axis} vs {y_axis}"
    )
    plotly_dark_layout(fig_scatter, f"{x_axis}  ×  {y_axis}", height=460)
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown(f"<hr style='border-color:{BORDER}; margin:20px 0 16px;'>", unsafe_allow_html=True)
    st.markdown('<p class="section-head">📋 Descriptive Statistics</p>', unsafe_allow_html=True)
    st.dataframe(explore_df[num_available].describe().round(3).style
                 .background_gradient(cmap="Blues"),
                 use_container_width=True, height=290)


# ═══════════════════════════════════════════════════════════════
#   FOOTER
# ═══════════════════════════════════════════════════════════════
st.markdown(f"""
<hr style='border-color:{BORDER}; margin: 40px 0 20px;'>
<div style='text-align:center; padding: 12px 0 20px;'>
    <div style='font-family:"Share Tech Mono",monospace; font-size:0.7rem;
                color:#475569; letter-spacing:0.12em;'>
        Python &nbsp;·&nbsp; Pandas &nbsp;·&nbsp; Scikit-Learn &nbsp;·&nbsp;
        Plotly &nbsp;·&nbsp; Seaborn &nbsp;·&nbsp; Streamlit
        &nbsp;&nbsp;|&nbsp;&nbsp;
        K-Means Clustering &nbsp;·&nbsp; PCA &nbsp;·&nbsp; Silhouette Analysis
    </div>
</div>
""", unsafe_allow_html=True)
