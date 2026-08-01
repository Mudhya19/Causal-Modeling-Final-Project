import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_agraph import agraph, Node, Edge, Config
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import statsmodels.api as sm
import os

# ═══════════════════════════════════════════════════════════════════
# KONFIGURASI HALAMAN & TEMA
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Visual Analytics — Causal Modeling",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Palet Warna Kustom (Gradasi Biru)
COLORS = {
    "bg_dark": "#0E1117",
    "card_bg": "#1A1F2E",
    "blue_light": "#B0E0E6",
    "blue_mid": "#4682B4",
    "blue_dark": "#00008B",
    "blue_accent": "#87CEEB",
    "red_accent": "#FF6B6B",
    "green_accent": "#51CF66",
    "purple_accent": "#9775FA",
    "text_primary": "#E8EAED",
    "text_secondary": "#9AA0A6",
}

BLUE_SCALE = ["#B0E0E6", "#87CEEB", "#4682B4", "#1E6091", "#00008B"]

# ═══════════════════════════════════════════════════════════════════
# CSS KUSTOM UNTUK TAMPILAN PREMIUM
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1A1F2E 0%, #252B3B 100%);
        border: 1px solid rgba(70, 130, 180, 0.3);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 139, 0.15);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0, 0, 139, 0.25);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(26, 31, 46, 0.8);
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4682B4, #00008B) !important;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1117 0%, #1A1F2E 100%);
    }

    h1, h2, h3 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
    }

    .info-box {
        background: linear-gradient(135deg, rgba(70,130,180,0.15), rgba(0,0,139,0.1));
        border-left: 4px solid #4682B4;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 12px 0;
    }

    .highlight-box {
        background: linear-gradient(135deg, rgba(255,107,107,0.12), rgba(255,107,107,0.05));
        border-left: 4px solid #FF6B6B;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 12px 0;
    }

    .success-box {
        background: linear-gradient(135deg, rgba(81,207,102,0.12), rgba(81,207,102,0.05));
        border-left: 4px solid #51CF66;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 12px 0;
    }

    .styled-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #4682B4, transparent);
        border: none;
        margin: 24px 0;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# FUNGSI UTILITAS & PEMODELAN KAUSAL
# ═══════════════════════════════════════════════════════════════════

@st.cache_data
def load_dataset():
    """Muat dan siapkan dataset dengan binarisasi treatment."""
    paths = [
        os.path.join(os.path.dirname(__file__), "..", "dataset", "heart_attack_prediction_indonesia.csv"),
        os.path.join("dataset", "heart_attack_prediction_indonesia.csv"),
    ]
    df = None
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            break
    if df is None:
        st.error("Dataset tidak ditemukan! Pastikan file CSV tersedia.")
        st.stop()

    df["Heart_Attack_Bin"] = df["heart_attack"].astype(int)
    df["smoking_bin"] = df["smoking_status"].apply(lambda x: 1 if str(x) == "Current" else 0)
    df["diet_bin"] = df["dietary_habits"].apply(lambda x: 1 if str(x) == "Unhealthy" else 0)
    df["pollution_bin"] = df["air_pollution_exposure"].apply(lambda x: 1 if str(x) == "High" else 0)
    return df


@st.cache_data
def load_csv_results():
    """Muat hasil CSV yang sudah di-generate oleh notebook."""
    ate_paths = [
        os.path.join(os.path.dirname(__file__), "..", "logs", "output", "ate_att_indonesia_results.csv"),
        os.path.join("logs", "output", "ate_att_indonesia_results.csv"),
    ]
    med_paths = [
        os.path.join(os.path.dirname(__file__), "..", "logs", "output", "mediasi_indonesia_results.csv"),
        os.path.join("logs", "output", "mediasi_indonesia_results.csv"),
    ]
    ate_df, med_df = None, None
    for p in ate_paths:
        if os.path.exists(p):
            ate_df = pd.read_csv(p)
            break
    for p in med_paths:
        if os.path.exists(p):
            med_df = pd.read_csv(p)
            break
    return ate_df, med_df


def binarize_treatment(data, col_name, positive_labels):
    df_temp = data.copy()
    df_temp["T_bin"] = df_temp[col_name].astype(str).apply(
        lambda x: 1 if x in positive_labels else 0
    )
    return df_temp


OUTCOME_VAR = "heart_attack"
TREATMENT_VARS = [
    "smoking_status", "dietary_habits", "air_pollution_exposure",
    "physical_activity", "alcohol_consumption", "sleep_hours"
]
CONFOUNDER_VARS = [
    "age", "gender", "region", "income_level",
    "family_history", "previous_heart_disease"
]
MEDIATOR_VARS = [
    "hypertension", "obesity", "diabetes", "cholesterol_level",
    "cholesterol_ldl", "cholesterol_hdl", "triglycerides",
    "fasting_blood_sugar", "stress_level", "waist_circumference"
]


def estimate_ate_ipw(data_df, confounders_list):
    df_clean = data_df.dropna(subset=confounders_list + ["T_bin", "Heart_Attack_Bin"]).copy()
    X = pd.get_dummies(df_clean[confounders_list], drop_first=True, dtype=float)
    y_treat = df_clean["T_bin"]
    y_out = df_clean["Heart_Attack_Bin"]

    ps_model = LogisticRegression(max_iter=1000, solver="lbfgs")
    ps_model.fit(X, y_treat)
    ps = np.clip(ps_model.predict_proba(X)[:, 1], 0.02, 0.98)
    df_clean["ipw_weight"] = np.where(df_clean["T_bin"] == 1, 1.0 / ps, 1.0 / (1.0 - ps))

    mean_y1 = np.average(df_clean.loc[df_clean["T_bin"] == 1, "Heart_Attack_Bin"],
                         weights=df_clean.loc[df_clean["T_bin"] == 1, "ipw_weight"])
    mean_y0 = np.average(df_clean.loc[df_clean["T_bin"] == 0, "Heart_Attack_Bin"],
                         weights=df_clean.loc[df_clean["T_bin"] == 0, "ipw_weight"])
    naive_diff = y_out[y_treat == 1].mean() - y_out[y_treat == 0].mean()

    return {
        "Naive_Diff": round(naive_diff, 4),
        "ATE_IPW": round(mean_y1 - mean_y0, 4),
        "n_samples": len(df_clean)
    }


def estimate_att_matching(data_df, confounders_list):
    df_clean = data_df.dropna(subset=confounders_list + ["T_bin", "Heart_Attack_Bin"]).copy()
    X = pd.get_dummies(df_clean[confounders_list], drop_first=True, dtype=float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    treated_idx = np.where(df_clean["T_bin"] == 1)[0]
    control_idx = np.where(df_clean["T_bin"] == 0)[0]

    if len(treated_idx) == 0 or len(control_idx) == 0:
        return {"ATT_Matching": np.nan, "n_matched": 0}

    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(X_scaled[control_idx])
    _, indices = nn.kneighbors(X_scaled[treated_idx])

    matched_control = df_clean["Heart_Attack_Bin"].iloc[control_idx[indices.flatten()]].values
    treated_outcomes = df_clean["Heart_Attack_Bin"].iloc[treated_idx].values
    att = np.mean(treated_outcomes - matched_control)

    return {"ATT_Matching": round(att, 4), "n_matched": len(treated_idx)}


# ═══════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════
df = load_dataset()
ate_df, med_df = load_csv_results()

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🫀 Causal Modeling")
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    **Proyek Tugas Akhir**
    Visual Analytics for Causal Modeling Analysis
    """)

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
    st.markdown("**Dataset:**")
    st.caption(f"📁 {df.shape[0]:,} baris × {df.shape[1]} kolom")
    st.caption("📦 Kaggle — Ankush Panday")
    st.markdown(
        "[🔗 GitHub Repository](https://github.com/Mudhya19/Causal-Modeling-Final-Project)"
    )


# ═══════════════════════════════════════════════════════════════════
# TAB UTAMA
# ═══════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "🏠 Ringkasan",
    "📊 Dimension View",
    "📋 Table View",
    "🕸️ Causal Graph",
    "🔬 Kausalitas Explorer",
    "📚 Metodologi",
])

# ─────────────────────────────────────────────────────────────────
# TAB 1 — RINGKASAN
# ─────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("# 🫀 Visual Analytics for Causal Modeling Analysis")
    st.markdown("### Analisis Kausal Faktor Risiko Gaya Hidup & Lingkungan terhadap Mortalitas Serangan Jantung di Indonesia")
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Observasi", f"{df.shape[0]:,}")
    with col2:
        st.metric("Jumlah Variabel", f"{df.shape[1]}")
    with col3:
        pct_positive = df["Heart_Attack_Bin"].mean() * 100
        st.metric("Serangan Jantung (+)", f"{pct_positive:.1f}%")
    with col4:
        pct_normal = (1 - df["Heart_Attack_Bin"].mean()) * 100
        st.metric("Normal (-)", f"{pct_normal:.1f}%")

    st.markdown("")

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("ATE Merokok (IPW)", "+19.35%", delta="+19.35%")
    with col6:
        st.metric("ATT Merokok (Matching)", "+16.09%", delta="+16.09%")
    with col7:
        st.metric("Direct Effect Rokok", "+19.36%", delta="+19.36%")
    with col8:
        st.metric("Indirect via Hipertensi", "~0%", delta="0%", delta_color="off")

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <h4>📌 Tentang Proyek Ini</h4>
    <p>Proyek ini merancang <strong>User Causality Explorer Interface</strong> yang mengidentifikasi dampak
    kausalitas sejati (<em>True Causal Effect</em>) dari gaya hidup terhadap risiko serangan jantung di Indonesia.
    Tiga komponen wajib telah dipenuhi:</p>
    <ul>
        <li><strong>Dimension View:</strong> Eksplorasi visual interaktif lintas dimensi demografi.</li>
        <li><strong>Table View:</strong> Matriks perbandingan <em>Naive Correlation</em> vs <em>Causal Effect</em>.</li>
        <li><strong>Causal Graph View:</strong> Pemetaan arsitektur DAG menggunakan teori Graf Kausal.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="highlight-box">
    <h4>🔥 Temuan Kritis: Simpson's Paradox pada Diet Buruk</h4>
    <p>Korelasi mentah menunjukkan diet buruk seolah <em>"melindungi"</em> jantung (<strong>-0.53%</strong>).
    Namun setelah pencocokan kausal pada <strong>95.030 pasangan identik</strong>, efek aslinya berbalik menjadi
    <strong>POSITIF (+0.57%)</strong> — membuktikan bahwa diet buruk berbahaya bagi jantung!</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="success-box">
    <h4>💡 Penemuan Mediasi: Rokok Menyerang Jantung Secara Langsung</h4>
    <p>Uji mediasi Baron & Kenny membuktikan bahwa bahaya rokok didominasi oleh <strong>Jalur Langsung
    (Direct Effect +19.36%)</strong>, bukan melalui hipertensi kronis (<em>Indirect Effect ≈ 0%</em>).</p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# TAB 2 — DIMENSION VIEW
# ─────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("# 📊 Dimension View — Eksplorasi Visual Interaktif")
    st.markdown("Komponen **Dimension View** mengeksplorasi distribusi risiko serangan jantung lintas dimensi demografi, gaya hidup, mediator klinis, dan korelasi antar variabel.")
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    st.markdown("### 4.1 Dimensi Demografi Latar Belakang (Confounders)")

    dem_col1, dem_col2, dem_col3 = st.columns(3)

    with dem_col1:
        gender_ha = df.groupby("gender")["Heart_Attack_Bin"].mean().reset_index()
        gender_ha.columns = ["Gender", "Proporsi"]
        fig_gender = px.bar(gender_ha, x="Gender", y="Proporsi",
                            color="Proporsi", color_continuous_scale=BLUE_SCALE,
                            title="Proporsi Serangan Jantung per Gender")
        fig_gender.update_layout(template="plotly_dark", showlegend=False,
                                 yaxis_title="Proporsi Positif", xaxis_title="",
                                 coloraxis_showscale=False, height=400)
        st.plotly_chart(fig_gender)

    with dem_col2:
        region_ha = df.groupby("region")["Heart_Attack_Bin"].mean().reset_index()
        region_ha.columns = ["Wilayah", "Proporsi"]
        fig_region = px.bar(region_ha, x="Wilayah", y="Proporsi",
                            color="Proporsi", color_continuous_scale=BLUE_SCALE,
                            title="Proporsi Serangan Jantung per Wilayah")
        fig_region.update_layout(template="plotly_dark", showlegend=False,
                                 yaxis_title="Proporsi Positif", xaxis_title="",
                                 coloraxis_showscale=False, height=400)
        st.plotly_chart(fig_region)

    with dem_col3:
        income_ha = df.groupby("income_level")["Heart_Attack_Bin"].mean().reset_index()
        income_ha.columns = ["Pendapatan", "Proporsi"]
        order = ["Low", "Middle", "High"]
        income_ha["Pendapatan"] = pd.Categorical(income_ha["Pendapatan"], categories=order, ordered=True)
        income_ha = income_ha.sort_values("Pendapatan")
        fig_income = px.bar(income_ha, x="Pendapatan", y="Proporsi",
                            color="Proporsi", color_continuous_scale=BLUE_SCALE,
                            title="Proporsi per Tingkat Pendapatan")
        fig_income.update_layout(template="plotly_dark", showlegend=False,
                                 yaxis_title="Proporsi Positif", xaxis_title="",
                                 coloraxis_showscale=False, height=400)
        st.plotly_chart(fig_income)

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    st.markdown("### 4.2 Dimensi Intervensi Gaya Hidup & Lingkungan (Treatment)")

    treatment_display = ["smoking_status", "dietary_habits", "air_pollution_exposure", "physical_activity"]
    t_col1, t_col2 = st.columns(2)

    for i, tvar in enumerate(treatment_display):
        col_target = t_col1 if i % 2 == 0 else t_col2
        with col_target:
            if df[tvar].nunique() > 10:
                temp = df.copy()
                temp["bin"] = pd.qcut(temp[tvar], q=4, duplicates="drop").astype(str)
                t_ha = temp.groupby("bin")["Heart_Attack_Bin"].mean().reset_index()
                t_ha.columns = ["Kategori", "Proporsi"]
            else:
                t_ha = df.groupby(tvar)["Heart_Attack_Bin"].mean().sort_values(ascending=False).reset_index()
                t_ha.columns = ["Kategori", "Proporsi"]

            fig_t = px.bar(t_ha, x="Kategori", y="Proporsi",
                           color="Proporsi", color_continuous_scale=BLUE_SCALE,
                           title=f"Heart Attack per {tvar}")
            fig_t.update_layout(template="plotly_dark", showlegend=False,
                                yaxis_title="Proporsi Positif", xaxis_title="",
                                coloraxis_showscale=False, height=350)
            st.plotly_chart(fig_t)

    st.markdown("""
    <div class="info-box">
    <strong>📌 Analisis Kontras Sinyal Kausalitas:</strong> Perokok aktif (<em>Current</em>) memiliki probabilitas
    serangan jantung mencapai <strong>~54.5%</strong>, melonjak drastis dibandingkan non-perokok (<em>Never</em>)
    yang hanya <strong>~35.2%</strong> (selisih +19.3%).
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    st.markdown("### 4.3 Distribusi Metrik Klinis (Mediator Boxplot)")

    mediator_numeric = ["waist_circumference", "cholesterol_ldl", "fasting_blood_sugar", "triglycerides"]
    m_col1, m_col2 = st.columns(2)

    for i, mvar in enumerate(mediator_numeric):
        col_target = m_col1 if i % 2 == 0 else m_col2
        with col_target:
            fig_box = px.box(df, x="heart_attack", y=mvar,
                             color="heart_attack",
                             color_discrete_sequence=[COLORS["blue_accent"], COLORS["blue_dark"]],
                             title=f"Distribusi {mvar} per Status Serangan Jantung",
                             labels={"heart_attack": "Heart Attack (0=Normal, 1=Positif)"})
            fig_box.update_layout(template="plotly_dark", showlegend=False, height=350)
            st.plotly_chart(fig_box)

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    st.markdown("### 4.4 Matriks Korelasi Klinis (Heatmap)")

    leakage_vars = ["blood_pressure_systolic", "blood_pressure_diastolic",
                    "EKG_results", "medication_usage", "participated_in_free_screening"]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_for_corr = [c for c in numeric_cols if c not in leakage_vars and c != "Heart_Attack_Bin"
                     and c not in ["smoking_bin", "diet_bin", "pollution_bin"]]

    corr_matrix = df[cols_for_corr].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    corr_masked = corr_matrix.copy()
    corr_masked = corr_masked.where(~mask)

    fig_corr = px.imshow(
        corr_masked.values,
        x=corr_masked.columns.tolist(),
        y=corr_masked.index.tolist(),
        color_continuous_scale="Blues",
        zmin=-0.5, zmax=1,
        title="Matriks Korelasi Variabel Klinis & Latar Belakang",
        text_auto=".2f",
    )
    fig_corr.update_layout(template="plotly_dark", height=700, width=900)
    st.plotly_chart(fig_corr)

    st.markdown("""
    <div class="info-box">
    <strong>📌 Keterbatasan Korelasi Linier:</strong> Heatmap ini menunjukkan korelasi antar variabel cenderung rendah
    dan tidak mampu menangkap hubungan sebab-akibat tersembunyi di balik <em>confounding bias</em>.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# TAB 3 — TABLE VIEW
# ─────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("# 📋 Table View — Evaluasi Output Pemodelan")
    st.markdown("Komponen **Table View** menyajikan rekapitulasi numerik komparasi korelasi mentah vs dampak kausal sejati.")
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Tabel 4.1 — Komparasi Naive Diff vs ATE (IPW) vs ATT (Matching)")

    if ate_df is not None:
        display_ate = ate_df.copy()
        display_ate.columns = [
            "Variabel Intervensi", "Perlakuan (T=1)",
            "Naive Diff", "ATE (IPW)", "ATT (Matching)", "Pasangan Matched"
        ]
        for col in ["Naive Diff", "ATE (IPW)", "ATT (Matching)"]:
            display_ate[col] = display_ate[col].apply(lambda x: f"{x:+.2%}" if pd.notna(x) else "N/A")
        display_ate["Pasangan Matched"] = display_ate["Pasangan Matched"].apply(
            lambda x: f"{int(x):,}" if pd.notna(x) else "N/A"
        )
        st.dataframe(display_ate, hide_index=True)
    else:
        st.warning("File `ate_att_indonesia_results.csv` tidak ditemukan.")

    st.markdown("""
    <div class="highlight-box">
    <h4>🔥 Temuan Kritis: Fenomena Simpson's Paradox pada Diet Buruk</h4>
    <p>Pada kolom <strong>Naive Diff</strong> dan <strong>ATE (IPW)</strong>, diet buruk justru menunjukkan angka negatif
    (<strong>-0.53%</strong> hingga <strong>-0.54%</strong>). Ketika algoritma <strong>1-to-1 Nearest Neighbor PSM</strong>
    diterapkan pada <strong>95.030 pasangan kembar</strong>, arah efek kausal terbukti
    <strong>berbalik menjadi POSITIF (+0.57%)</strong>.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Visualisasi Komparasi: Korelasi Mentah vs Efek Kausal")

    if ate_df is not None:
        compare_data = []
        for _, row in ate_df.iterrows():
            var = row.iloc[0]
            compare_data.append({"Variabel": var, "Metode": "Naive Diff", "Efek": row.iloc[2]})
            compare_data.append({"Variabel": var, "Metode": "ATE (IPW)", "Efek": row.iloc[3]})
            compare_data.append({"Variabel": var, "Metode": "ATT (Matching)", "Efek": row.iloc[4]})

        compare_df = pd.DataFrame(compare_data)
        fig_compare = px.bar(
            compare_df, x="Variabel", y="Efek", color="Metode",
            barmode="group",
            color_discrete_sequence=[COLORS["blue_light"], COLORS["blue_mid"], COLORS["blue_dark"]],
            title="Komparasi Korelasi Mentah vs Efek Kausal (ATE & ATT)",
        )
        fig_compare.update_layout(template="plotly_dark", yaxis_title="Efek (Proporsi)",
                                  xaxis_title="Variabel Intervensi", height=450)
        fig_compare.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
        st.plotly_chart(fig_compare)

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Tabel 5.1 — Hasil Analisis Mediasi Baron & Kenny")
    st.markdown("**Jalur:** `Merokok Aktif (T)` → `Hipertensi (M)` → `Serangan Jantung (Y)`")

    if med_df is not None:
        st.dataframe(med_df, hide_index=True)
    else:
        st.warning("File `mediasi_indonesia_results.csv` tidak ditemukan.")

    st.markdown("""
    <div class="success-box">
    <h4>💡 Interpretasi Mediasi Klinis</h4>
    <p><strong>Direct Effect (c') = +0.1936:</strong> Bahaya langsung rokok terhadap jantung menyumbang <strong>99,97%</strong>
    dari total efek.</p>
    <p><strong>Indirect Effect (a × b) ≈ 0:</strong> Jalur tidak langsung via hipertensi mendekati nol.</p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# TAB 4 — CAUSAL GRAPH (DAG)
# ─────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("# 🕸️ Causal Graph View — Directed Acyclic Graph (DAG)")
    st.markdown("Komponen **Causal Graph View** memetakan arsitektur kausalitas medis serangan jantung Indonesia.")
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    dag_nodes_def = {
        "age": ("C", "Usia", "#9775FA"),
        "gender": ("C", "Gender", "#9775FA"),
        "region": ("C", "Wilayah", "#9775FA"),
        "income_level": ("C", "Pendapatan", "#9775FA"),
        "family_history": ("C", "Riwayat Keluarga", "#9775FA"),
        "previous_heart_disease": ("C", "Riwayat Jantung", "#9775FA"),
        "smoking_status": ("T", "Merokok", "#51CF66"),
        "dietary_habits": ("T", "Diet", "#51CF66"),
        "air_pollution_exposure": ("T", "Polusi Udara", "#51CF66"),
        "hypertension": ("M", "Hipertensi", "#4682B4"),
        "obesity": ("M", "Obesitas", "#4682B4"),
        "diabetes": ("M", "Diabetes", "#4682B4"),
        "cholesterol_ldl": ("M", "Kolesterol LDL", "#4682B4"),
        "heart_attack": ("Y", "Serangan Jantung", "#FF6B6B"),
    }

    dag_edges_def = [
        ("region", "dietary_habits"), ("income_level", "dietary_habits"),
        ("age", "smoking_status"), ("region", "air_pollution_exposure"),
        ("age", "heart_attack"), ("family_history", "heart_attack"),
        ("previous_heart_disease", "heart_attack"), ("gender", "heart_attack"),
        ("age", "hypertension"), ("age", "diabetes"),
        ("smoking_status", "hypertension"), ("smoking_status", "cholesterol_ldl"),
        ("dietary_habits", "obesity"), ("dietary_habits", "diabetes"),
        ("dietary_habits", "hypertension"),
        ("hypertension", "heart_attack"), ("obesity", "heart_attack"),
        ("diabetes", "heart_attack"), ("cholesterol_ldl", "heart_attack"),
        ("smoking_status", "heart_attack"), ("air_pollution_exposure", "heart_attack"),
    ]

    leg_col1, leg_col2, leg_col3, leg_col4 = st.columns(4)
    with leg_col1:
        st.markdown("🟣 **Confounder** (Demografi)")
    with leg_col2:
        st.markdown("🟢 **Treatment** (Gaya Hidup)")
    with leg_col3:
        st.markdown("🔵 **Mediator** (Komorbiditas)")
    with leg_col4:
        st.markdown("🔴 **Outcome** (Serangan Jantung)")

    st.markdown("")

    nodes = []
    for key, (grp, label, color) in dag_nodes_def.items():
        size_val = 35 if grp == "Y" else 28
        nodes.append(Node(
            id=key, label=label, size=size_val, color=color,
            font={"color": "#FFFFFF", "size": 14, "face": "Inter"},
            borderWidth=2, borderWidthSelected=4, shape="dot",
        ))

    edges = []
    for src, tgt in dag_edges_def:
        edge_color = "#666666"
        edge_width = 1.5
        if tgt == "heart_attack" and dag_nodes_def.get(src, ("",))[0] == "T":
            edge_color = "#FF6B6B"
            edge_width = 3.0
        edges.append(Edge(
            source=src, target=tgt, color=edge_color, width=edge_width,
            arrows={"to": {"enabled": True, "scaleFactor": 1.2}},
            smooth={"type": "curvedCW", "roundness": 0.15},
        ))

    config = Config(
        width=900, height=600, directed=True, physics=True,
        hierarchical=False, nodeHighlightBehavior=True,
        highlightColor="#F7A072", collapsible=False,
    )

    agraph(nodes=nodes, edges=edges, config=config)

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Taksonomi Hierarki Node DAG")

    dag_table = pd.DataFrame([
        {"Lapis": "Atas (Root Confounders)", "Warna": "🟣 Ungu",
         "Variabel": "Age, Gender, Region, Income, Family History, Previous Heart Disease",
         "Peran": "Sumber backdoor path — harus ditutup dalam estimasi kausal"},
        {"Lapis": "Tengah (Biological Mediators)", "Warna": "🔵 Biru",
         "Variabel": "Hypertension, Obesity, Diabetes, Cholesterol LDL",
         "Peran": "Rute transmisi sekunder — menjembatani treatment ke outcome"},
        {"Lapis": "Bawah (Treatment → Outcome)", "Warna": "🟢 Hijau → 🔴 Merah",
         "Variabel": "Smoking, Diet, Pollution → Heart Attack",
         "Peran": "Hubungan langsung dari perilaku/paparan menuju outcome akhir"},
    ])
    st.dataframe(dag_table, hide_index=True)

    st.markdown("""
    <div class="info-box">
    <strong>📌 Cara Membaca DAG:</strong> Setiap panah menunjukkan arah sebab-akibat berdasarkan pengetahuan domain medis.
    Panah <span style="color:#FF6B6B;font-weight:bold">merah tebal</span> menandakan jalur langsung Treatment → Outcome.
    Variabel perancu (ungu) harus dikontrol untuk menutup <em>backdoor path</em>.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# TAB 5 — KAUSALITAS EXPLORER (INTERAKTIF)
# ─────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("# 🔬 Kausalitas Explorer — Estimasi Real-Time")
    st.markdown("Pilih variabel intervensi untuk menghitung **ATE (IPW)** dan **ATT (Matching)** secara langsung.")
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    explorer_options = {
        "Merokok Aktif (smoking_status = Current)": ("smoking_status", ["Current"]),
        "Diet Buruk (dietary_habits = Unhealthy)": ("dietary_habits", ["Unhealthy"]),
        "Polusi Tinggi (air_pollution_exposure = High)": ("air_pollution_exposure", ["High"]),
        "Aktivitas Fisik Tinggi (physical_activity = High)": ("physical_activity", ["High"]),
    }

    selected = st.selectbox("🎯 Pilih Variabel Intervensi (Treatment)", list(explorer_options.keys()))

    if st.button("⚡ Hitung Estimasi Kausal", type="primary"):
        col_name, pos_labels = explorer_options[selected]
        df_sub = binarize_treatment(df, col_name, pos_labels)

        with st.spinner("Menghitung ATE (IPW) — menimbang ulang seluruh populasi..."):
            ate_r = estimate_ate_ipw(df_sub, CONFOUNDER_VARS)

        with st.spinner("Menghitung ATT (Matching) — mencocokkan pasangan kembar statistik..."):
            att_r = estimate_att_matching(df_sub, CONFOUNDER_VARS)

        st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.metric("Naive Diff (Korelasi Mentah)", f"{ate_r['Naive_Diff']:+.2%}")
        with res_col2:
            st.metric("ATE (IPW) — Efek Populasi", f"{ate_r['ATE_IPW']:+.2%}",
                       delta=f"{ate_r['ATE_IPW']:+.2%}")
        with res_col3:
            att_val = att_r["ATT_Matching"]
            st.metric("ATT (Matching) — Efek Subgrup",
                       f"{att_val:+.2%}" if not np.isnan(att_val) else "N/A",
                       delta=f"{att_val:+.2%}" if not np.isnan(att_val) else None)

        st.markdown(f"""
        <div class="info-box">
        <strong>📊 Detail Komputasi:</strong><br>
        • Total sampel bersih: <strong>{ate_r['n_samples']:,}</strong> individu<br>
        • Pasangan matched (1-to-1 KNN): <strong>{att_r['n_matched']:,}</strong> pasang
        </div>
        """, unsafe_allow_html=True)

        fig_exp = go.Figure()
        labels = ["Naive Diff", "ATE (IPW)", "ATT (Matching)"]
        values = [ate_r["Naive_Diff"], ate_r["ATE_IPW"], att_val if not np.isnan(att_val) else 0]
        colors_bar = [COLORS["blue_light"], COLORS["blue_mid"], COLORS["blue_dark"]]

        fig_exp.add_trace(go.Bar(
            x=labels, y=values,
            marker_color=colors_bar,
            text=[f"{v:+.4f}" for v in values],
            textposition="outside",
            textfont=dict(size=14, color="white"),
        ))
        fig_exp.update_layout(template="plotly_dark",
                              title=f"Komparasi Estimasi Efek Kausal: {col_name}",
                              yaxis_title="Efek (Proporsi Risiko)", height=400)
        fig_exp.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
        st.plotly_chart(fig_exp)


# ─────────────────────────────────────────────────────────────────
# TAB 6 — METODOLOGI
# ─────────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown("# 📚 Metodologi & Rujukan Ilmiah")
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    st.markdown("### 1. Inverse Probability Weighting (IPW)")
    st.markdown(r"""
    IPW mengestimasi **Average Treatment Effect (ATE)** dengan membobot setiap individu berdasarkan
    kebalikan dari *propensity score* mereka:

    $$ATE_{IPW} = \frac{1}{n}\sum_{i=1}^{n}\left[\frac{T_i \cdot Y_i}{e(X_i)} - \frac{(1-T_i) \cdot Y_i}{1-e(X_i)}\right]$$

    di mana $e(X_i) = P(T=1|X_i)$ adalah propensity score yang diestimasi melalui Regresi Logistik.
    """)

    st.markdown("### 2. Propensity Score Matching (PSM)")
    st.markdown(r"""
    PSM mengestimasi **Average Treatment Effect on the Treated (ATT)** dengan mencocokkan setiap
    individu *treated* dengan "kembar statistik" dari kelompok kontrol:

    $$ATT_{PSM} = \frac{1}{n_T}\sum_{i \in T}\left[Y_i(1) - Y_{j(i)}(0)\right]$$

    di mana $j(i)$ adalah indeks individu kontrol terdekat berdasarkan jarak Euclidean terstandardisasi.
    """)

    st.markdown("### 3. Analisis Mediasi Baron & Kenny (1986)")
    st.markdown(r"""
    Kerangka 4 langkah untuk menguji jalur mediasi:
    1. **Jalur $a$:** $M = \alpha_0 + a \cdot T + \gamma \cdot C + \varepsilon_1$
    2. **Jalur $b$ & $c'$:** $Y = \beta_0 + c' \cdot T + b \cdot M + \delta \cdot C + \varepsilon_2$
    3. **Indirect Effect:** $IE = a \times b$
    4. **Total Effect:** $TE = c' + a \times b$
    """)

    st.markdown("### 4. Directed Acyclic Graph (DAG)")
    st.markdown("""
    DAG adalah representasi graf berarah tanpa siklus yang memetakan **asumsi kausal** berdasarkan
    pengetahuan domain medis. Dalam kerangka Pearl's *Do-Calculus*:
    - **Backdoor criterion** mengidentifikasi *adjustment set* (variabel yang harus dikontrol).
    - **Front-door criterion** digunakan ketika mediator diketahui.
    """)

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    st.markdown("### 📖 Daftar Rujukan")
    st.markdown("""
    1. **Rubin, D. B.** (1974). *Estimating causal effects of treatments in randomized and nonrandomized studies*. Journal of Educational Psychology, 66(5), 688–701.
    2. **Baron, R. M., & Kenny, D. A.** (1986). *The moderator–mediator variable distinction in social psychological research*. Journal of Personality and Social Psychology, 51(6), 1173–1182.
    3. **Rosenbaum, P. R., & Rubin, D. B.** (1983). *The central role of the propensity score in observational studies for causal effects*. Biometrika, 70(1), 41–55.
    4. **Pearl, J.** (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.
    5. **Hernán, M. A., & Robins, J. M.** (2020). *Causal Inference: What If*. Chapman & Hall/CRC.
    6. **Panday, A.** (2026). *Heart Attack Prediction in Indonesia Dataset*. Kaggle Repository.
    """)
