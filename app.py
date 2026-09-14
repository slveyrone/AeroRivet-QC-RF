import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

st.set_page_config(page_title="AeroRivet QC Dashboard", layout="wide")

VARSAYILAN_ESIK = 0.25


@st.cache_resource
def modeli_yukle():
    return joblib.load("montaj_modeli.pkl")


@st.cache_data
def metrikleri_yukle():
    return joblib.load("metrikler.pkl")


model = modeli_yukle()
metrikler = metrikleri_yukle()

st.title("✈️ AeroRivet-QC-RF — Havacılık Montaj Kalite Kontrol Sistemi")
st.caption("Random Forest tabanlı NAS1097 flush perçin/delik/havşa kalite kontrolü — FAA-H-8083-31A tolerans referanslı")

sekme1, sekme2 = st.tabs(["🚀 Anlık Saha Kalite Kontrolü", "📊 Model Başarı & Güvenilirlik Analizi"])

with sekme1:
    st.subheader("Saha Ölçüm Girişi")
    st.write("Operatör tarafından alınan ölçümleri girin ve parçanın kalite durumunu kontrol edin.")

    col1, col2 = st.columns(2)
    with col1:
        delik_capi = st.number_input(
            "Delik Çapı (mm)", min_value=4.50, max_value=5.00, value=4.76, step=0.001, format="%.3f"
        )
        baski_kuvveti = st.number_input(
            "Baskı Kuvveti (PSI)", min_value=1000, max_value=5000, value=3000, step=10
        )
    with col2:
        havsa_derinligi = st.number_input(
            "Havşa Derinliği (mm)", min_value=0.80, max_value=1.60, value=1.20, step=0.001, format="%.3f"
        )
        operator_tecrube = st.slider("Operatör Tecrübe (Yıl)", min_value=1, max_value=15, value=5)

    if st.button("🔍 Kalite Kontrol Yap", type="primary"):
        girdi = pd.DataFrame(
            [[delik_capi, baski_kuvveti, havsa_derinligi, operator_tecrube]],
            columns=metrikler["features"],
        )
        olasilik = model.predict_proba(girdi)[0, 1]
        rework_gerekli = olasilik >= VARSAYILAN_ESIK

        st.metric("Rework Olasılığı", f"%{olasilik * 100:.1f}")

        if rework_gerekli:
            st.error(
                f"⛔ REWORK GEREKLİ — Parça toleransları karşılamıyor olabilir. "
                f"(Hata olasılığı: %{olasilik * 100:.1f}, Karar eşiği: {VARSAYILAN_ESIK})"
            )
        else:
            st.success(
                f"✅ PARÇA UYGUN — Kalite kriterlerini karşılıyor. "
                f"(Hata olasılığı: %{olasilik * 100:.1f}, Karar eşiği: {VARSAYILAN_ESIK})"
            )

with sekme2:
    st.subheader("Model Performans Metrikleri")

    esik = st.slider(
        "Karar Eşiği (Decision Threshold)",
        min_value=0.05,
        max_value=0.95,
        value=VARSAYILAN_ESIK,
        step=0.01,
        help="Düşük eşik = daha yüksek Recall (daha az kaçan hata), daha düşük Precision",
    )

    y_test = metrikler["y_test"]
    olasiliklar = metrikler["olasiliklar"]
    tahminler = (olasiliklar >= esik).astype(int)

    accuracy = accuracy_score(y_test, tahminler)
    precision = precision_score(y_test, tahminler, zero_division=0)
    recall = recall_score(y_test, tahminler, zero_division=0)
    f1 = f1_score(y_test, tahminler, zero_division=0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"%{accuracy * 100:.2f}")
    m2.metric("Recall (Kritik)", f"%{recall * 100:.2f}")
    m3.metric("Precision", f"%{precision * 100:.2f}")
    m4.metric("F1-Score", f"%{f1 * 100:.2f}")

    st.divider()
    st.subheader("Precision-Recall Eğrisi")

    precisions, recalls, thresholds = precision_recall_curve(y_test, olasiliklar)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.set_style("whitegrid")
    ax.plot(recalls, precisions, color="#1f4e8c", linewidth=2, label="Precision-Recall Eğrisi")

    hedef_recall_idx = np.argmin(np.abs(recalls - 0.99))
    ax.axvline(x=0.99, color="red", linestyle="--", linewidth=1.5, label="%99 Recall Hedefi")
    ax.scatter(
        recalls[hedef_recall_idx],
        precisions[hedef_recall_idx],
        color="red",
        s=100,
        zorder=5,
        label="Hedef Nokta (~%99 Recall)",
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Rework Tespiti")
    ax.legend(loc="lower left")
    ax.set_xlim([0, 1.02])
    ax.set_ylim([0, 1.02])

    st.pyplot(fig)

    st.info(
        "✈️ **Havacılıkta Recall Neden Kritiktir?**\n\n"
        "False Negative (Kaçan Hata), gerçekte tolerans dışı olan bir parçanın model tarafından "
        "'sağlam' olarak sınıflandırılması demektir. Bu tip bir hata, kusurlu bir perçin/delik "
        "bağlantısının uçağa monte edilmesine ve yapısal bütünlüğün tehlikeye girmesine yol açabilir. "
        "Bu yüzden kalite kontrol sistemlerinde **Accuracy yerine Recall** öncelikli metrik olarak "
        "kabul edilir; düşük bir karar eşiği ile False Negative sayısı minimize edilerek "
        "'şüpheli her parça incelensin' güvenlik felsefesi uygulanır."
    )
