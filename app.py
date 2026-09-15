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

PERCIN_TIPLERI = {
    "MS20470": "MS20470 — Universal/Protruding Head (Düz Baş, Havşa Yok)",
    "NAS1097": "NAS1097 — Flush/Countersunk Head (Havşalı)",
}


@st.cache_resource
def modeli_yukle(percin_tipi):
    return joblib.load(f"montaj_modeli_{percin_tipi}.pkl")


@st.cache_data
def metrikleri_yukle(percin_tipi):
    return joblib.load(f"metrikler_{percin_tipi}.pkl")


st.title("✈️ AeroRivet-QC-RF — Havacılık Montaj Kalite Kontrol Sistemi")
st.caption("Random Forest tabanlı perçin/delik kalite kontrolü — FAA-H-8083-31A tolerans referanslı")

sekme1, sekme2 = st.tabs(["🚀 Anlık Saha Kalite Kontrolü", "📊 Model Başarı & Güvenilirlik Analizi"])

with sekme1:
    percin_tipi_1 = st.selectbox(
        "Perçin Tipi",
        options=list(PERCIN_TIPLERI.keys()),
        format_func=lambda k: PERCIN_TIPLERI[k],
        key="percin_tipi_sekme1",
    )
    model = modeli_yukle(percin_tipi_1)
    metrikler = metrikleri_yukle(percin_tipi_1)

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
        havsa_derinligi = None
        if percin_tipi_1 == "NAS1097":
            havsa_derinligi = st.number_input(
                "Havşa Derinliği (mm)", min_value=0.80, max_value=1.60, value=1.20, step=0.001, format="%.3f"
            )
        else:
            st.info("MS20470 (Universal/Protruding Head) perçinlerde havşa açılmaz — bu parametre bu perçin tipi için geçerli değildir.")
        operator_tecrube = st.slider("Operatör Tecrübe (Yıl)", min_value=1, max_value=15, value=5)

    if st.button("🔍 Kalite Kontrol Yap", type="primary"):
        deger_haritasi = {
            "Delik_Capi_mm": delik_capi,
            "Baski_Kuvveti_PSI": baski_kuvveti,
            "Havsa_Derinligi_mm": havsa_derinligi,
            "Operator_Tecrube_Yil": operator_tecrube,
        }
        girdi = pd.DataFrame(
            [[deger_haritasi[ozellik] for ozellik in metrikler["features"]]],
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
    percin_tipi_2 = st.selectbox(
        "Perçin Tipi",
        options=list(PERCIN_TIPLERI.keys()),
        format_func=lambda k: PERCIN_TIPLERI[k],
        key="percin_tipi_sekme2",
    )
    metrikler2 = metrikleri_yukle(percin_tipi_2)

    st.subheader("Model Performans Metrikleri")

    esik = st.slider(
        "Karar Eşiği (Decision Threshold)",
        min_value=0.05,
        max_value=0.95,
        value=VARSAYILAN_ESIK,
        step=0.01,
        help="Düşük eşik = daha yüksek Recall (daha az kaçan hata), daha düşük Precision",
    )

    y_test = metrikler2["y_test"]
    olasiliklar = metrikler2["olasiliklar"]
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
    ax.set_title(f"Precision-Recall Curve — {percin_tipi_2} Rework Tespiti")
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
