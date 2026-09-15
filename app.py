import os
from datetime import date, datetime, timedelta

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import precision_recall_curve

st.set_page_config(page_title="AeroRivet QC Dashboard", layout="wide")

IZLENEBILIRLIK_DOSYASI = "izlenebilirlik_kaydi.csv"

PERCIN_TIPLERI = {
    "MS20470": "MS20470 — Universal/Protruding Head (Bombe Baş, Havşa Yok)",
    "NAS1097": "NAS1097 — Flush/Countersunk Head (Havşalı)",
}

PARCA_SINIFLARI = {
    "A": "Sınıf A — Birincil yapısal / kritik",
    "B": "Sınıf B — İkincil yapısal",
    "C": "Sınıf C — Kritik olmayan / braket",
}


@st.cache_resource
def modeli_yukle(percin_tipi):
    return joblib.load(f"montaj_modeli_{percin_tipi}.pkl")


@st.cache_data
def metrikleri_yukle(percin_tipi):
    return joblib.load(f"metrikler_{percin_tipi}.pkl")


def kademe_belirle(olasilik, esik_sari, esik_kirmizi, parca_sinifi):
    if olasilik >= esik_kirmizi:
        return "KIRMIZI", "Yüksek güvenli tolerans dışı tahmin"
    if olasilik >= esik_sari:
        if parca_sinifi == "A":
            return "KIRMIZI", "Sınıf A parçada sarı kademe uyarısı — kritiklik nedeniyle yükseltildi"
        return "SARI", "Emniyet tabanı eşiği aşıldı"
    return "YESIL", "Ölçümler tolerans bandının içinde"


def kayit_ekle(satir):
    df = pd.DataFrame([satir])
    df.to_csv(
        IZLENEBILIRLIK_DOSYASI,
        mode="a",
        header=not os.path.exists(IZLENEBILIRLIK_DOSYASI),
        index=False,
        encoding="utf-8",
    )


st.title("AeroRivet-QC — Havacılık Montaj Kalite Kontrol Sistemi")
st.caption("Kademeli müdahale mimarisi · FAA-H-8083-31A · AS9100D / AS9102 Rev C referanslı")

sekme1, sekme2, sekme3 = st.tabs(
    ["Saha Kalite Kontrolü", "Model Başarımı ve FAI Kaydı", "İzlenebilirlik Kaydı"]
)

with sekme1:
    ust1, ust2 = st.columns(2)
    with ust1:
        percin_tipi = st.selectbox(
            "Perçin Tipi",
            options=list(PERCIN_TIPLERI.keys()),
            format_func=lambda k: PERCIN_TIPLERI[k],
            key="percin_tipi_sekme1",
        )
    with ust2:
        parca_sinifi = st.selectbox(
            "Parça Kritiklik Sınıfı",
            options=list(PARCA_SINIFLARI.keys()),
            format_func=lambda k: PARCA_SINIFLARI[k],
            index=1,
            key="parca_sinifi",
        )

    model = modeli_yukle(percin_tipi)
    metrikler = metrikleri_yukle(percin_tipi)

    if metrikler["fai"]["genel_sonuc"] != "UYGUN":
        st.warning(
            f"Bu model ({percin_tipi}, parmak izi {metrikler['model_parmak_izi']}) Model FAI doğrulamasından "
            f"UYGUNSUZ döndü. Üretimde serbest bırakılmadan önce kalite mühendisi incelemesi gerekir — "
            f"ayrıntı için FAI sekmesine bakın."
        )

    st.subheader("Ölçüm Aleti Doğrulaması")
    kalibrasyon_tarihi = st.date_input(
        "Ölçüm aleti kalibrasyon geçerlilik tarihi",
        value=date.today() + timedelta(days=90),
        help="Kalibrasyon süresi geçmiş aletten gelen ölçüm değerlendirmeye alınmaz (AS9100 §7.1.5)",
    )
    kalibrasyon_gecerli = kalibrasyon_tarihi >= date.today()
    if not kalibrasyon_gecerli:
        st.error(
            "GEÇERSİZ ÖLÇÜM — Alet kalibrasyon süresi dolmuş. Kalite kontrol yapılamaz; "
            "geçerli kalibrasyonlu alet ile yeniden ölçüm alın."
        )

    st.subheader("Saha Ölçüm Girişi")
    col1, col2 = st.columns(2)
    with col1:
        delik_capi = st.number_input(
            "Delik Çapı (mm)", min_value=4.50, max_value=5.10, value=4.76, step=0.001, format="%.3f"
        )
        baski_kuvveti = st.number_input(
            "Baskı Kuvveti (PSI)", min_value=1500, max_value=5000, value=3000, step=10
        )
    with col2:
        havsa_derinligi = None
        if percin_tipi == "NAS1097":
            havsa_derinligi = st.number_input(
                "Havşa Derinliği (mm)", min_value=0.90, max_value=1.60, value=1.20, step=0.001, format="%.3f"
            )
        else:
            st.info(
                "MS20470 perçinlerde havşa açılmaz — bu parametre bu perçin tipi için geçerli değildir."
            )
        operator_tecrube = st.slider("Operatör Tecrübe (Yıl)", min_value=1, max_value=15, value=5)

    if st.button("Kalite Kontrol Yap", type="primary", disabled=not kalibrasyon_gecerli):
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
        olasilik = float(model.predict_proba(girdi)[0, 1])
        kademe, gerekce = kademe_belirle(
            olasilik, metrikler["esik_sari"], metrikler["esik_kirmizi"], parca_sinifi
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Uygunsuzluk Olasılığı", f"%{olasilik * 100:.2f}")
        m2.metric("Sarı Eşik", f"{metrikler['esik_sari']:.4f}")
        m3.metric("Kırmızı Eşik", f"{metrikler['esik_kirmizi']:.4f}")

        if kademe == "KIRMIZI":
            st.error(
                f"KIRMIZI KADEME — İŞ DURDURULDU. {gerekce}. "
                "Parça MRB'ye sevk edilmeli; dispozisyon yetkili imza olmadan kapatılamaz. "
                "Sınıf A parçalarda NDT örnekleme planı da tetiklenir."
            )
        elif kademe == "SARI":
            st.warning(
                f"SARI KADEME — BELGELİ İKİNCİL KONTROL. {gerekce}. "
                "İş durmaz; ölçüm ikinci kez alınıp kayda geçirilmeli veya amir onayı istenmeli."
            )
        else:
            st.success(f"YEŞİL KADEME — DEVAM. {gerekce}. Standart SPC izlemesi dışında ek adım yok.")

        kayit_ekle({
            "Zaman_Damgasi": datetime.now().isoformat(timespec="seconds"),
            "Percin_Tipi": percin_tipi,
            "Parca_Sinifi": parca_sinifi,
            "Delik_Capi_mm": delik_capi,
            "Baski_Kuvveti_PSI": baski_kuvveti,
            "Havsa_Derinligi_mm": havsa_derinligi if havsa_derinligi is not None else "",
            "Operator_Tecrube_Yil": operator_tecrube,
            "Alet_Kalibrasyon_Gecerlilik": kalibrasyon_tarihi.isoformat(),
            "Model_Parmak_Izi": metrikler["model_parmak_izi"],
            "Rework_Olasiligi": round(olasilik, 6),
            "Kademe": kademe,
            "Dispozisyon": "MRB_SEVK" if kademe == "KIRMIZI" else ("IKINCIL_KONTROL" if kademe == "SARI" else "KABUL"),
        })
        st.caption("Bu kontrol, dijital doğum sertifikası kaydına işlendi (AS9100 §8.5.2).")

with sekme2:
    percin_tipi_2 = st.selectbox(
        "Perçin Tipi",
        options=list(PERCIN_TIPLERI.keys()),
        format_func=lambda k: PERCIN_TIPLERI[k],
        key="percin_tipi_sekme2",
    )
    m = metrikleri_yukle(percin_tipi_2)

    st.subheader("Emniyet Tabanı ve Kademe Eşikleri")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Fiziksel Recall (Sarı)", f"%{m['recall_sari'] * 100:.2f}", f"Hedef: %{m['hedef_recall'] * 100:.0f}")
    k2.metric("Sarı Hat Yükü", f"%{m['sari_yuku'] * 100:.1f}", f"Tavan: %{m['sari_kapasite_tavani'] * 100:.0f}")
    k3.metric("Kırmızı Hat Yükü", f"%{m['kirmizi_yuku'] * 100:.1f}")
    k4.metric("Brier Skoru", f"{m['brier']:.4f}", "Kalibrasyon")

    if m["recall_sari"] < m["hedef_recall"]:
        st.error(
            f"Emniyet tabanı karşılanmıyor: kapasite tavanı (%{m['sari_kapasite_tavani'] * 100:.0f}) içinde "
            f"fiziksel recall %{m['recall_sari'] * 100:.2f}. Hedefe ulaşmak için ikincil kontrol kapasitesi "
            f"artırılmalı veya model iyileştirilmelidir."
        )

    st.divider()
    st.subheader("Model FAI Kaydı (AS9102 Form 3 benzeri)")
    fai = m["fai"]
    b1, b2, b3 = st.columns(3)
    b1.metric("Genel Sonuç", fai["genel_sonuc"])
    b2.metric("Model Parmak İzi", fai["model_parmak_izi"])
    b3.metric("Doğrulama Tarihi", fai["dogrulama_tarihi"])
    st.dataframe(pd.DataFrame(fai["karakteristikler"]), width="stretch", hide_index=True)
    if fai["genel_sonuc"] != "UYGUN":
        st.error(
            "Bu model sürümü FAI doğrulamasından geçemedi. AS9100 §8.5.1.3 gereği, üretimde serbest "
            "bırakılmadan önce düzeltici faaliyet ve yeniden doğrulama gerekir."
        )

    st.divider()
    st.subheader("Precision-Recall Eğrisi ve Kademe Eşikleri")
    precisions, recalls, thresholds = precision_recall_curve(m["y_test"], m["olasiliklar"])

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(recalls, precisions, color="#1f4e8c", linewidth=2, label="Precision-Recall eğrisi")

    for esik, renk, etiket in [
        (m["esik_sari"], "#b8860b", "Sarı eşik (emniyet tabanı)"),
        (m["esik_kirmizi"], "#a23a2f", "Kırmızı eşik (yüksek precision)"),
    ]:
        idx = int(np.searchsorted(thresholds, esik))
        idx = min(idx, len(thresholds) - 1)
        ax.scatter(recalls[idx], precisions[idx], color=renk, s=90, zorder=5, label=etiket)

    ax.axvline(x=m["hedef_recall"], color="#a23a2f", linestyle="--", linewidth=1.2, label="%99 Recall hedefi")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{percin_tipi_2} — Kademe eşiklerinin PR eğrisi üzerindeki konumu")
    ax.legend(loc="center left")
    ax.set_xlim([0, 1.02])
    ax.set_ylim([0, 1.02])
    st.pyplot(fig)

    st.info(
        "**İki eşik neden var?** Ucuz eylem (Sarı: belgeli ikincil kontrol) yüksek recall için düşük eşikle "
        "tetiklenir; pahalı eylem (Kırmızı: iş durdurma ve MRB sevki) yalnızca yüksek precision bölgesinde "
        "devreye girer. Böylece kaçan hata riski düşük tutulurken hat gereksiz yere durdurulmaz."
    )

with sekme3:
    st.subheader("Dijital Doğum Sertifikası Kayıtları")
    if not os.path.exists(IZLENEBILIRLIK_DOSYASI):
        st.info("Henüz kayıt yok. Saha Kalite Kontrolü sekmesinden bir kontrol yapıldığında kayıt oluşur.")
    else:
        kayitlar = pd.read_csv(IZLENEBILIRLIK_DOSYASI)
        s1, s2, s3 = st.columns(3)
        s1.metric("Toplam Kontrol", len(kayitlar))
        s2.metric("Kırmızı Kademe", int((kayitlar["Kademe"] == "KIRMIZI").sum()))
        s3.metric("Sarı Kademe", int((kayitlar["Kademe"] == "SARI").sum()))
        st.dataframe(kayitlar.iloc[::-1], width="stretch", hide_index=True)
        st.download_button(
            "Kaydı CSV olarak indir",
            data=kayitlar.to_csv(index=False).encode("utf-8"),
            file_name=IZLENEBILIRLIK_DOSYASI,
            mime="text/csv",
        )
