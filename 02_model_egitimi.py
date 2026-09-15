import hashlib
import io
import json
from datetime import date

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
TARGET = "Rework_Gerekli"

HEDEF_RECALL = 0.99
KIRMIZI_HEDEF_PRECISION = 0.90
SARI_KAPASITE_TAVANI = 0.15
KONTROL_HATASI_ETIKETI = "KONTROL_YANLIS_RED"

FAI_KABUL_NEDEN_RECALL = 0.95
FAI_KABUL_BRIER = 0.05

FEATURES_BY_TIP = {
    "MS20470": ["Delik_Capi_mm", "Baski_Kuvveti_PSI", "Operator_Tecrube_Yil"],
    "NAS1097": ["Delik_Capi_mm", "Baski_Kuvveti_PSI", "Havsa_Derinligi_mm", "Operator_Tecrube_Yil"],
}


def esik_sec_emniyet_tabani(fiziksel_hata, olasiliklar, kapasite_tavani):
    adaylar = np.unique(olasiliklar)
    for esik in adaylar:
        yuk = float((olasiliklar >= esik).mean())
        if yuk <= kapasite_tavani:
            recall = float((olasiliklar >= esik)[fiziksel_hata].mean()) if fiziksel_hata.any() else 0.0
            return float(esik), recall, yuk
    return float(adaylar.max()), 0.0, 0.0


def hedef_icin_gereken_yuk(fiziksel_hata, olasiliklar, hedef_recall):
    if not fiziksel_hata.any():
        return None
    esik = float(olasiliklar[fiziksel_hata].min())
    isaretli = olasiliklar >= esik
    recall = float(isaretli[fiziksel_hata].mean())
    if recall < hedef_recall:
        return None
    return esik, recall, float(isaretli.mean())


def esik_sec_precision_kisitli(y_true, olasiliklar, hedef_precision):
    precisions, recalls, thresholds = precision_recall_curve(y_true, olasiliklar)
    uygun = np.where(precisions[:-1] >= hedef_precision)[0]
    if len(uygun) == 0:
        f1_skorlari = 2 * precisions[:-1] * recalls[:-1] / np.maximum(precisions[:-1] + recalls[:-1], 1e-9)
        idx = int(np.argmax(f1_skorlari))
    else:
        idx = uygun[0]
    return float(thresholds[idx]), float(precisions[idx]), float(recalls[idx])


def model_parmak_izi(model):
    tampon = io.BytesIO()
    joblib.dump(model, tampon)
    return hashlib.sha256(tampon.getvalue()).hexdigest()[:12]


def fai_dogrulamasi(percin_tipi, model, esik, fiziksel_hata, olasiliklar, nedenler, brier):
    tahminler = (olasiliklar >= esik).astype(int)
    karakteristikler = []

    for neden in sorted(set(nedenler[fiziksel_hata])):
        maske = fiziksel_hata & (nedenler == neden)
        yakalanan = int(tahminler[maske].sum())
        toplam = int(maske.sum())
        oran = yakalanan / toplam if toplam else 0.0
        karakteristikler.append({
            "karakteristik": f"Uygunsuzluk tespiti: {neden}",
            "yontem": "Test seti alt grup recall",
            "kabul_kriteri": f">= {FAI_KABUL_NEDEN_RECALL:.2f}",
            "olculen": round(oran, 4),
            "adet": f"{yakalanan}/{toplam}",
            "sonuc": "UYGUN" if oran >= FAI_KABUL_NEDEN_RECALL else "UYGUNSUZ",
        })

    genel_recall = float(tahminler[fiziksel_hata].mean()) if fiziksel_hata.any() else 0.0
    karakteristikler.append({
        "karakteristik": "Genel Recall (MRB dogrulanmis uygunsuzluklar)",
        "yontem": "Fiziksel uygunsuzluk alt kumesinde recall",
        "kabul_kriteri": f">= {HEDEF_RECALL:.2f}",
        "olculen": round(genel_recall, 4),
        "adet": f"{int(tahminler[fiziksel_hata].sum())}/{int(fiziksel_hata.sum())}",
        "sonuc": "UYGUN" if genel_recall >= HEDEF_RECALL else "UYGUNSUZ",
    })
    karakteristikler.append({
        "karakteristik": "Olasilik kalibrasyonu",
        "yontem": "Brier skoru",
        "kabul_kriteri": f"<= {FAI_KABUL_BRIER:.2f}",
        "olculen": round(float(brier), 4),
        "adet": "-",
        "sonuc": "UYGUN" if brier <= FAI_KABUL_BRIER else "UYGUNSUZ",
    })

    return {
        "belge_tipi": "Model FAI Kaydi (AS9102 Form 3 benzeri karakteristik dogrulamasi)",
        "percin_tipi": percin_tipi,
        "model_parmak_izi": model_parmak_izi(model),
        "dogrulama_tarihi": date.today().isoformat(),
        "karar_esigi": round(esik, 6),
        "ozellikler": FEATURES_BY_TIP[percin_tipi],
        "karakteristikler": karakteristikler,
        "genel_sonuc": "UYGUN" if all(k["sonuc"] == "UYGUN" for k in karakteristikler) else "UYGUNSUZ",
        "kalite_muhendisi_onayi": "",
    }


def modeli_egit(percin_tipi):
    features = FEATURES_BY_TIP[percin_tipi]
    df = pd.read_csv(f"havacilik_montaj_verisi_{percin_tipi}.csv")

    egitim_df, test_df = train_test_split(
        df, test_size=0.20, random_state=RANDOM_SEED, stratify=df[TARGET]
    )

    X_train, y_train = egitim_df[features], egitim_df[TARGET]
    X_test, y_test = test_df[features], test_df[TARGET]
    nedenler = test_df["Hata_Nedeni"].to_numpy()

    model = RandomForestClassifier(
        n_estimators=400,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
    )
    model.fit(X_train, y_train)

    olasiliklar = model.predict_proba(X_test)[:, 1]
    fiziksel_hata = (y_test.to_numpy() == 1) & (nedenler != KONTROL_HATASI_ETIKETI)

    esik_sari, recall_sari, sari_yuku = esik_sec_emniyet_tabani(
        fiziksel_hata, olasiliklar, SARI_KAPASITE_TAVANI
    )
    esik_kirmizi, precision_kirmizi, recall_kirmizi = esik_sec_precision_kisitli(
        y_test, olasiliklar, KIRMIZI_HEDEF_PRECISION
    )
    brier = brier_score_loss(y_test, olasiliklar)

    tahminler = (olasiliklar >= esik_sari).astype(int)
    cm = confusion_matrix(y_test, tahminler)
    tn, fp, fn, tp = cm.ravel()

    kirmizi_yuku = float((olasiliklar >= esik_kirmizi).mean())
    kacan = int(fiziksel_hata.sum() - tahminler[fiziksel_hata].sum())
    hedef_durumu = "KARSILANDI" if recall_sari >= HEDEF_RECALL else "KARSILANMADI"

    print(f"=== {percin_tipi} ===")
    print(f"Egitim hata orani: %{y_train.mean() * 100:.2f} | Test: {int(y_test.sum())} etiketli, {int(fiziksel_hata.sum())} fiziksel uygunsuzluk")
    print()
    print(f"SARI esik (kapasite tavani %{SARI_KAPASITE_TAVANI * 100:.0f}): {esik_sari:.4f}")
    print(f"  Fiziksel recall: {recall_sari:.4f} [{hedef_durumu}] | Hat yuku: %{sari_yuku * 100:.1f} | Kacan: {kacan}")
    gereken = hedef_icin_gereken_yuk(fiziksel_hata, olasiliklar, HEDEF_RECALL)
    if recall_sari < HEDEF_RECALL:
        if gereken is None:
            print(f"  UYARI: Hicbir esik %{HEDEF_RECALL * 100:.0f} fiziksel recall saglamiyor (model yetersiz).")
        else:
            print(f"  UYARI: %{HEDEF_RECALL * 100:.0f} recall icin gereken hat yuku: %{gereken[2] * 100:.1f} (esik {gereken[0]:.4f})")
    print(f"KIRMIZI esik (Precision >= {KIRMIZI_HEDEF_PRECISION:.2f}): {esik_kirmizi:.4f}")
    print(f"  Recall: {recall_kirmizi:.4f} | Precision: {precision_kirmizi:.4f} | Hat yuku: %{kirmizi_yuku * 100:.1f}")
    print()
    print(f"Accuracy (sari esikte): {accuracy_score(y_test, tahminler):.4f}")
    print(f"F1-Score (sari esikte): {f1_score(y_test, tahminler, zero_division=0):.4f}")
    print(f"Precision (sari esikte): {precision_score(y_test, tahminler, zero_division=0):.4f}")
    print(f"Brier skoru (kalibrasyon): {brier:.4f}")
    print()
    print("Hata Matrisi (sari esik, kayitli etikete gore):")
    print("                 Tahmin: Saglam  Tahmin: Uyari")
    print(f"Gercek: Saglam        {tn:6d}         {fp:6d}")
    print(f"Gercek: Rework        {fn:6d}         {tp:6d}")
    print()

    fai = fai_dogrulamasi(percin_tipi, model, esik_sari, fiziksel_hata, olasiliklar, nedenler, brier)
    print(f"Model FAI dogrulamasi: {fai['genel_sonuc']} (parmak izi {fai['model_parmak_izi']})")
    for k in fai["karakteristikler"]:
        print(f"  [{k['sonuc']:<9}] {k['karakteristik']:<42} olculen={k['olculen']} ({k['adet']})")

    model_dosyasi = f"montaj_modeli_{percin_tipi}.pkl"
    metrik_dosyasi = f"metrikler_{percin_tipi}.pkl"
    fai_dosyasi = f"model_fai_kaydi_{percin_tipi}.json"

    joblib.dump(model, model_dosyasi)
    with open(fai_dosyasi, "w", encoding="utf-8") as f:
        json.dump(fai, f, ensure_ascii=False, indent=2)

    joblib.dump({
        "percin_tipi": percin_tipi,
        "features": features,
        "model_parmak_izi": fai["model_parmak_izi"],
        "X_test": X_test,
        "y_test": y_test,
        "olasiliklar": olasiliklar,
        "hata_nedenleri": nedenler,
        "fiziksel_hata": fiziksel_hata,
        "esik_sari": esik_sari,
        "esik_kirmizi": esik_kirmizi,
        "hedef_recall": HEDEF_RECALL,
        "recall_sari": recall_sari,
        "kirmizi_hedef_precision": KIRMIZI_HEDEF_PRECISION,
        "sari_kapasite_tavani": SARI_KAPASITE_TAVANI,
        "sari_yuku": sari_yuku,
        "kirmizi_yuku": kirmizi_yuku,
        "brier": float(brier),
        "confusion_matrix": cm,
        "fai": fai,
    }, metrik_dosyasi)

    print(f"\nKaydedildi -> {model_dosyasi} | {metrik_dosyasi} | {fai_dosyasi}\n")


if __name__ == "__main__":
    for percin_tipi in FEATURES_BY_TIP:
        modeli_egit(percin_tipi)
