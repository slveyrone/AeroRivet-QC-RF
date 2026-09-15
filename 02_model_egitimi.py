import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
KARAR_ESIGI = 0.25
TARGET = "Rework_Gerekli"

FEATURES_BY_TIP = {
    "MS20470": ["Delik_Capi_mm", "Baski_Kuvveti_PSI", "Operator_Tecrube_Yil"],
    "NAS1097": ["Delik_Capi_mm", "Baski_Kuvveti_PSI", "Havsa_Derinligi_mm", "Operator_Tecrube_Yil"],
}


def modeli_egit(percin_tipi):
    features = FEATURES_BY_TIP[percin_tipi]
    df = pd.read_csv(f"havacilik_montaj_verisi_{percin_tipi}.csv")
    X = df[features]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )

    model = RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(X_train, y_train)

    olasiliklar = model.predict_proba(X_test)[:, 1]
    tahminler = (olasiliklar >= KARAR_ESIGI).astype(int)

    accuracy = accuracy_score(y_test, tahminler)
    precision = precision_score(y_test, tahminler)
    recall = recall_score(y_test, tahminler)
    f1 = f1_score(y_test, tahminler)
    cm = confusion_matrix(y_test, tahminler)
    tn, fp, fn, tp = cm.ravel()

    print(f"=== {percin_tipi} ===")
    print(f"Karar Eşiği: {KARAR_ESIGI}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}  (Havacılık Güvenliği hedefi: %99+)")
    print(f"F1-Score : {f1:.4f}")
    print("\nHata Matrisi (Confusion Matrix):")
    print(f"                 Tahmin: Sağlam  Tahmin: Rework")
    print(f"Gerçek: Sağlam        {tn:6d}         {fp:6d}")
    print(f"Gerçek: Rework        {fn:6d}         {tp:6d}")
    print(f"\nFalse Negative (kaçan hatalı parça) sayısı: {fn}")

    precisions_curve, recalls_curve, thresholds_curve = precision_recall_curve(y_test, olasiliklar)
    hedef_uyan_idx = np.where(recalls_curve[:-1] >= 0.99)[0]
    if len(hedef_uyan_idx) > 0:
        en_iyi_idx = hedef_uyan_idx[-1]
        print(
            f"\n[Bilgi] %99+ Recall hedefini karşılayan en yüksek eşik: "
            f"{thresholds_curve[en_iyi_idx]:.3f} (bu eşikte Precision: {precisions_curve[en_iyi_idx]:.4f})"
        )
    else:
        print("\n[Bilgi] Test verisinde hiçbir eşik %99+ Recall hedefini karşılamıyor.")

    model_dosyasi = f"montaj_modeli_{percin_tipi}.pkl"
    metrik_dosyasi = f"metrikler_{percin_tipi}.pkl"

    joblib.dump(model, model_dosyasi)

    metrikler = {
        "percin_tipi": percin_tipi,
        "X_test": X_test,
        "y_test": y_test,
        "olasiliklar": olasiliklar,
        "karar_esigi": KARAR_ESIGI,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "features": features,
    }
    joblib.dump(metrikler, metrik_dosyasi)

    print(f"\nModel kaydedildi -> {model_dosyasi}")
    print(f"Metrikler kaydedildi -> {metrik_dosyasi}\n")


if __name__ == "__main__":
    for percin_tipi in FEATURES_BY_TIP:
        modeli_egit(percin_tipi)
