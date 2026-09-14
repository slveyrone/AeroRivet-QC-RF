import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_SATIR = 10_000

DELIK_CAPI_ALT_LIMIT = 4.74
DELIK_CAPI_UST_LIMIT = 4.84
BASKI_KUVVETI_UST_LIMIT = 3500
HAVSA_DERINLIGI_UST_LIMIT = 1.25

OLCUM_GURULTUSU_DELIK_CAPI = 0.008
OLCUM_GURULTUSU_BASKI_KUVVETI = 40
OLCUM_GURULTUSU_HAVSA_DERINLIGI = 0.006

ETIKET_HATASI_ORANI = 0.015


def uret_veri(n=N_SATIR, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)

    operator_tecrube_yil = rng.integers(low=1, high=16, size=n)

    deneyim_carpani = 1.0 + np.clip(6 - operator_tecrube_yil, 0, 5) * 0.08

    gercek_delik_capi = rng.normal(loc=4.76, scale=0.025 * deneyim_carpani)
    gercek_baski_kuvveti = rng.uniform(low=2000, high=4000, size=n)
    gercek_havsa_derinligi = rng.normal(loc=1.20, scale=0.015 * deneyim_carpani)

    delik_capi_disi = (gercek_delik_capi < DELIK_CAPI_ALT_LIMIT) | (gercek_delik_capi > DELIK_CAPI_UST_LIMIT)
    baski_kuvveti_asiri = gercek_baski_kuvveti > BASKI_KUVVETI_UST_LIMIT
    havsa_derinligi_uygunsuz = gercek_havsa_derinligi > HAVSA_DERINLIGI_UST_LIMIT

    rework_gerekli = (delik_capi_disi | baski_kuvveti_asiri | havsa_derinligi_uygunsuz).astype(int)

    etiket_hatasi_maskesi = rng.random(n) < ETIKET_HATASI_ORANI
    rework_gerekli = np.where(etiket_hatasi_maskesi, 1 - rework_gerekli, rework_gerekli)

    olculen_delik_capi = gercek_delik_capi + rng.normal(0, OLCUM_GURULTUSU_DELIK_CAPI, size=n)
    olculen_baski_kuvveti = gercek_baski_kuvveti + rng.normal(0, OLCUM_GURULTUSU_BASKI_KUVVETI, size=n)
    olculen_havsa_derinligi = gercek_havsa_derinligi + rng.normal(0, OLCUM_GURULTUSU_HAVSA_DERINLIGI, size=n)

    df = pd.DataFrame({
        "Delik_Capi_mm": olculen_delik_capi,
        "Baski_Kuvveti_PSI": olculen_baski_kuvveti,
        "Havsa_Derinligi_mm": olculen_havsa_derinligi,
        "Operator_Tecrube_Yil": operator_tecrube_yil,
        "Rework_Gerekli": rework_gerekli,
    })
    return df, etiket_hatasi_maskesi.sum()


if __name__ == "__main__":
    df, etiket_hatasi_sayisi = uret_veri()
    df.to_csv("havacilik_montaj_verisi.csv", index=False)

    print(f"{len(df)} satırlık sentetik montaj verisi üretildi -> havacilik_montaj_verisi.csv")
    print(f"Ölçüm gürültüsü uygulandı (kumpas/sensör hassasiyeti) ve {etiket_hatasi_sayisi} satırda insan kontrol hatası simüle edildi.")
    print("\nSınıf dağılımı (Rework_Gerekli):")
    print(df["Rework_Gerekli"].value_counts(normalize=True).rename("oran"))
    print("\nBetimsel istatistikler:")
    print(df.describe())
