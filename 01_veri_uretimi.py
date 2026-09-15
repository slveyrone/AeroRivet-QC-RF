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

PERCIN_TIPLERI = ("MS20470", "NAS1097")


def uret_veri(percin_tipi, n=N_SATIR, seed=RANDOM_SEED):
    if percin_tipi not in PERCIN_TIPLERI:
        raise ValueError(f"Gecersiz percin tipi: {percin_tipi}")

    rng = np.random.default_rng(seed)

    operator_tecrube_yil = rng.integers(low=1, high=16, size=n)
    deneyim_carpani = 1.0 + np.clip(6 - operator_tecrube_yil, 0, 5) * 0.08

    gercek_delik_capi = rng.normal(loc=4.76, scale=0.025 * deneyim_carpani)
    gercek_baski_kuvveti = rng.uniform(low=2000, high=4000, size=n)

    delik_capi_disi = (gercek_delik_capi < DELIK_CAPI_ALT_LIMIT) | (gercek_delik_capi > DELIK_CAPI_UST_LIMIT)
    baski_kuvveti_asiri = gercek_baski_kuvveti > BASKI_KUVVETI_UST_LIMIT
    rework_kosulu = delik_capi_disi | baski_kuvveti_asiri

    veri = {
        "Delik_Capi_mm": gercek_delik_capi + rng.normal(0, OLCUM_GURULTUSU_DELIK_CAPI, size=n),
        "Baski_Kuvveti_PSI": gercek_baski_kuvveti + rng.normal(0, OLCUM_GURULTUSU_BASKI_KUVVETI, size=n),
    }

    if percin_tipi == "NAS1097":
        gercek_havsa_derinligi = rng.normal(loc=1.20, scale=0.015 * deneyim_carpani)
        havsa_derinligi_uygunsuz = gercek_havsa_derinligi > HAVSA_DERINLIGI_UST_LIMIT
        rework_kosulu = rework_kosulu | havsa_derinligi_uygunsuz
        veri["Havsa_Derinligi_mm"] = gercek_havsa_derinligi + rng.normal(0, OLCUM_GURULTUSU_HAVSA_DERINLIGI, size=n)

    veri["Operator_Tecrube_Yil"] = operator_tecrube_yil

    rework_gerekli = rework_kosulu.astype(int)
    etiket_hatasi_maskesi = rng.random(n) < ETIKET_HATASI_ORANI
    rework_gerekli = np.where(etiket_hatasi_maskesi, 1 - rework_gerekli, rework_gerekli)
    veri["Rework_Gerekli"] = rework_gerekli

    df = pd.DataFrame(veri)
    return df, etiket_hatasi_maskesi.sum()


if __name__ == "__main__":
    for percin_tipi in PERCIN_TIPLERI:
        dosya_adi = f"havacilik_montaj_verisi_{percin_tipi}.csv"
        df, etiket_hatasi_sayisi = uret_veri(percin_tipi)
        df.to_csv(dosya_adi, index=False)

        print(f"[{percin_tipi}] {len(df)} satırlık sentetik montaj verisi üretildi -> {dosya_adi}")
        print(f"[{percin_tipi}] Ölçüm gürültüsü uygulandı ve {etiket_hatasi_sayisi} satırda insan kontrol hatası simüle edildi.")
        print(f"[{percin_tipi}] Sınıf dağılımı (Rework_Gerekli):")
        print(df["Rework_Gerekli"].value_counts(normalize=True).rename("oran"))
        print()
