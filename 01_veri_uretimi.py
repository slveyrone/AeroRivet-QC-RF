import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_SATIR = 10_000

DELIK_CAPI_NOMINAL = 4.76
DELIK_CAPI_ALT_LIMIT = 4.74
DELIK_CAPI_UST_LIMIT = 4.84

BASKI_KUVVETI_NOMINAL = 3000.0
BASKI_KUVVETI_ALT_LIMIT = 2500.0
BASKI_KUVVETI_UST_LIMIT = 3500.0

HAVSA_DERINLIGI_NOMINAL = 1.20
HAVSA_DERINLIGI_ALT_LIMIT = 1.15
HAVSA_DERINLIGI_UST_LIMIT = 1.25

SUREC_SIGMA_DELIK_CAPI = 0.005
SUREC_SIGMA_BASKI_KUVVETI = 120.0
SUREC_SIGMA_HAVSA_DERINLIGI = 0.012

OLCUM_GURULTUSU_DELIK_CAPI = 0.0015
OLCUM_GURULTUSU_BASKI_KUVVETI = 35.0
OLCUM_GURULTUSU_HAVSA_DERINLIGI = 0.0035

OZEL_SEBEP_ORANI = 0.035
KONTROL_KACAK_ORANI = 0.03
KONTROL_YANLIS_RED_ORANI = 0.001

PERCIN_TIPLERI = ("MS20470", "NAS1097")


def uret_veri(percin_tipi, n=N_SATIR, seed=RANDOM_SEED):
    if percin_tipi not in PERCIN_TIPLERI:
        raise ValueError(f"Gecersiz percin tipi: {percin_tipi}")

    rng = np.random.default_rng(seed)
    havsali = percin_tipi == "NAS1097"

    operator_tecrube_yil = rng.integers(low=1, high=16, size=n)
    deneyim_carpani = 1.0 + np.clip(6 - operator_tecrube_yil, 0, 5) * 0.08

    gercek_delik_capi = rng.normal(DELIK_CAPI_NOMINAL, SUREC_SIGMA_DELIK_CAPI * deneyim_carpani)
    gercek_baski_kuvveti = rng.normal(BASKI_KUVVETI_NOMINAL, SUREC_SIGMA_BASKI_KUVVETI * deneyim_carpani)
    gercek_havsa_derinligi = rng.normal(HAVSA_DERINLIGI_NOMINAL, SUREC_SIGMA_HAVSA_DERINLIGI * deneyim_carpani)

    ozel_sebep = rng.random(n) < (OZEL_SEBEP_ORANI * deneyim_carpani)
    sebep_turu = rng.integers(0, 3 if havsali else 2, size=n)

    matkap_asinmasi = ozel_sebep & (sebep_turu == 0)
    regulator_kaymasi = ozel_sebep & (sebep_turu == 1)
    kesici_asinmasi = ozel_sebep & (sebep_turu == 2)

    gercek_delik_capi = np.where(
        matkap_asinmasi, gercek_delik_capi + rng.uniform(0.04, 0.14, size=n), gercek_delik_capi
    )
    kuvvet_sapmasi = rng.uniform(300, 700, size=n) * rng.choice([-1.0, 1.0], size=n)
    gercek_baski_kuvveti = np.where(
        regulator_kaymasi, gercek_baski_kuvveti + kuvvet_sapmasi, gercek_baski_kuvveti
    )
    gercek_havsa_derinligi = np.where(
        kesici_asinmasi, gercek_havsa_derinligi + rng.uniform(0.03, 0.10, size=n), gercek_havsa_derinligi
    )

    cap_ust = gercek_delik_capi > DELIK_CAPI_UST_LIMIT
    cap_alt = gercek_delik_capi < DELIK_CAPI_ALT_LIMIT
    kuvvet_ust = gercek_baski_kuvveti > BASKI_KUVVETI_UST_LIMIT
    kuvvet_alt = gercek_baski_kuvveti < BASKI_KUVVETI_ALT_LIMIT
    havsa_ust = gercek_havsa_derinligi > HAVSA_DERINLIGI_UST_LIMIT
    havsa_alt = gercek_havsa_derinligi < HAVSA_DERINLIGI_ALT_LIMIT

    kosullar = [cap_ust, cap_alt, kuvvet_ust, kuvvet_alt]
    nedenler = ["CAP_UST_LIMIT", "CAP_ALT_LIMIT", "KUVVET_UST_LIMIT", "KUVVET_ALT_LIMIT"]
    if havsali:
        kosullar += [havsa_ust, havsa_alt]
        nedenler += ["HAVSA_UST_LIMIT", "HAVSA_ALT_LIMIT"]

    gercek_hata = np.logical_or.reduce(kosullar)
    hata_nedeni = np.select(kosullar, nedenler, default="-")

    kacak = gercek_hata & (rng.random(n) < KONTROL_KACAK_ORANI)
    yanlis_red = (~gercek_hata) & (rng.random(n) < KONTROL_YANLIS_RED_ORANI)
    rework_gerekli = np.where(kacak, 0, np.where(yanlis_red, 1, gercek_hata.astype(int)))
    hata_nedeni = np.where(yanlis_red, "KONTROL_YANLIS_RED", hata_nedeni)

    veri = {
        "Delik_Capi_mm": gercek_delik_capi + rng.normal(0, OLCUM_GURULTUSU_DELIK_CAPI, size=n),
        "Baski_Kuvveti_PSI": gercek_baski_kuvveti + rng.normal(0, OLCUM_GURULTUSU_BASKI_KUVVETI, size=n),
    }
    if havsali:
        veri["Havsa_Derinligi_mm"] = gercek_havsa_derinligi + rng.normal(
            0, OLCUM_GURULTUSU_HAVSA_DERINLIGI, size=n
        )
    veri["Operator_Tecrube_Yil"] = operator_tecrube_yil
    veri["Rework_Gerekli"] = rework_gerekli
    veri["Hata_Nedeni"] = hata_nedeni

    ozet = {
        "ozel_sebep_sayisi": int(ozel_sebep.sum()),
        "gercek_hata_sayisi": int(gercek_hata.sum()),
        "kacak_kontrol_sayisi": int(kacak.sum()),
        "yanlis_red_sayisi": int(yanlis_red.sum()),
    }
    return pd.DataFrame(veri), ozet


def surec_yetenegi(sigma, alt_limit, ust_limit, nominal):
    return min(nominal - alt_limit, ust_limit - nominal) / (3 * sigma)


if __name__ == "__main__":
    print("Surec yetenegi (Cpk, deneyimli operator):")
    print(f"  Delik capi       : {surec_yetenegi(SUREC_SIGMA_DELIK_CAPI, DELIK_CAPI_ALT_LIMIT, DELIK_CAPI_UST_LIMIT, DELIK_CAPI_NOMINAL):.2f}")
    print(f"  Baski kuvveti    : {surec_yetenegi(SUREC_SIGMA_BASKI_KUVVETI, BASKI_KUVVETI_ALT_LIMIT, BASKI_KUVVETI_UST_LIMIT, BASKI_KUVVETI_NOMINAL):.2f}")
    print(f"  Havsa derinligi  : {surec_yetenegi(SUREC_SIGMA_HAVSA_DERINLIGI, HAVSA_DERINLIGI_ALT_LIMIT, HAVSA_DERINLIGI_UST_LIMIT, HAVSA_DERINLIGI_NOMINAL):.2f}")
    print()

    for percin_tipi in PERCIN_TIPLERI:
        dosya_adi = f"havacilik_montaj_verisi_{percin_tipi}.csv"
        df, ozet = uret_veri(percin_tipi)
        df.to_csv(dosya_adi, index=False)

        hata_orani = df["Rework_Gerekli"].mean()
        print(f"[{percin_tipi}] {len(df)} satir uretildi -> {dosya_adi}")
        print(f"[{percin_tipi}] Kayitli hata orani: %{hata_orani * 100:.2f} ({int(df['Rework_Gerekli'].sum())} adet)")
        print(f"[{percin_tipi}] Ozel sebep (alet asinmasi/regulator kaymasi): {ozet['ozel_sebep_sayisi']} adet")
        print(f"[{percin_tipi}] Kontrolor kacagi: {ozet['kacak_kontrol_sayisi']} | Yanlis red: {ozet['yanlis_red_sayisi']}")
        print(f"[{percin_tipi}] Uygunsuzluk dagilimi:")
        dagilim = df.loc[df["Rework_Gerekli"] == 1, "Hata_Nedeni"].value_counts()
        for neden, adet in dagilim.items():
            print(f"    {neden:<20} {adet}")
        print()
