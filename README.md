# AeroRivet-QC-RF

*Türkçe | [English](README.en.md)*

Havacılık yapısal montaj (perçin/delik) süreçleri için uçtan uca bir **Makine Öğrenmesi tabanlı Kalite Kontrol** projesi. İki perçin tipi (**MS20470** ve **NAS1097**) için ayrı Random Forest modelleri, gerçekçi bir üretim hattı süreç modeli üzerinde eğitilir; kararlar ikili kabul/red yerine **üç kademeli müdahale mimarisi** ile verilir ve her kontrol izlenebilirlik kaydına işlenir.

## Proje Özeti

Uçak gövdesi montajında perçin delikleri; çap, havşa (countersink) derinliği ve baskı/çakma kuvveti açısından sıkı toleranslara tabidir. Ancak tüm perçin tipleri aynı kontrolden geçmez: **MS20470** (Universal/Protruding Head, bombe başlı) perçinlerde havşa açılmazken, **NAS1097** (100° Flush/Countersunk Head) perçinlerde havşa derinliği de kritik bir parametredir. Bu proje:

1. **FAA-H-8083-31A** referanslı, Cpk ≈ 1.33 yetenekli bir süreç ve gerçekçi (%2 civarı) uygunsuzluk oranı üreten sentetik bir veri seti kurar,
2. Her perçin tipi için sınıf ağırlıklı ayrı bir **Random Forest** sınıflandırıcı eğitir,
3. Sabit eşik yerine **kapasite kısıtlı emniyet tabanı** ve **yüksek precision** olmak üzere iki eşiği veriden türetir,
4. Sonuçları, perçin tipi ve parça kritikliğine göre kademelendiren, kalibrasyon kapısı ve izlenebilirlik kaydı içeren bir **Streamlit dashboard**'unda sunar.

## Kademeli Müdahale Mimarisi

İkili "kilitli/kilitsiz" mantığı yerine, eylemin maliyetiyle eşiğin sıkılığını eşleştiren üç kademe kullanılır:

| Kademe | Eşik mantığı | Gereken eylem | Hat yükü |
|---|---|---|---|
| **Yeşil** | Sarı eşiğin altı | Standart SPC izlemesi, ek adım yok | ~%85 |
| **Sarı** | Kapasite tavanı içinde recall'u maksimize eden eşik | Belgeli ikincil kontrol — iş durmaz | ~%15 |
| **Kırmızı** | Precision ≥ %90 sağlayan eşik | İş durur, MRB'ye sevk, Sınıf A'da NDT tetiklenir | ~%2 |

**Ucuz eylem yüksek recall ile, pahalı eylem yüksek precision ile tetiklenir.** Ayrıca **Sınıf A** (birincil yapısal) parçalarda sarı kademe otomatik olarak kırmızıya yükseltilir — operasyon × parça kritikliği matrisinin kod karşılığı budur.

## Gerçekçi Süreç Modeli

Erken sürümlerde uygunsuzluk oranı ~%40'tı; bu, gerçek bir üretim hattını temsil etmiyordu. Mevcut veri üretimi dört katmandan oluşur:

1. **Yetenekli temel süreç:** Tüm karakteristikler tolerans bandı içinde merkezlenmiş, Cpk ≈ 1.33–1.39. Yalnız ortak sebep değişkenliğiyle uygunsuzluk oranı binde birin altındadır.
2. **Özel sebep sapmaları:** Gerçek hataların ana kaynağı — matkap aşınması (delik büyümesi), regülatör kayması (kuvvet sapması) ve havşa kesici aşınması (derinlik sapması). Parçaların ~%4'ü bir sapma koşulu altında üretilir; uygunsuzlukların çoğu buradan gelir. SPC'nin var olma nedeni de tam olarak budur.
3. **Ölçüm/sensör gürültüsü:** Karar gerçek fiziksel değere göre verilir; veri setinde yalnız ölçülen (gürültülü) değerler yer alır. Gauge değişkenliği süreç değişkenliğinin ~%30'u seviyesinde tutulmuştur.
4. **Asimetrik insan kontrol hatası:** Gerçek uygunsuzlukların %3'ü gözden kaçar (`kaçak`), sağlam parçaların binde biri yanlışlıkla reddedilir (`KONTROL_YANLIS_RED`). Bu asimetri sahadaki davranışı yansıtır.

Sonuç: MS20470 için %1.90, NAS1097 için %2.32 kayıtlı uygunsuzluk oranı.

## Emniyet Tabanı: %99 Recall Neden Doğrudan Dayatılamaz?

Bu proje geliştirilirken ortaya çıkan en önemli teknik bulgu: **%99 recall hedefi, gürültülü etikete karşı kovalanamaz.** `KONTROL_YANLIS_RED` satırları fiziksel olarak kusursuz ama kontrolörün yanlışlıkla reddettiği parçalardır; model onlara doğru şekilde ~0 olasılık verir. Recall hedefi bu etiketlere karşı dayatıldığında eşik sıfıra çöker ve **hattın %100'ü uyarıya düşer** — sistem kullanılamaz hale gelir.

Çözüm iki parçalıdır:

- **Emniyet tabanı, MRB tarafından doğrulanmış fiziksel uygunsuzluklara karşı ölçülür**, ham kontrolör etiketine karşı değil. Gerçek hayattaki karşılığı: doğruluk referansı MRB dispozisyonudur, ilk kontrol etiketi değil.
- **Eşik, operasyonel kapasite tavanı altında seçilir** (varsayılan: ikincil kontrole gidebilecek parçaların en fazla %15'i). Hedef bu kapasite içinde karşılanamıyorsa sistem bunu gizlemez; gereken kapasiteyi hesaplayıp raporlar.

## Model FAI Protokolü

AS9102 Rev C Form 3'ün karakteristik bazlı doğrulama mantığı model sürümlerine taşınmıştır: **yeni model sürümü, yeni bir "ilk parça"dır.** Her eğitim sonunda üretilen `model_fai_kaydi_<tip>.json`, model parmak izi (SHA-256 kısaltması), karar eşiği ve her uygunsuzluk türü için ayrı recall doğrulaması içerir; kalite mühendisi onay alanı boş bırakılır.

Kabul kriterleri: uygunsuzluk türü bazında recall ≥ %95, fiziksel uygunsuzluklarda genel recall ≥ %99, Brier skoru ≤ 0.05.

**Mevcut durum — protokol işliyor:** NAS1097 modeli FAI doğrulamasından **UYGUNSUZ** dönmektedir. %15 kapasite tavanı içinde bir havşa üst limit uygunsuzluğunu kaçırıyor (recall %97.67). Script, hedefe ulaşmak için ikincil kontrol kapasitesinin %16'ya çıkarılması gerektiğini raporlar. Model bilerek "geçene kadar" ayarlanmamıştır; kapının gerçekten çalıştığını göstermek için bu sonuç olduğu gibi bırakılmıştır. Dashboard bu modeli her açılışta uyarı bandıyla işaretler.

## Güncel Sonuçlar (seed=42 ile tekrarlanabilir)

| Metrik | MS20470 | NAS1097 |
|---|---|---|
| Kayıtlı uygunsuzluk oranı | %1.90 | %2.32 |
| Sarı eşik | 0.0073 | 0.0049 |
| Fiziksel recall (sarı eşikte) | %100.00 (37/37) | %97.67 (42/43) |
| Sarı hat yükü | %15.0 | %15.0 |
| Kırmızı eşik | 0.7609 | 0.7105 |
| Kırmızı precision / recall | %91.67 / %86.84 | %90.48 / %82.61 |
| Kırmızı hat yükü | %1.8 | %2.1 |
| Brier skoru (kalibrasyon) | 0.0035 | 0.0047 |
| Model FAI sonucu | UYGUN | **UYGUNSUZ** |

## Proje Yapısı

```
AeroRivet-QC-RF/
├── 01_veri_uretimi.py                      # Yetenekli süreç + özel sebep modeliyle sentetik veri üretimi
├── 02_model_egitimi.py                     # Sınıf ağırlıklı RF eğitimi, iki eşik türetimi, Model FAI doğrulaması
├── app.py                                  # Kademeli müdahale dashboard'u (3 sekme)
├── requirements.txt                        # Python bağımlılıkları
├── havacilik_montaj_verisi_<tip>.csv       # Üretilen sentetik veri (script çalıştırılınca oluşur)
├── montaj_modeli_<tip>.pkl                 # Eğitilmiş modeller (script çalıştırılınca oluşur)
├── metrikler_<tip>.pkl                     # Test verisi, eşikler, FAI kaydı (script çalıştırılınca oluşur)
├── model_fai_kaydi_<tip>.json              # Model FAI doğrulama kaydı (script çalıştırılınca oluşur)
└── izlenebilirlik_kaydi.csv                # Dijital doğum sertifikası kayıtları (dashboard kullanıldıkça birikir)
```

## Veri Seti Değişkenleri

Ölçüm sütunları **ölçülen** (gürültülü) değerlerdir; model yalnız bunları görür. `Hata_Nedeni` sütunu modele özellik olarak verilmez — yalnızca Model FAI doğrulamasında uygunsuzluk türü bazlı recall hesabı için kullanılır.

| Değişken | Açıklama | Tolerans / Dağılım | MS20470 | NAS1097 |
|---|---|---|:---:|:---:|
| `Delik_Capi_mm` | Ölçülen delik çapı | Nominal 4.76 · limit 4.74–4.84 · σ=0.005 | Var | Var |
| `Baski_Kuvveti_PSI` | Ölçülen çakma/baskı kuvveti | Nominal 3000 · limit 2500–3500 · σ=120 | Var | Var |
| `Havsa_Derinligi_mm` | Ölçülen havşa derinliği | Nominal 1.20 · limit 1.15–1.25 · σ=0.012 | Yok | Var |
| `Operator_Tecrube_Yil` | Operatör tecrübe yılı | Tam sayı 1–15; deneyimsizlikte süreç sapması ×1.4 | Var | Var |
| `Rework_Gerekli` | Hedef değişken (0=Sağlam, 1=Rework) | Kural tabanlı + asimetrik kontrol hatası | Var | Var |
| `Hata_Nedeni` | Uygunsuzluk türü (FAI doğrulaması için) | Özellik olarak kullanılmaz | Var | Var |

Alt limitler bilinçli olarak eklenmiştir: yetersiz çakma kuvveti eksik şişirilmiş kapanış başına, sığ havşa başın yüzeyden taşmasına, derin havşa ise knife-edge koşuluna yol açar — üçü de gerçek ret nedenleridir.

## Kurulum ve Çalıştırma

```bash
pip install -r requirements.txt

python 01_veri_uretimi.py     # Her iki perçin tipi için sentetik veri
python 02_model_egitimi.py    # Eğitim, eşik türetimi, Model FAI doğrulaması
streamlit run app.py          # Dashboard
```

Dashboard sekmeleri:
- **Saha Kalite Kontrolü:** Perçin tipi ve parça kritiklik sınıfı seçilir, ölçüm aleti kalibrasyon geçerliliği doğrulanır (süresi geçmişse kontrol yapılamaz), ölçümler girilir ve kademeli karar üretilir. Her kontrol izlenebilirlik kaydına yazılır.
- **Model Başarımı ve FAI Kaydı:** Emniyet tabanı durumu, kademe eşikleri, Brier skoru, karakteristik bazlı FAI tablosu ve eşiklerin PR eğrisi üzerindeki konumu.
- **İzlenebilirlik Kaydı:** Dijital doğum sertifikası kayıtları ve CSV dışa aktarımı.

## Kapsam Dışı

Aşağıdakiler mimari raporda tanımlıdır ancak bu kod tabanında uygulanmamıştır; kurumsal sistem entegrasyonu gerektirir: MES/PLM entegrasyonu, akıllı alet (smart tool) fiziksel inhibisyonu, MRB iş akışı ve imza yönetimi, Operasyonel Yetkinlik Profili (OYP) ve OJT kayıtları, NDT ekipman entegrasyonu, sürüklenme izleme otomasyonu.

## Kullanılan Teknolojiler

- **scikit-learn** — sınıf ağırlıklı Random Forest, PR eğrisi, Brier skoru
- **pandas / numpy** — süreç simülasyonu ve veri işleme
- **streamlit** — kademeli müdahale dashboard'u
- **matplotlib / seaborn** — PR eğrisi görselleştirmesi
- **joblib** — model ve metrik serileştirme
