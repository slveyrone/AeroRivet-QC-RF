# AeroRivet-QC-RF ✈️

Havacılık yapısal montaj (perçin/delik) süreçleri için uçtan uca bir **Makine Öğrenmesi tabanlı Kalite Kontrol** projesi. Random Forest sınıflandırma modeli ile, imalat hattındaki ölçüm verilerinden yola çıkarak bir parçanın **rework (yeniden işlem) gerektirip gerektirmediğini** tahmin eder ve sonuçları interaktif bir Streamlit dashboard üzerinden sunar.

## Proje Özeti

Uçak gövdesi montajında perçin delikleri; çap, havşa (countersink) derinliği ve baskı kuvveti gibi parametreler açısından sıkı toleranslara tabidir. Bu toleransların dışına çıkılması, yapısal bütünlüğü tehlikeye atabilecek kusurlu bağlantılara yol açar. Bu proje:

1. **FAA-H-8083-31A** (Aircraft Structural Repair) ve **NAS1097** (100° Flush/Countersunk Head Rivet) perçin/havşa tolerans standardına dayalı sentetik bir imalat veri seti üretir,
2. Bu veri üzerinde bir **Random Forest** sınıflandırıcı eğitir,
3. Uçuş güvenliği önceliğiyle **düşük bir karar eşiği (0.25)** kullanarak Recall'u optimize eder,
4. Sonuçları saha operatörlerinin kullanabileceği bir **Streamlit dashboard**'unda sunar.

## Kalite Mantığı: Neden Recall, Accuracy'den Daha Önemlidir?

Standart sınıflandırma problemlerinde Accuracy (doğruluk) yaygın bir başarı ölçütüdür. Ancak havacılık kalite kontrolünde asıl risk **False Negative** (kaçan hata) durumudur: model, tolerans dışı ve aslında rework gerektiren bir parçayı "sağlam" olarak etiketlerse, bu kusurlu parça uçağa monte edilebilir.

- **False Negative (Kaçan Hata):** Kusurlu parça "sağlam" denilip geçer → **uçuş güvenliği riski**.
- **False Positive (Gereksiz Rework):** Sağlam parça "kusurlu" denilip fazladan kontrol edilir → sadece **zaman/maliyet kaybı**.

Bu asimetri nedeniyle model, varsayılan 0.50 karar eşiği yerine **0.25 eşiği** ile çalıştırılır: bir parçanın rework olasılığı %25'i geçtiği anda "dikkat" sinyali verilir. Bu, Precision'da bir miktar kayıp pahasına Recall'u yükseltmeyi hedefler — havacılıkta "şüpheliyse incele" güvenlik felsefesinin doğrudan uygulamasıdır.

Gerçekçi ölçüm gürültüsü eklendikten sonra (bkz. aşağıdaki bölüm) 0.25 eşiği tek başına %99+ Recall hedefine tam ulaşmaz (~%94.7 Recall) — bu **beklenen ve gerçekçi** bir durumdur, çünkü gerçek üretim hatlarında ölçüm belirsizliği hiçbir zaman sıfır değildir. Test verisinde %99+ Recall'a ulaşmak için eşiğin ~0.003'e kadar düşürülmesi gerekir; bu noktada Precision ~%48'e düşer (neredeyse her iki uyarıdan biri yanlış alarmdır). Bu tam olarak dashboard'daki **canlı eşik kaydırıcısının var olma nedenidir**: kalite mühendisi, o günün risk toleransına göre Recall/Precision dengesini canlı olarak ayarlayabilir.

## Gerçekçilik Modeli: Üç Gürültü Katmanı

İlk sürümde `Rework_Gerekli` etiketi, modele verilen aynı 4 özellikten *gürültüsüz ve deterministik* bir kuralla türetiliyordu; bu da Random Forest'ın kuralı ezbere öğrenip %100 skor almasına yol açıyordu (gerçek dünyada asla görülmeyecek bir durum). Bunu düzeltmek için veri üretim script'ine üç gerçekçi gürültü katmanı eklendi:

1. **Operatör deneyimi etkisi (heteroskedastik süreç sapması):** Deneyimsiz operatörlerde (örn. 1 yıl) süreç kontrolü daha gevşektir; bu yüzden montajın **gerçek fiziksel değerleri** deneyimli operatörlere göre daha geniş bir dağılımdan (~1.4x standart sapma) üretilir.
2. **Ölçüm/sensör gürültüsü:** Kumpas, basınç sensörü ve derinlik ölçer gibi aletlerin kendi hassasiyet payı vardır. Rework kararı **gerçek (fiziksel) değere** göre verilir, ancak veri setinde (ve dolayısıyla modelde) sadece **ölçülen (gürültülü) değerler** yer alır — tıpkı gerçek bir üretim hattında olduğu gibi. Bu, mükemmel ayrımı (100% accuracy) istatistiksel olarak imkânsız kılar.
3. **İnsan kontrol/kayıt hatası:** Sınır vakalarındaki kalite kontrolör hatasını simüle etmek için etiketlerin ~%1.5'i rastgele ters çevrilir.

Bu değişikliklerden sonra elde edilen (seed=42 ile tekrarlanabilir) örnek sonuçlar:

| Metrik | Değer (Eşik = 0.25) |
|---|---|
| Accuracy | %90.10 |
| Precision | %84.27 |
| Recall | %94.65 |
| F1-Score | %89.16 |
| False Negative | 46 / 860 pozitif vaka içinden |
| %99+ Recall için gereken eşik | ~0.003 (Precision bu noktada ~%48.25'e düşer) |

## Proje Yapısı

```
AeroRivet-QC-RF/
├── 01_veri_uretimi.py       # Sentetik montaj verisi üretimi (10.000 satır)
├── 02_model_egitimi.py      # Random Forest eğitimi + threshold optimizasyonu
├── app.py                   # Streamlit kalite kontrol dashboard'u
├── requirements.txt         # Python bağımlılıkları
├── havacilik_montaj_verisi.csv  # Üretilen sentetik veri (script çalıştırılınca oluşur)
├── montaj_modeli.pkl        # Eğitilmiş model (script çalıştırılınca oluşur)
└── metrikler.pkl            # Test verisi, olasılıklar ve metrikler (script çalıştırılınca oluşur)
```

## Veri Seti Değişkenleri

> **Perçin tipi notu:** Havşa (countersink) işlemi yalnızca **NAS1097** gibi flush/gömme başlı (100° Flush Head) perçinlerde uygulanır. **MS20470** (Universal/Protruding Head, düz başlı) perçinlerde baş yüzeyden dışarı taştığı için havşa açılmaz — bu proje bu nedenle **NAS1097 tipi flush perçin montajını** temsil eder; MS20470 kapsam dışıdır.

CSV'deki `Delik_Capi_mm`, `Baski_Kuvveti_PSI` ve `Havsa_Derinligi_mm` sütunları **ölçülen** (sensör/kumpas gürültülü) değerlerdir — model de yalnızca bunları görür. Rework kararı ise arka planda üretilen **gerçek (fiziksel)** değerlere göre verilir, bu nedenle sınır civarındaki bazı satırlarda ölçülen değer ile etiket birebir örtüşmeyebilir (gerçekçi ölçüm belirsizliği).

| Değişken | Açıklama | Temel Dağılım (gerçek değer) |
|---|---|---|
| `Delik_Capi_mm` | Ölçülen delik çapı (nominal 4.76 mm) | Normal(μ=4.76, σ=0.025, operatör deneyimine göre ±1.4x) + ölçüm gürültüsü (σ=0.008) |
| `Baski_Kuvveti_PSI` | Ölçülen perçin baskı/sıkıştırma kuvveti | Uniform(2000, 4000) + ölçüm gürültüsü (σ=40) |
| `Havsa_Derinligi_mm` | Ölçülen havşa (countersink) derinliği | Normal(μ=1.20, σ=0.015, operatör deneyimine göre ±1.4x) + ölçüm gürültüsü (σ=0.006) |
| `Operator_Tecrube_Yil` | Operatör tecrübe yılı | Tam sayı, 1–15 |
| `Rework_Gerekli` | Hedef değişken (0=Sağlam, 1=Rework) | Kural tabanlı (gerçek değerlere göre) + ~%1.5 insan kontrol hatası |

**Rework kuralı (gerçek/fiziksel değerlere uygulanır):** Delik çapı `< 4.74 mm` veya `> 4.84 mm` **VEYA** baskı kuvveti `> 3500 PSI` **VEYA** havşa derinliği `> 1.25 mm` ise parça rework gerektirir. Ayrıntılar için [Gerçekçilik Modeli](#gerçekçilik-modeli-üç-gürültü-katmanı) bölümüne bakın.

## Kurulum

```bash
pip install -r requirements.txt
```

## Çalıştırma Adımları

```bash
# 1. Sentetik veri üret
python 01_veri_uretimi.py

# 2. Modeli eğit, threshold optimizasyonu yap, .pkl dosyalarını kaydet
python 02_model_egitimi.py

# 3. Dashboard'u başlat
streamlit run app.py
```

Dashboard tarayıcıda açıldığında:
- **Sekme 1 — Anlık Saha Kalite Kontrolü:** Operatör ölçüm değerlerini girer, sistem anında kalite kararı verir.
- **Sekme 2 — Model Başarı & Güvenilirlik Analizi:** Accuracy/Recall/Precision/F1 metrikleri, canlı eşik kaydırıcısı ve Precision-Recall eğrisi (%99 Recall hedef noktası işaretli) görüntülenir.

## Kullanılan Teknolojiler

- **scikit-learn** — Random Forest sınıflandırıcı
- **pandas / numpy** — veri üretimi ve işleme
- **streamlit** — interaktif dashboard
- **matplotlib / seaborn** — Precision-Recall görselleştirmesi
- **joblib** — model ve metrik serileştirme
