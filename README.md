# AeroRivet-QC-RF ✈️

*Türkçe | [English](README.en.md)*

Havacılık yapısal montaj (perçin/delik) süreçleri için uçtan uca bir **Makine Öğrenmesi tabanlı Kalite Kontrol** projesi. İki farklı perçin tipi (**MS20470** ve **NAS1097**) için ayrı ayrı eğitilmiş Random Forest sınıflandırma modelleri ile, imalat hattındaki ölçüm verilerinden yola çıkarak bir parçanın **rework (yeniden işlem) gerektirip gerektirmediğini** tahmin eder ve sonuçları interaktif bir Streamlit dashboard üzerinden sunar.

## Proje Özeti

Uçak gövdesi montajında perçin delikleri; çap, havşa (countersink) derinliği ve baskı kuvveti gibi parametreler açısından sıkı toleranslara tabidir. Bu toleransların dışına çıkılması, yapısal bütünlüğü tehlikeye atabilecek kusurlu bağlantılara yol açar. Ancak tüm perçin tipleri aynı kontrolden geçmez: **MS20470** (Universal/Protruding Head, düz başlı) perçinlerde havşa açılmazken, **NAS1097** (100° Flush/Countersunk Head) perçinlerde havşa derinliği de kritik bir kalite parametresidir. Bu proje:

1. **FAA-H-8083-31A** (Aircraft Structural Repair) standardına dayalı, her perçin tipi için kendi tolerans kurallarını uygulayan iki ayrı sentetik imalat veri seti üretir,
2. Her perçin tipi için kendi özellik setiyle ayrı bir **Random Forest** sınıflandırıcı eğitir,
3. Uçuş güvenliği önceliğiyle **düşük bir karar eşiği (0.25)** kullanarak Recall'u optimize eder,
4. Kullanıcının önce perçin tipini seçtiği, sonrasında ilgili parametreleri girdiği bir **Streamlit dashboard**'unda sonuçları sunar.

## İki Perçin Tipi: MS20470 ve NAS1097

| | **MS20470** | **NAS1097** |
|---|---|---|
| Baş tipi | Universal / Protruding Head (düz baş) | 100° Flush / Countersunk Head (gömme baş) |
| Havşa (countersink) | ❌ Açılmaz — baş yüzeyden dışarı taşar | ✅ Açılır — baş yüzeyle aynı hizada oturur |
| Model girdi özellikleri | Delik Çapı, Baskı Kuvveti, Operatör Tecrübesi (3 özellik) | Delik Çapı, Baskı Kuvveti, Havşa Derinliği, Operatör Tecrübesi (4 özellik) |
| Model/veri dosyası eki | `_MS20470` | `_NAS1097` |

Bu iki perçin tipi **farklı özellik şemalarına** sahip olduğu için tek bir model yerine **iki bağımsız model** eğitilir; MS20470 verisine sahte/doldurma bir havşa değeri atamak yanıltıcı olacağından tercih edilmemiştir. Dashboard'da kullanıcı önce perçin tipini seçer, form alanları seçime göre otomatik değişir.

## Kalite Mantığı: Neden Recall, Accuracy'den Daha Önemlidir?

Standart sınıflandırma problemlerinde Accuracy (doğruluk) yaygın bir başarı ölçütüdür. Ancak havacılık kalite kontrolünde asıl risk **False Negative** (kaçan hata) durumudur: model, tolerans dışı ve aslında rework gerektiren bir parçayı "sağlam" olarak etiketlerse, bu kusurlu parça uçağa monte edilebilir.

- **False Negative (Kaçan Hata):** Kusurlu parça "sağlam" denilip geçer → **uçuş güvenliği riski**.
- **False Positive (Gereksiz Rework):** Sağlam parça "kusurlu" denilip fazladan kontrol edilir → sadece **zaman/maliyet kaybı**.

Bu asimetri nedeniyle her iki model de varsayılan 0.50 karar eşiği yerine **0.25 eşiği** ile çalıştırılır: bir parçanın rework olasılığı %25'i geçtiği anda "dikkat" sinyali verilir. Bu, Precision'da bir miktar kayıp pahasına Recall'u yükseltmeyi hedefler — havacılıkta "şüpheliyse incele" güvenlik felsefesinin doğrudan uygulamasıdır.

Gerçekçi ölçüm gürültüsü eklendikten sonra (bkz. aşağıdaki bölüm) 0.25 eşiği tek başına %99+ Recall hedefine tam ulaşmaz (~%94 Recall) — bu **beklenen ve gerçekçi** bir durumdur, çünkü gerçek üretim hatlarında ölçüm belirsizliği hiçbir zaman sıfır değildir. Bu tam olarak dashboard'daki **canlı eşik kaydırıcısının var olma nedenidir**: kalite mühendisi, o günün risk toleransına göre Recall/Precision dengesini canlı olarak ayarlayabilir.

## Gerçekçilik Modeli: Üç Gürültü Katmanı

İlk sürümde `Rework_Gerekli` etiketi, modele verilen aynı özelliklerden *gürültüsüz ve deterministik* bir kuralla türetiliyordu; bu da Random Forest'ın kuralı ezbere öğrenip %100 skor almasına yol açıyordu (gerçek dünyada asla görülmeyecek bir durum). Bunu düzeltmek için veri üretim script'ine üç gerçekçi gürültü katmanı eklendi (her iki perçin tipi için de geçerlidir):

1. **Operatör deneyimi etkisi (heteroskedastik süreç sapması):** Deneyimsiz operatörlerde (örn. 1 yıl) süreç kontrolü daha gevşektir; bu yüzden montajın **gerçek fiziksel değerleri** deneyimli operatörlere göre daha geniş bir dağılımdan (~1.4x standart sapma) üretilir.
2. **Ölçüm/sensör gürültüsü:** Kumpas, basınç sensörü ve derinlik ölçer gibi aletlerin kendi hassasiyet payı vardır. Rework kararı **gerçek (fiziksel) değere** göre verilir, ancak veri setinde (ve dolayısıyla modelde) sadece **ölçülen (gürültülü) değerler** yer alır — tıpkı gerçek bir üretim hattında olduğu gibi. Bu, mükemmel ayrımı (100% accuracy) istatistiksel olarak imkânsız kılar.
3. **İnsan kontrol/kayıt hatası:** Sınır vakalarındaki kalite kontrolör hatasını simüle etmek için etiketlerin ~%1.5'i rastgele ters çevrilir.

Bu değişikliklerden sonra elde edilen (seed=42 ile tekrarlanabilir) örnek sonuçlar:

| Metrik | MS20470 (Eşik = 0.25) | NAS1097 (Eşik = 0.25) |
|---|---|---|
| Accuracy | %90.35 | %90.40 |
| Precision | %84.97 | %84.50 |
| Recall | %94.03 | %94.97 |
| F1-Score | %89.27 | %89.43 |
| False Negative | 51 / 854 pozitif vaka içinden | 43 / 855 pozitif vaka içinden |
| %99+ Recall için gereken eşik | ~0.010 (Precision ~%58.55'e düşer) | ~0.023 (Precision ~%61.38'e düşer) |

## Proje Yapısı

```
AeroRivet-QC-RF/
├── 01_veri_uretimi.py                      # Her iki perçin tipi için sentetik veri üretimi (10.000 satır)
├── 02_model_egitimi.py                     # Her perçin tipi için ayrı Random Forest eğitimi + threshold analizi
├── app.py                                  # Perçin tipi seçimli Streamlit kalite kontrol dashboard'u
├── requirements.txt                        # Python bağımlılıkları
├── havacilik_montaj_verisi_MS20470.csv     # MS20470 sentetik verisi (script çalıştırılınca oluşur)
├── havacilik_montaj_verisi_NAS1097.csv     # NAS1097 sentetik verisi (script çalıştırılınca oluşur)
├── montaj_modeli_MS20470.pkl               # MS20470 eğitilmiş model (script çalıştırılınca oluşur)
├── montaj_modeli_NAS1097.pkl               # NAS1097 eğitilmiş model (script çalıştırılınca oluşur)
├── metrikler_MS20470.pkl                   # MS20470 test verisi/metrikleri (script çalıştırılınca oluşur)
└── metrikler_NAS1097.pkl                   # NAS1097 test verisi/metrikleri (script çalıştırılınca oluşur)
```

## Veri Seti Değişkenleri

CSV'lerdeki ölçüm sütunları **ölçülen** (sensör/kumpas gürültülü) değerlerdir — model de yalnızca bunları görür. Rework kararı ise arka planda üretilen **gerçek (fiziksel)** değerlere göre verilir, bu nedenle sınır civarındaki bazı satırlarda ölçülen değer ile etiket birebir örtüşmeyebilir (gerçekçi ölçüm belirsizliği).

| Değişken | Açıklama | Temel Dağılım (gerçek değer) | MS20470 | NAS1097 |
|---|---|---|:---:|:---:|
| `Delik_Capi_mm` | Ölçülen delik çapı (nominal 4.76 mm) | Normal(μ=4.76, σ=0.025, operatör deneyimine göre ±1.4x) + ölçüm gürültüsü (σ=0.008) | ✅ | ✅ |
| `Baski_Kuvveti_PSI` | Ölçülen perçin baskı/sıkıştırma kuvveti | Uniform(2000, 4000) + ölçüm gürültüsü (σ=40) | ✅ | ✅ |
| `Havsa_Derinligi_mm` | Ölçülen havşa (countersink) derinliği | Normal(μ=1.20, σ=0.015, operatör deneyimine göre ±1.4x) + ölçüm gürültüsü (σ=0.006) | ❌ | ✅ |
| `Operator_Tecrube_Yil` | Operatör tecrübe yılı | Tam sayı, 1–15 | ✅ | ✅ |
| `Rework_Gerekli` | Hedef değişken (0=Sağlam, 1=Rework) | Kural tabanlı (gerçek değerlere göre) + ~%1.5 insan kontrol hatası | ✅ | ✅ |

**Rework kuralı (gerçek/fiziksel değerlere uygulanır):**
- **MS20470:** Delik çapı `< 4.74 mm` veya `> 4.84 mm` **VEYA** baskı kuvveti `> 3500 PSI` ise parça rework gerektirir.
- **NAS1097:** Yukarıdaki iki koşula ek olarak, havşa derinliği `> 1.25 mm` ise de parça rework gerektirir.

Ayrıntılar için [Gerçekçilik Modeli](#gerçekçilik-modeli-üç-gürültü-katmanı) bölümüne bakın.

## Kurulum

```bash
pip install -r requirements.txt
```

## Çalıştırma Adımları

```bash
# 1. Her iki perçin tipi için sentetik veri üret
python 01_veri_uretimi.py

# 2. Her iki perçin tipi için modeli eğit, threshold analizini yap, .pkl dosyalarını kaydet
python 02_model_egitimi.py

# 3. Dashboard'u başlat
streamlit run app.py
```

Dashboard tarayıcıda açıldığında:
- **Sekme 1 — Anlık Saha Kalite Kontrolü:** Önce perçin tipi (MS20470 / NAS1097) seçilir; forma yalnızca o tipe ait parametreler gelir (MS20470 için Havşa Derinliği alanı gizlenir). Operatör ölçüm değerlerini girer, sistem anında kalite kararı verir.
- **Sekme 2 — Model Başarı & Güvenilirlik Analizi:** Perçin tipi seçimine göre Accuracy/Recall/Precision/F1 metrikleri, canlı eşik kaydırıcısı ve Precision-Recall eğrisi (%99 Recall hedef noktası işaretli) görüntülenir.

## Kullanılan Teknolojiler

- **scikit-learn** — Random Forest sınıflandırıcı
- **pandas / numpy** — veri üretimi ve işleme
- **streamlit** — interaktif dashboard
- **matplotlib / seaborn** — Precision-Recall görselleştirmesi
- **joblib** — model ve metrik serileştirme
