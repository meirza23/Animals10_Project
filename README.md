# Animals-10 Image Classification Project

Bu proje, **Animals-10** veri setini kullanarak 10 farklı hayvan türünü sınıflandırmak için geliştirilmiş bir derin öğrenme (Deep Learning) projesidir. Model olarak **ResNet18** mimarisi ve **Transfer Learning** yöntemi kullanılmıştır.

## 🚀 Özellikler

- **Model:** Pre-trained ResNet18 (ImageNet ağırlıkları ile).
- **Hızlandırma:** Apple Silicon (M1/M2/M3) cihazlar için **MPS (Metal Performance Shaders)** desteği. NVIDIA GPU'lar için CUDA desteği.
- **Veri Artırma (Data Augmentation):** Eğitim başarısını artırmak için rastgele döndürme ve yatay çevirme işlemleri.
- **Görselleştirme:** Eğitim sonunda Confusion Matrix (Karmaşıklık Matrisi) oluşturulur.

## 📂 Proje Yapısı

```
Animals10_Project/
├── data/
│   └── raw-img/       # Veri seti (Otomatik indirilmelidir veya buraya konulmalıdır)
├── src/
│   ├── model.py       # ResNet18 model tanımı
│   └── __init__.py
├── main.py            # Eğitim ve test döngüsü
├── requirements.txt   # Gerekli kütüphaneler
└── README.md          # Proje dokümantasyonu
```

## 🛠️ Kurulum

1.  **Sanal Ortamı Oluşturun ve Aktif Edin:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Mac/Linux
    # venv\Scripts\activate   # Windows
    ```

2.  **Gerekli Kütüphaneleri Yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```

## ▶️ Kullanım

Modeli eğitmek ve test etmek için ana dosyayı çalıştırın:

```bash
python main.py
```

### Eğitim Süreci

- Script çalıştığında önce verileri hazırlar.
- 10 Epoch boyunca eğitimi gerçekleştirir.
- Her epoch sonunda Loss (Kayıp) ve Accuracy (Doğruluk) değerlerini ekrana basar.
- Eğitim bittiğinde modeli `model.pth` olarak kaydeder.

### Test ve Değerlendirme

- Eğitim tamamlandıktan sonra test verisi üzerinde değerlendirme yapılır.
- **Classification Report** (Precision, Recall, F1-Score) ekrana yazdırılır.
- **Confusion Matrix** oluşturulur ve `confusion_matrix.png` olarak kaydedilir.

## 📊 Beklenen Sonuçlar

ResNet18 ve Transfer Learning kullanımı sayesinde, sadece 10 epoch sonunda **%90 ve üzeri** bir doğruluk oranı (accuracy) hedeflenmektedir.

## 📝 Notlar

- Veri seti `data/raw-img` klasöründe olmalıdır.
- Mac kullanıcıları için MPS (GPU) otomatik olarak devreye girer.
