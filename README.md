 Laporan Proyek Machine Learning

### Nama : Faiha Atsaa Sukendar

### Nim : 211351052

### Kelas : IF Malam B

## Domain Proyek
- Proyek ini bermaksud untuk menjelajahi pola pembelian konsumen sebelum mereka memutuskan membeli kopi atau teh, dengan fokus pada strategi cross-selling atau upselling.

[Dataset yang saya gunakan] (https://www.kaggle.com/code/najibmh/coffee-vs-tea-sales)

## Business Understanding
- langkah awal dalam merancang strategi bisnis yang efektif. Ini melibatkan pemahaman mendalam tentang pasar, pelanggan, dan tren yang memengaruhi industri kopi dan teh yang melibatkan segmentasi pasar, Saluran Distribusi, Persaingan di Pasar dan lain-lain yang memengaruhi aspek pemasaran.

### Problem Statements
- Ketidakjelasan Pola Pembelian
- Kurangnya Personalisasi dalam Penawaran Produk
- Kesulitan dalam Menganalisis Perilaku Pembelian
- Rendahnya Kesadaran Terhadap Produk Tambahan
- Tidak Optimalnya Penggunaan Analisis Prediktif

### Goals
- Optimalisasi penawaran produk Meningkatkan pemahaman tentang pola pembelian konsumen untuk mengoptimalkan penawaran produk tambahan yang sesuai dengan preferensi masing-masing pelanggan.
- Implementasi strategi cross-selling yang efektif dengan mengidentifikasi dan menawarkan produk tambahan yang sesuai dengan preferensi pelanggan sebelum membeli kopi atau teh. 
-  Peningkatan penggunaan analisis prediktif untuk meramalkan kebutuhan dan preferensi konsumen, memberikan rekomendasi yang lebih akurat, dan meningkatkan efektivitas strategi penjualan.

## Data Understanding

Paragraf awal bagian ini menjelaskan informasi mengenai data digunakan dalam proyek. dataset wajib menggunakan kaggle dan atribut yang digunakan minimal 8 atribut.

### Variabel-variabel pada penjualan Coffe vs Tea Sales adalah sebagai berikut:

- Jumlah Penjualan

Jumlah Transaksi = 9531

- Nilai Penjualan

Coffe
Tea

- Segmentasi Produk

Penjualan Coffe
Penjualan Tea
dsb

## Data Preparation

Disini kita upload file kaggle.json kita

```bash
from google.colab import files
files.upload()
```

Setelah upload akan keliatan file kaggle.json

```bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle
```

Masukan dataset yang sudah dipilih

```bash
!kaggle datasets download -d sulmansarwar/transactions-from-a-bakery
```

Extract dataset yang sudah di download 

```bash
!kaggle datasets download -d sulmansarwar/transactions-from-a-bakery
```

# Import library 

Seteleh menentukan dataset yang akan dibuatkan model Machine Learningnya selanjutnya kita ketikan library python yang ingin di gunakan

```bash
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mlxtend.frequent_patterns import association_rules, apriori
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
```

# Data Exploration

- Disini kita makan menampilkan dataset yang tadi di download

```bash
df = pd.read_csv('/content/BreadBasket_DMS.csv')
```

```bash
df.head()
```
