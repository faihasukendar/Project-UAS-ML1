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

Data ini diambil dari data penjualan Coffe vs Tea sales dari toko Backery.

### Variabel-variabel pada penjualan Coffe vs Tea Sales adalah sebagai berikut:

- Date = tanggal terjadi nya transaksi dengan format Tahun-Bulan-Tanggal  ( bertipe String )
- Time = waktu yang digunakan untuk pembelian ( bertipe Datetime )
- Transaction = Jumlah transaksi yang dilakukan oleh customer ( bertipe Int )
- Item = Produk/Barang dari toko Backery ( bertipe String )

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

```bash
df['Date']=pd.to_datetime(df['Date'], format = "%Y-%m-%d")
```

```bash
df['Date'].dtype
```

Menambahkan hari,bulan dan tahun

```bash
df['month']=df['Date'].dt.month
df['day']=df['Date'].dt.weekday
df['year']=df['Date'].dt.year
```

Hasil dari yang ditambahkan tersebut

```bash
df.head()
```

# EDA

- Menampilkan penjualan terbanyak dari tahun 2016 - 2017

```bash
n_credits = df.groupby("year")["Transaction"].count().rename("Count").reset_index()
n_credits.sort_values(by=["Count"], ascending=False, inplace=True)

plt.figure(figsize=(10,6))
bar = sns.barplot(x="year",y="Count",data=n_credits)
bar.set_xticklabels(bar.get_xticklabels(), rotation=60)
plt.ylabel("Perbandingan Jumlah Transaksi Tahun 2016 dan 2017")
plt.tight_layout()
```

![Grafik foto penjualan terbanyak](https://github.com/faihasukendar/Project-UAS-ML1/assets/149061885/bbdef2bc-69a2-475a-a1ed-51b4d93cd029)


- Menunjukan grafik dengan jumlah bulan terbanyak adalah November

```bash
n_credits = df.groupby("month")["Transaction"].count().rename("Count").reset_index()
n_credits.sort_values(by=["Count"], ascending=False, inplace=True)

plt.figure(figsize=(10,6))
bar = sns.barplot(x="month",y="Count",data=n_credits)
bar.set_xticklabels(bar.get_xticklabels(), rotation=60)
plt.ylabel("Jumlah Transaksi Perbulan")
plt.tight_layout()
```

![Grafik foto bulan dengan penjualan terbanyak](https://github.com/faihasukendar/Project-UAS-ML1/assets/149061885/e5798733-3564-4182-a769-bbb8da4a456e)


- Menunjukan  hari dengan penjulan terbanyak adalah hari Sabtu

```bash
data_perday = df.groupby('day')['Transaction'].count()

plt.figure(figsize= (11,5))
sns.barplot(
    x=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday', 'Sunday'],
    y=data_perday.values)
plt.xticks(size = 12, rotation = 30)
plt.title('Total Transaksi Perhari')
```

![Grafik foto hari dengan penjualan terbanyak](https://github.com/faihasukendar/Project-UAS-ML1/assets/149061885/20e8cc61-c9c8-4dad-bc6c-dc5dd9f08f5f)


- Menunjukan grafik dengan jam yang paling banyak dipakai

```bash
palette = sns.color_palette("pastel")
sns.countplot(x = 'Time', data = df, order = df['Time'].value_counts().iloc[:25].index,palette=palette)
plt.title("Data Penjualan Perjam")
plt.xticks(rotation=90)
```

![Grafik foto jam dengan penjualan terbanayak](https://github.com/faihasukendar/Project-UAS-ML1/assets/149061885/e1775733-7b0c-4e0a-b249-58f0444f2876)


- Menunjuakan grafik dengan menu yang paling banyak adalah Coffe

```bash
palette = sns.color_palette("pastel")
plt.xticks(rotation=45)
sns.countplot(x = 'Item', data = df, order = df['Item'].value_counts().iloc[:10].index, palette=palette)
```

![Grafik foto item dengan penjualan terbanyak](https://github.com/faihasukendar/Project-UAS-ML1/assets/149061885/02acf822-eb15-4d8f-98a3-bc78f623d518)


# Data Preparation

- Pada tahapan ini kita akan memastikan tipe data yang digunakan sesuai, tidak ada nilai kosong

- Menampilakn seluruh item dan transaksi 

```bash
df['Item'] = df['Item'].apply(lambda item: item.lower())
df['Item'] = df['Item'].apply(lambda item: item.strip())
```

```bash
df = df[['Transaction','Item']].copy()
df.head(10)
```

Import berdasarkan metode yang dipilih

```bash
from mlxtend.frequent_patterns import association_rules, apriori
```

Menambahkan hitungan dari setiap transaksi

```bash
item_count = df.groupby(["Transaction","Item"])["Item"].count().reset_index(name="Count")
item_count.head(10)
```


```bash
item_count_pivot = item_count.pivot_table(index='Transaction', columns= 'Item', values = 'Count', aggfunc = 'sum').fillna(0)
print("Ukuran Dataset", item_count_pivot.shape)
item_count_pivot.head(5)
```

```bash
item_count_pivot = item_count_pivot.astype('int32')
item_count_pivot.head()
```

lalu kita sederhanakan data tersebut menggunakan biner

```bash
def encode(x):
  if x <=0:
    return 0
  elif x >= 0:
    return 1

item_count_pivot = item_count_pivot.applymap(encode)
item_count_pivot.head()
```

Mencetak data yang sudah ditampilkan

```bash
print("Ukuran Dataset", item_count_pivot.shape)
print("Jumlah Transaksi", item_count_pivot.shape[0])
print("Jumlah Item", item_count_pivot.shape[1])
```

# Modelling

Menampilkan 10 itemset 

```bash
support = 0.02
frequent_items = apriori(item_count_pivot, min_support=support, use_colnames=True)
frequent_items.sort_values('support', ascending=False).head(10)
```

Menampilkan aturan asosiasi berdasarkan frequent itemset dengan menggunakan metrik lift

```bash
metric ="lift"
min_threshold = 0.5

rules = association_rules(frequent_items, metric=metric, min_threshold=min_threshold)
rules.sort_values('support', ascending=False,inplace=True)
rules.head(5)
```

# Evaluation

Berikut adalah beberapa item yang bisa dijual dalam satu paket pembelian dimana item-tem tersebut memiliki nilai confidence yang cukup tinggi dibandingkan lainnya

```bash
support = rules.support.to_numpy()
confidence = rules.confidence.to_numpy()
rec_rules = rules[ (rules['lift'] > 1) & (rules['confidence'] >= 0.5) ]

cols_keep = {'antecedents':'item_1', 'consequents':'item_2', 'support':'support', 'confidence':'confidence', 'lift':'lift'}
cols_drop = ['antecedent support', 'consequent support', 'leverage', 'conviction']

recommendation_basket = pd.DataFrame(rec_rules).rename(columns= cols_keep).drop(columns=cols_drop).sort_values(by=['confidence'], ascending = False)

print("Rekomendasi Paket Penjualan")
display(recommendation_basket)
```

![Screenshot (126)](https://github.com/faihasukendar/Project-UAS-ML1/assets/149061885/3647aab6-bb8d-489f-9633-cf1317d50307)


## Deployment
