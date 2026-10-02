# Proyek Analisis Data: Beijing Multi-Site Air Quality (PRSA) 🌫️

Analisis kualitas udara (fokus **PM2.5**) dari 12 stasiun pemantau di Beijing, 1 Maret 2013 – 28 Februari 2017, lengkap dengan notebook analisis dan dashboard interaktif Streamlit.

## Pertanyaan Bisnis
1. Stasiun mana yang memiliki rata-rata PM2.5 tertinggi dan terendah, dan berapa persen hari yang melampaui baku mutu harian 75 µg/m³?
2. Pada bulan dan jam berapa PM2.5 mencapai puncak dan titik terendah, dan berapa persen selisihnya?
3. Berapa persen perubahan rata-rata PM2.5 tahunan (periode Maret–Februari) dari 2013/14 ke 2016/17, dan berapa kali lipat level terbaru di atas baku mutu tahunan 35 µg/m³?
4. Seberapa kuat korelasi PM2.5 dengan faktor cuaca dan polutan lain, dan pada kecepatan angin berapa rata-rata PM2.5 turun di bawah 75 µg/m³?

Analisis lanjutan (tanpa *machine learning*): **manual grouping/binning** (kategori China AQI dan zona jarak dari pusat kota) serta **geospatial analysis** (peta stasiun statis dan interaktif).

## Struktur Direktori
```
submission
├── dashboard
│   ├── dashboard.py        # aplikasi Streamlit
│   └── main_data.csv       # data bersih per jam (hasil notebook)
├── data                    # 12 file CSV asli (satu per stasiun)
├── .streamlit
│   └── config.toml         # tema dashboard
├── notebook.ipynb          # proses analisis lengkap (sudah dijalankan)
├── README.md
├── requirements.txt
└── url.txt                 # tautan dashboard di Streamlit Community Cloud
```

## Setup Environment

### Anaconda
```
conda create --name main-ds python=3.11
conda activate main-ds
pip install -r requirements.txt
```

### Shell/Terminal
```
cd submission
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Menjalankan Dashboard
Jalankan dari folder `submission`:
```
streamlit run dashboard/dashboard.py
```
Dashboard terbuka di `http://localhost:8501`. Gunakan panel filter (rentang tanggal dan stasiun) di sebelah kiri; tiap tab menjawab satu pertanyaan bisnis, ditambah tab analisis lanjutan.

## Menjalankan Notebook
Buka `notebook.ipynb` dengan Jupyter, VS Code, atau Google Colab (unggah folder `data` bersama notebook), lalu pilih *Run All*. Jalankan dari folder `submission` agar jalur `data/` terbaca. Menjalankan notebook akan memperbarui `dashboard/main_data.csv`. Untuk Jupyter, pasang dulu: `pip install jupyter`.

## Deploy ke Streamlit Community Cloud
1. Unggah folder `submission` ke repositori GitHub publik.
2. Buka [share.streamlit.io](https://share.streamlit.io), pilih **Create app**, lalu isi repositori, *branch*, dan **Main file path**: `dashboard/dashboard.py`.
3. Klik **Deploy**, lalu salin URL aplikasi ke `url.txt`.

## Sumber Data
Beijing Multi-Site Air-Quality Data (PRSA), 12 stasiun pemantau nasional, pengukuran per jam. Baku mutu PM2.5 mengacu pada China GB 3095-2012 Kelas II (75 µg/m³ harian, 35 µg/m³ tahunan). Koordinat stasiun pada peta adalah perkiraan lokasi (tidak ada pada dataset) dan hanya dipakai untuk visualisasi.
