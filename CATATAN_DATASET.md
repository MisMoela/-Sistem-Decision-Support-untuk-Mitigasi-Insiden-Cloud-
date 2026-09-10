# Catatan Penggunaan Dataset RE2-OB

Dokumen ini menjadi panduan mengenai file dataset yang digunakan oleh proyek
**Sistem Decision Support untuk Mitigasi Insiden Cloud Menggunakan Random Forest,
TreeSHAP, dan Agentic AI**.

Catatan ini membedakan data aktif, fungsi setiap file, hubungan antarsumber data,
cara pemuatan, serta data yang tidak dipakai oleh pipeline utama.

## Ringkasan cepat

Dataset aktif berada di `data/RE2-OB/RE2-OB/`. Dataset terdiri atas **30
scenario**, setiap scenario mempunyai **3 run**, sehingga totalnya adalah **90
run**. Setiap run berisi **7 file aktif**, atau **630 file aktif** untuk seluruh
dataset.

| File                 |                          Fungsi singkat                                  |
|----------------------|--------------------------------------------------------------------------|
| `simple_metrics.csv` | Fitur numerik metrik CPU, memory, disk, network, workload, error, dan latency                                                                                           |
| `logts.csv`          | Fitur numerik frekuensi cluster/template log                             |
| `tracets_lat.csv`    | Fitur time-series latency operasi/service                                |
| `tracets_err.csv`    | Fitur time-series error operasi/service                                  |
| `traces.csv`         | Trace/span mentah untuk agregasi dan hubungan antarservice               |
| `cluster_info.json`  | Kamus yang menerjemahkan ID cluster pada `logts.csv` menjadi template log dan service                                                                                       |
| `inject_time.txt`    | Ground truth waktu injeksi untuk menentukan baseline dan periode insiden |


Empat file time-series (`simple_metrics.csv`, `logts.csv`, `tracets_lat.csv`, dan
`tracets_err.csv`) menjadi sumber fitur numerik utama. `traces.csv` diproses per
chunk karena ukurannya besar. `cluster_info.json` digunakan sebagai metadata
penjelasan log, sedangkan `inject_time.txt` digunakan sebagai acuan waktu dan
bukan sebagai fitur model.

Urutan pemrosesan yang disarankan:

```text
Validasi struktur dan 7 file
        |
        v
Load satu run
        |
        v
Validasi isi dan timestamp
        |
        v
Ambil baseline serta incident window
        |
        v
Selaraskan time-series berdasarkan timestamp
        |
        v
Proses traces.csv secara bertahap
        |
        v
Ekstraksi fitur
        |
        v
Random Forest -> TreeSHAP -> rekomendasi Agentic AI
```

File yang tidak digunakan oleh pipeline utama ditempatkan di
`data/unused/RE2-OB/`. Data mentah aktif sebaiknya tidak diubah; hasil
preprocessing disimpan sebagai file baru di `data/processed/`.

-----------------------------------------------------------------------------------------------
Bagian-bagian berikut menjelaskan setiap poin pada ringkasan ini secara lebih
terperinci.

## 1. Lokasi dataset aktif

Root dataset yang harus digunakan oleh kode adalah:
- data/RE2-OB/RE2-OB/

## 2. Struktur dataset

Dataset terdiri atas 5 service dan 6 jenis fault:

### Service yang menjadi target fault

- `checkoutservice`
- `currencyservice`
- `emailservice`
- `productcatalogservice`
- `recommendationservice`

### Jenis fault

- `cpu`
- `delay`
- `disk`
- `loss`
- `mem`
- `socket`

Setiap kombinasi service dan fault membentuk satu scenario:

- 5 service x 6 fault = 30 scenario

Setiap scenario mempunyai tiga run (`1`, `2`, dan `3`):

- 30 scenario x 3 run = 90 run

Contoh struktur satu scenario:

data/RE2-OB/RE2-OB/
└── productcatalogservice_cpu/
    ├── 1/
    │   ├── simple_metrics.csv
    │   ├── logts.csv
    │   ├── tracets_lat.csv
    │   ├── tracets_err.csv
    │   ├── traces.csv
    │   ├── cluster_info.json
    │   └── inject_time.txt
    ├── 2/
    │   └── tujuh file yang sama
    └── 3/
        └── tujuh file yang sama
```

Hasil pemeriksaan struktur aktif:

| Komponen           |    Jumlah   |
|--------------------|------------:|
| Scenario           |     30      |
| Run per scenario   |     3       |
| Total run          |     90      |
| File aktif per run |     7       |
| Total file aktif   |    630      |

## 3. Tujuh file yang digunakan

Ketujuh file tersedia pada **setiap run**. File-file ini digunakan pada tahap yang
berbeda; tidak semuanya menjadi fitur numerik langsung untuk Random Forest.

### 3.1 `simple_metrics.csv`

**Peran:** sumber utama metrik numerik sistem.

Isi utamanya berupa time-series metrik berbagai service, antara lain:

- CPU;
- memory;
- disk I/O;
- socket/network;
- workload;
- error;
- latency persentil.

Karakteristik:

- memiliki kolom `time` berupa Unix timestamp dalam detik;
- sebagian besar fitur berbentuk numerik;
- interval pengamatan pada sampel dataset sekitar 1 detik;
- menjadi kandidat fitur utama Random Forest setelah dipotong, disejajarkan, dan
  diagregasi.

Cara pemuatan:

```python
metrics = pd.read_csv(run_path / "simple_metrics.csv")
metrics["time"] = pd.to_datetime(metrics["time"], unit="s", utc=True)
```

### 3.2 `logts.csv`

**Peran:** time-series frekuensi cluster/template log.

Contoh nama kolom:

```text
frontend_1
currencyservice_3
checkoutservice_21
```

Nama kolom memiliki pola:

```text
<service>_<cluster_id>
```

Nilainya merepresentasikan frekuensi kemunculan cluster log pada suatu timestamp.
Arti `cluster_id` harus dicari pada `cluster_info.json` milik run yang sama.

Karakteristik:

- memiliki kolom `time` berupa Unix timestamp dalam detik;
- pada sampel dataset intervalnya sekitar 15 detik;
- digunakan sebagai fitur numerik log;
- nama cluster tidak boleh diasumsikan mempunyai arti global untuk semua run.

### 3.3 `tracets_lat.csv`

**Peran:** ringkasan time-series latency operasi/service yang berasal dari trace.

Contoh kelompok fitur:

```text
currencyservice_Convert
productcatalogservice_GetProduct
checkoutservice_PlaceOrder
```

File ini digunakan untuk mendeteksi operasi yang mengalami kenaikan latency pada
periode insiden. File dapat menjadi input numerik Random Forest setelah timestamp
dan intervalnya diselaraskan.

### 3.4 `tracets_err.csv`

**Peran:** ringkasan time-series error pada operasi/service yang berasal dari trace.

File ini melengkapi `tracets_lat.csv` dengan informasi kegagalan. Kenaikan nilai
error dapat membantu model membedakan gangguan performa dengan gangguan yang
menyebabkan kegagalan request.

Karakteristik `tracets_lat.csv` dan `tracets_err.csv`:

- memiliki kolom `time` berupa Unix timestamp dalam detik;
- pada sampel dataset intervalnya sekitar 15 detik;
- kolom lainnya merupakan fitur operasi/service;
- harus diselaraskan berdasarkan timestamp, bukan nomor baris.

### 3.5 `traces.csv`

**Peran:** trace/span mentah untuk memperoleh hubungan request antarservice dan
konteks insiden yang lebih detail.

Kolom penting yang tersedia:

- `traceID`;
- `spanID`;
- `parentSpanID`;
- `serviceName`;
- `methodName`;
- `operationName`;
- `startTimeMillis`;
- `duration`;
- `statusCode`.

Penggunaan yang disarankan:

- menghitung jumlah span per service atau operasi;
- menghitung statistik duration/latency;
- menghitung jumlah status gagal;
- membentuk hubungan parent-child antarservice;
- menyediakan konteks service dependency untuk penjelasan insiden.

`traces.csv` berukuran jauh lebih besar daripada file ringkasan. Oleh karena itu,
file ini **tidak boleh dimuat untuk seluruh 90 run sekaligus**. Baca per run dan per
chunk:

```python
trace_chunks = pd.read_csv(
    run_path / "traces.csv",
    chunksize=100_000,
)
```

Untuk membentuk timestamp trace, gunakan `startTimeMillis`:

```python
chunk["timestamp"] = pd.to_datetime(
    chunk["startTimeMillis"],
    unit="ms",
    utc=True,
)
```

Kolom `time` pada `traces.csv` hanya berbentuk jam/menit sehingga tidak cukup
sebagai timestamp utama. Satuan `duration` harus dipastikan dari dokumentasi
dataset sebelum dikonversi.

### 3.6 `cluster_info.json`

**Peran:** kamus cluster/template log untuk menerjemahkan kolom `logts.csv`.

Meskipun namanya `cluster_info`, kata *cluster* pada file ini merujuk pada
pengelompokan pola log, bukan topologi Kubernetes, node, atau cluster cloud.

Contoh struktur:

```json
{
  "21": {
    "template": "checkoutservice [PlaceOrder] user id=...",
    "container": ["checkoutservice"]
  }
}
```

Jika `logts.csv` memiliki kolom `checkoutservice_21`, angka `21` dicari pada JSON
tersebut. Hasilnya memberikan:

- pesan/template log yang diwakili fitur;
- container/service yang menghasilkan log;
- konteks manusiawi untuk hasil SHAP;
- konteks semantik untuk rekomendasi Gemini.

Pemetaan ID harus dimuat **per run**. Pemeriksaan terhadap 90 file menunjukkan 89
isi unik, sehingga ID yang sama dapat berarti template berbeda pada run lain.
Jangan menggunakan `cluster_info.json` dari run pertama sebagai kamus global.

File ini biasanya tidak dimasukkan langsung ke Random Forest karena berisi teks
dan metadata. Fitur numeriknya berasal dari `logts.csv`; JSON digunakan untuk
menerjemahkan fitur tersebut ketika melakukan analisis dan penjelasan.

### 3.7 `inject_time.txt`

**Peran:** ground truth waktu ketika fault mulai diinjeksikan.

Contoh isi:

```text
1705340542
```

Nilai tersebut adalah Unix timestamp dalam detik. Konversi yang digunakan:

```python
inject_epoch = int((run_path / "inject_time.txt").read_text().strip())
inject_time = pd.to_datetime(inject_epoch, unit="s", utc=True)
```

File ini digunakan untuk:

- menentukan baseline sebelum fault;
- menentukan observation window sesudah fault;
- membentuk label fase normal/insiden jika diperlukan;
- mengevaluasi perubahan metrik, log, dan trace setelah injeksi.

`inject_time` adalah acuan pemotongan data, bukan fitur prediktor. Memasukkannya
langsung sebagai fitur Random Forest berisiko menyebabkan *data leakage*.

## 4. Hubungan antarsumber data

| Sumber | Jenis | Pemakaian utama |
|---|---|---|
| `simple_metrics.csv` | Numerik time-series | Fitur metrik Random Forest |
| `logts.csv` | Numerik time-series | Fitur frekuensi cluster log |
| `tracets_lat.csv` | Numerik time-series | Fitur latency operasi |
| `tracets_err.csv` | Numerik time-series | Fitur error operasi |
| `traces.csv` | Event/span mentah | Agregasi trace dan relasi service |
| `cluster_info.json` | Metadata teks | Arti cluster log dan konteks penjelasan |
| `inject_time.txt` | Ground truth waktu | Baseline, incident window, dan evaluasi |

Hubungan khusus yang perlu dipertahankan:

```text
logts.csv --cluster_id--> cluster_info.json

simple_metrics.csv ----┐
logts.csv -------------┼--timestamp--> observation window
tracets_lat.csv -------┤                    │
tracets_err.csv -------┘                    └--acuan: inject_time.txt

traces.csv --startTimeMillis--> agregasi trace pada window yang sama
```

## 5. Label scenario dan run

Nama folder scenario menyimpan dua label:

```text
productcatalogservice_cpu
└────── service ──────┘ └fault
```

Cara memisahkannya:

```python
service_name, fault_type = scenario_name.rsplit("_", 1)
```

Contoh hasil:

```python
service_name = "productcatalogservice"
fault_type = "cpu"
```

Nomor run harus ikut disimpan sebagai metadata. Jika nantinya satu run menghasilkan
banyak baris/window, gunakan gabungan scenario dan run sebagai `group_id`, misalnya:

```text
productcatalogservice_cpu__run_1
```

Pemisahan train/test harus mempertimbangkan `group_id` agar potongan waktu dari run
yang sama tidak tersebar ke train dan test (*data leakage*).

## 6. Kebijakan pemuatan data

Data harus diproses per run:

```text
Validasi struktur dataset
        ↓
Untuk setiap run:
    validasi 7 file
        ↓
    load 4 time-series ringkasan
        ↓
    load cluster_info dan inject_time
        ↓
    simpan path traces.csv
        ↓
    validasi isi hasil loading
        ↓
    ambil observation window
        ↓
    selaraskan timestamp
        ↓
    proses traces.csv per chunk
        ↓
    ekstrak fitur
```

Empat time-series ringkasan dapat dimuat langsung per run:

```python
TIMESERIES_FILES = {
    "metrics": "simple_metrics.csv",
    "logs": "logts.csv",
    "trace_latency": "tracets_lat.csv",
    "trace_error": "tracets_err.csv",
}
```

`cluster_info.json` dan `inject_time.txt` juga dimuat per run. `traces.csv` disimpan
sebagai path dan baru dibaca per chunk ketika fitur trace diekstrak.

## 7. Penyelarasan waktu

Jangan menggabungkan file berdasarkan nomor baris karena interval dan titik awal
timestamp dapat berbeda.

Langkah yang disarankan:

1. Konversi timestamp ke `DatetimeIndex` dengan UTC.
2. Tentukan baseline dan incident window menggunakan `inject_time`.
3. Ubah metrik 1 detik menjadi interval bersama, misalnya 15 detik.
4. Berikan prefix pada nama fitur agar tidak bertabrakan.
5. Gabungkan menggunakan timestamp.

Contoh prefix:

```text
metric__checkoutservice_cpu
log__checkoutservice_21
latency__checkoutservice_PlaceOrder
error__checkoutservice_PlaceOrder
```

Durasi baseline, incident window, dan interval resampling harus menjadi konfigurasi
eksperimen yang terdokumentasi, bukan angka yang tersebar di banyak fungsi.

## 8. Validasi yang wajib dilakukan

### Sebelum loading penuh

- `DATASET_ROOT` tersedia;
- terdapat 30 scenario dan 90 run;
- nama scenario mengikuti `<service>_<fault>`;
- setiap run mempunyai tujuh file wajib;
- file tidak kosong;
- empat CSV time-series memiliki kolom `time`;
- `traces.csv` memiliki kolom trace yang dibutuhkan;
- JSON dapat dibaca;
- `inject_time.txt` berisi Unix timestamp yang valid.

### Setelah loading

- DataFrame tidak kosong;
- timestamp berhasil dikonversi;
- index timestamp berurutan dan tidak duplikat;
- kolom fitur berbentuk numerik;
- `inject_time` berada dalam rentang data;
- baseline dan incident window tersedia;
- missing value dan infinity diperiksa;
- setiap kolom cluster pada `logts.csv` mempunyai pasangan ID di
  `cluster_info.json` milik run yang sama.

### Setelah preprocessing

- hasil tidak kosong;
- fitur mempunyai interval waktu yang konsisten;
- tidak ada nama kolom duplikat;
- fitur yang masuk model numerik;
- tidak ada target atau ground truth yang tidak sengaja masuk ke `X`;
- label service, fault, scenario, run, dan `group_id` tersimpan terpisah.

## 9. File yang tidak digunakan pipeline utama

File yang dipisahkan berada di:

```text
data/unused/RE2-OB/
```

Lima file berikut dipindahkan dari setiap run:

| File | Alasan tidak digunakan |
|---|---|
| `metrics.csv` | Metrik mentah; pipeline menggunakan `simple_metrics.csv` |
| `logs.csv` | Log mentah; pipeline menggunakan time-series `logts.csv` |
| `pod-node-1.csv` | Metrik pod/node tambahan di luar fitur utama |
| `pod-node-2.csv` | Metrik pod/node tambahan di luar fitur utama |
| `metrics_postprocess.log` | Catatan proses pembentukan dataset, bukan input model |

Folder/arsip duplikat `multi-source-data`, `multi-source-data.zip`, dan metadata
`.DS_Store` juga ditempatkan di `data/unused`.

File tidak dihapus. Jika metodologi berubah, file dapat dikembalikan ke path relatif
asalnya. Lihat juga `data/unused/README.md`.

## 10. Output preprocessing

Hasil preprocessing sebaiknya disimpan terpisah dari data mentah:

```text
data/processed/
```

Contoh artefak yang dapat dihasilkan:

```text
data/processed/features.csv
data/processed/feature_metadata.json
data/processed/validation_report.json
```

Data mentah dalam `data/RE2-OB/RE2-OB` sebaiknya diperlakukan sebagai read-only.
Pembersihan nilai, agregasi, dan transformasi disimpan sebagai output baru agar
eksperimen dapat direproduksi.

## 11. Pembagian tanggung jawab file kode

| File kode | Tanggung jawab |
|---|---|
| `src/preprocessing.py` | Validasi, loading, penyelarasan, dan ekstraksi fitur |
| `src/classifier.py` | Pelatihan dan prediksi Random Forest |
| `src/evaluator.py` | Evaluasi model dan metrik eksperimen |
| `src/explainer.py` | Penjelasan prediksi menggunakan TreeSHAP |
| `src/agent_loop.py` | Penyusunan konteks insiden dan rekomendasi Gemini |
| `main.py` | Mengatur urutan keseluruhan pipeline |

Random Forest seharusnya menerima dataset hasil preprocessing, bukan membaca tujuh
file mentah secara langsung.

## 12. Ringkasan keputusan

1. Gunakan tujuh file aktif pada setiap run.
2. Proses data satu run pada satu waktu.
3. Muat empat time-series ringkasan secara langsung.
4. Gunakan `inject_time.txt` sebagai acuan waktu, bukan fitur model.
5. Gunakan `cluster_info.json` sebagai kamus semantik `logts.csv` per run.
6. Baca `traces.csv` secara bertahap menggunakan chunk.
7. Gabungkan time-series berdasarkan timestamp, bukan nomor baris.
8. Simpan output di `data/processed` tanpa mengubah data mentah.
9. Pisahkan label dan metadata dari matriks fitur model.
10. Abaikan `data/unused` dan `data/RE2-OB.zip` ketika menjalankan pipeline.
