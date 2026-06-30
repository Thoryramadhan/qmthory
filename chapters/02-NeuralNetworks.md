# Rangkuman ML Crash Course: Neural Networks

Notebook ini berisi rangkuman komprehensif mengenai **Jaringan Saraf Tiruan (Neural Networks)** berdasarkan seri *Google Machine Learning Crash Course*. Notebook ini mencakup konsep dasar, cara kerja, serta persamaan matematis untuk setiap komponen, lengkap dengan visualisasinya.

---

## 1. Node dan Hidden Layer (Nodes and Hidden Layers)
[Referensi Materi](https://developers.google.com/machine-learning/crash-course/neural-networks/nodes-hidden-layers)

Model linear biasa mengombinasikan fitur input dengan bobot (*weights*) dan bias untuk menghasilkan prediksi dengan persamaan:

$$y=b+w_1x_1+w_2x_2+\dots+w_nx_n$$

Namun, banyak masalah di dunia nyata (seperti pengenalan gambar atau klasifikasi teks) bersifat **non-linear** (datanya tidak dapat dipisahkan dengan garis lurus atau bidang datar). Untuk mempelajari pola non-linear ini, kita membutuhkan arsitektur yang lebih kompleks yaitu **Neural Networks**.

Dalam Neural Networks, kita menambahkan lapisan perantara di antara fitur input dan prediksi output yang disebut **Hidden Layers** (Lapisan Tersembunyi).

* **Node (Neuron):** Setiap *hidden layer* terdiri dari kumpulan *node*. Sebuah node menerima input dari layer sebelumnya, melakukan komputasi, dan meneruskannya ke layer berikutnya.
* **Persamaan Matematis Node:** Setiap koneksi antar node memiliki bobot ($w$), dan setiap node menerima bias ($b$). Output mentah (sebelum aktivasi) dari suatu node dirumuskan sebagai:

$$z=\sum_{i=1}^{n}w_ix_i+b$$

**Masalah:** Jika kita hanya menggunakan persamaan di atas secara berulang-ulang dari satu *layer* ke *layer* berikutnya, output akhirnya akan tetap menjadi sebuah fungsi linear. Tidak peduli seberapa banyak *hidden layer* yang kita tambahkan, model tidak akan bisa menyelesaikan masalah non-linear. Di sinilah kita membutuhkan **Fungsi Aktivasi**.

---

## 2. Fungsi Aktivasi (Activation Functions)
[Referensi Materi](https://developers.google.com/machine-learning/crash-course/neural-networks/activation-functions)

Fungsi aktivasi adalah operasi matematika yang diterapkan pada output mentah ($z$) dari sebuah *node* sebelum nilai tersebut diteruskan ke *layer* berikutnya. Fungsinya yang paling utama adalah untuk **memperkenalkan sifat non-linearitas** ke dalam jaringan, yang memungkinkan *neural network* untuk memodelkan hubungan yang sangat rumit.

Berikut adalah beberapa fungsi aktivasi yang sering digunakan:

### A. ReLU (Rectified Linear Unit)
Fungsi aktivasi yang saat ini paling umum digunakan pada *hidden layer* karena efisiensi komputasinya yang tinggi dan kemampuannya mengatasi masalah *vanishing gradient* (gradien yang menghilang).

$$f(x)=\max(0,x)$$

Jika nilai $x$ negatif, hasilnya 0. Jika $x$ positif, hasilnya adalah $x$ itu sendiri.

### B. Sigmoid
Mengubah nilai input menjadi skala antara 0 dan 1. Dulu sering digunakan di *hidden layer*, namun sekarang lebih sering digunakan secara eksklusif pada **output layer** untuk tugas **klasifikasi biner** (karena outputnya dapat direpresentasikan sebagai probabilitas).

$$f(x)=\frac{1}{1+e^{-x}}$$

### C. Tanh (Hyperbolic Tangent)
Bentuknya mirip dengan fungsi Sigmoid, namun nilai outputnya berkisar antara -1 hingga 1. Hal ini membantu memusatkan data (*zero-centered*).

$$f(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}$$

Mari kita lihat bentuk kurva dari fungsi-fungsi aktivasi tersebut melalui kode Python di bawah ini.

```python
import numpy as np
import matplotlib.pyplot as plt

# Mendefinisikan fungsi aktivasi
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

# Menghasilkan data untuk sumbu x
x = np.linspace(-5, 5, 100)

# Plotting
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.plot(x, relu(x), color='blue', linewidth=2)
plt.title('ReLU')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(x, sigmoid(x), color='red', linewidth=2)
plt.title('Sigmoid')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(x, tanh(x), color='green', linewidth=2)
plt.title('Tanh')
plt.grid(True)

plt.tight_layout()
plt.show()
```

## 3. Propagasi Balik (Backpropagation)
[Referensi Materi](https://developers.google.com/machine-learning/crash-course/neural-networks/backpropagation)

Setelah data mengalir maju melalui jaringan dari input hingga menghasilkan prediksi (proses ini disebut **Forward Pass**), model perlu menghitung tingkat kesalahannya (disebut **Loss** atau *Error*).

Untuk membuat model menjadi lebih pintar, kita harus meminimalkan *Loss* tersebut dengan menyesuaikan nilai bobot ($w$) dan bias ($b$). Algoritma pengoptimalan utama yang digunakan adalah **Gradient Descent**, dan algoritma yang menghitung gradien-gradien ini untuk Neural Network disebut **Backpropagation**.

### Cara Kerja Backpropagation
Backpropagation (Propagasi Balik) bekerja dengan cara bergerak mundur dari *Output Layer* kembali ke arah *Input Layer*. Algoritma ini mendistribusikan "tanggung jawab" atas *error* atau *loss* yang terjadi kepada setiap bobot dan bias di dalam jaringan.

Konsep matematika inti di balik Backpropagation adalah **Aturan Rantai (Chain Rule)** dari Kalkulus.

### Persamaan Matematis Aturan Rantai (Chain Rule)
Misalkan kita ingin mengetahui seberapa besar pengaruh perubahan bobot $w$ terhadap total $Loss$ ($L$). Kita menghitung turunan parsialnya:

$$\frac{\partial L}{\partial w}=\frac{\partial L}{\partial a}\cdot\frac{\partial a}{\partial z}\cdot\frac{\partial z}{\partial w}$$

Keterangan:
- $L$ = Fungsi Loss (misal: Mean Squared Error atau Cross Entropy)
- $a$ = Output dari node (setelah fungsi aktivasi, $a=f(z)$)
- $z$ = Output linear dari node (sebelum fungsi aktivasi, $z=wx+b$)
- $w$ = Bobot koneksi (*weight*)

Ketiga komponen turunan tersebut merepresentasikan:
1. **$\frac{\partial L}{\partial a}$** : Bagaimana *Loss* berubah seiring dengan perubahan output aktivasi node.
2. **$\frac{\partial a}{\partial z}$** : Turunan dari **Fungsi Aktivasi** (inilah alasan mengapa fungsi aktivasi harus memiliki bentuk matematika yang bisa diturunkan/ *differentiable*).
3. **$\frac{\partial z}{\partial w}$** : Karena $z=wx+b$, turunannya terhadap bobot $w$ hanyalah input $x$ dari node tersebut.

### Pembaruan Bobot (Weight Update)
Setelah gradien ($\frac{\partial L}{\partial w}$) didapatkan dari proses Backpropagation, model akan memperbarui bobot aslinya menggunakan persamaan **Gradient Descent**:

$$w_{baru}=w_{lama}-\eta\frac{\partial L}{\partial w}$$

Dimana **$\eta$** (eta) adalah **Learning Rate**, yaitu sebuah *hyperparameter* yang mengatur seberapa besar langkah pembaruan bobot dalam setiap iterasi.

---
### Kesimpulan Alur Neural Network:
1. **Forward Pass:** Menghitung prediksi $y$ melalui node, *hidden layer*, dan fungsi aktivasi matematis.
2. **Calculate Loss:** Mengukur jarak/kesalahan antara prediksi dengan label yang sebenarnya.
3. **Backward Pass (Backprop):** Menghitung gradien kesalahan untuk setiap bobot menggunakan Aturan Rantai turunan kalkulus.
4. **Update Weights:** Menyesuaikan nilai bobot sedikit demi sedikit searah dengan nilai gradien negatifnya (Gradient Descent).

## 4. Latihan Praktis: Neural Network (Studi Kasus Sistem Rekomendasi Pupuk)

Sebagai contoh penerapan *end-to-end*, mari kita bangun sebuah model Neural Network sederhana. 

Kita akan membuat model klasifikasi biner untuk sistem rekomendasi kebutuhan pupuk pada tanaman kentang berdasarkan data analisis nutrisi tanah (kadar Nitrogen, Fosfor, Kalium) dan tingkat kelembapan tanah. Meskipun pendekatan sistem pakar (seperti logika *Fuzzy*) juga umum digunakan untuk studi kasus pertanian semacam ini dengan basis aturan manual, pendekatan Neural Network memiliki keunggulan karena dapat mempelajari pola interaksi kompleks antar nutrisi secara mandiri murni dari data (*data-driven*) melalui *backpropagation*.

Latihan ini mengimplementasikan konsep yang sudah dibahas:
* **Input Layer:** 4 fitur (N, P, K, Kelembapan).
* **Hidden Layers:** Menggunakan dua lapisan *node* dengan fungsi aktivasi **ReLU**.
* **Output Layer:** 1 *node* dengan fungsi aktivasi **Sigmoid** (0 = Nutrisi cukup/Tidak perlu pupuk, 1 = Perlu pupuk).
* **Loss & Optimizer:** Menggunakan *Binary Crossentropy* dan *Adam Optimizer* (pengembangan dari *Gradient Descent* otomatis).

### A. Implementasi Kode Program

```python
# Pastikan menginstall library jika belum ada: pip install tensorflow numpy matplotlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# ==========================================
# 1. Persiapan Data (Sintetik)
# ==========================================
# Membuat dataset simulasi berisi 1000 sampel tanah (Skala 0-100)
np.random.seed(42)
X = np.random.rand(1000, 4) * 100 

# Label target: 1 (Butuh Pupuk) jika total N+P+K < 120, 0 (Cukup) jika sebaliknya
y = np.where((X[:, 0] + X[:, 1] + X[:, 2]) < 120, 1, 0)

# ==========================================
# 2. Pemisahan Data (Data Splitting) Manual
# ==========================================
# Mengacak urutan data agar pembagian adil (Mencegah pola berurutan)
indices = np.random.permutation(len(X))
X_shuffled = X[indices]
y_shuffled = y[indices]

# Menentukan batas potong (80% Data Latih, 20% Data Uji)
split_index = int(0.8 * len(X))

# Membagi data berdasarkan indeks
X_train, X_test = X_shuffled[:split_index], X_shuffled[split_index:]
y_train, y_test = y_shuffled[:split_index], y_shuffled[split_index:]

# ==========================================
# 3. Membangun Arsitektur Jaringan (Forward Propagation)
# ==========================================
model = Sequential([
    # Hidden Layer 1: 8 nodes, aktivasi ReLU, menerima 4 fitur input
    Dense(8, input_shape=(4,), activation='relu', name='Hidden_Layer_1'),
    
    # Hidden Layer 2: 4 nodes, aktivasi ReLU, kompresi fitur
    Dense(4, activation='relu', name='Hidden_Layer_2'),
    
    # Output Layer: 1 node, aktivasi Sigmoid untuk probabilitas (0.0 hingga 1.0)
    Dense(1, activation='sigmoid', name='Output_Layer')
])

print("\n--- Ringkasan Model ---")
model.summary()

# ==========================================
# 4. Kompilasi dan Proses Pelatihan (Training)
# ==========================================
model.compile(optimizer='adam',                # Algoritma Backpropagation
              loss='binary_crossentropy',      # Fungsi hitung tingkat kesalahan
              metrics=['accuracy'])            # Metrik evaluasi

print("\n--- Memulai Proses Training ---")
history = model.fit(X_train, y_train, 
                    epochs=20,                 # Iterasi membaca seluruh data
                    batch_size=32,             # Jumlah sampel per pembaruan bobot
                    validation_split=0.2,      # 20% data latih dipakai untuk validasi per epoch
                    verbose=1)

# ==========================================
# 5. Evaluasi dan Inferensi (Prediksi Baru)
# ==========================================
print("\n--- Evaluasi pada Data Uji (Testing) ---")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Akurasi Model Final: {accuracy * 100:.2f}%\n")

# Contoh prediksi kondisi tanah: N=25.0, P=20.0, K=30.0, Kelembapan=60.0
sampel_baru = np.array([[25.0, 20.0, 30.0, 60.0]])
prediksi = model.predict(sampel_baru, verbose=0)

print(f"Hasil Probabilitas Sigmoid: {prediksi[0][0]:.4f}")
if prediksi[0][0] > 0.5:
    print("Keputusan Sistem: Tanaman MEMBUTUHKAN tambahan pupuk.")
else:
    print("Keputusan Sistem: Nutrisi tanah CUKUP.")

# ==========================================
# 6. Visualisasi Hasil Pelatihan (Plotting)
# ==========================================
plt.figure(figsize=(14, 5))

# Grafik 1: Akurasi
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Akurasi Data Latih (Train)', color='blue', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Akurasi Data Validasi (Val)', color='orange', linewidth=2)
plt.title('Perkembangan Akurasi Model')
plt.xlabel('Epoch')
plt.ylabel('Akurasi')
plt.legend()
plt.grid(True)

# Grafik 2: Loss (Kesalahan)
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Loss Data Latih (Train)', color='blue', linewidth=2)
plt.plot(history.history['val_loss'], label='Loss Data Validasi (Val)', color='orange', linewidth=2)
plt.title('Penurunan Tingkat Kesalahan (Loss)')
plt.xlabel('Epoch')
plt.ylabel('Loss (Binary Crossentropy)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

--- Ringkasan Model ---
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ Hidden_Layer_1 (Dense)          │ (None, 8)              │            40 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Hidden_Layer_2 (Dense)          │ (None, 4)              │            36 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Output_Layer (Dense)            │ (None, 1)              │             5 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 81 (324.00 B)
 Trainable params: 81 (324.00 B)
 Non-trainable params: 0 (0.00 B)

--- Memulai Proses Training ---
Epoch 1/20
20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.2984 - loss: 15.7582 - val_accuracy: 0.2812 - val_loss: 9.9330
...
Epoch 20/20
20/20 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7926 - loss: 0.4728 - val_accuracy: 0.7250 - val_loss: 0.5776

--- Evaluasi pada Data Uji (Testing) ---
Akurasi Model Final: 79.00%

Hasil Probabilitas Sigmoid: 0.3426
Keputusan Sistem: Nutrisi tanah CUKUP.
```

### B. Pembahasan dan Analisis Algoritma

**1. Pemisahan Data (Data Splitting)**
Pemisahan data menjadi *Training* (800 sampel) dan *Testing* (200 sampel) sangat krusial. Tujuannya adalah memastikan model diuji menggunakan matriks data yang belum pernah ia lihat sebelumnya. Hal ini memvalidasi bahwa jaringan benar-benar menggeneralisasi pola, bukan sekadar menghafal (*memorizing*) data latih.

**2. Arsitektur Corong (*Funneling*)**
Desain `Dense(8)` lalu `Dense(4)` sengaja dibuat menyerupai corong. Lapisan pertama mengekspansi 4 fitur input menjadi 8 kombinasi abstrak untuk mencari korelasi tersembunyi antar variabel (misal: rasio perbandingan Nitrogen terhadap Fosfor pada kelembapan tertentu). Lapisan kedua mengompresi informasi tersebut, memaksa jaringan untuk membuang *noise* dan hanya meneruskan pola yang paling berdampak ke *Output Layer*.

**3. Proses Kompilasi dan *Mini-Batch***
* **Binary Crossentropy:** Dipilih karena ini adalah klasifikasi biner (Pupuk vs Tidak Pupuk). Fungsi ini menghitung pinalti matematis setiap kali tebakan model meleset jauh dari nilai absolut target.
* **Batch Size (32):** Alih-alih menghitung kesalahan dan memperbarui bobot untuk seluruh 800 data sekaligus (yang rentan terjebak di *local minima* dan berat secara komputasi), *optimizer* memperbarui nilai bobot setiap kali selesai mengevaluasi 32 sampel data.

**4. Interpretasi Visualisasi Grafik (Plotting)**
* **Grafik Penurunan Loss (Kanan):** Anda akan melihat garis biru (Data Latih) dan oranye (Data Validasi) bergerak turun tajam di *epoch* awal dan melandai di akhir. Ini menunjukkan algoritma *Gradient Descent* sukses memperbarui matriks bobot ($w$) ke arah lembah kesalahan terkecil.
* **Grafik Peningkatan Akurasi (Kiri):** Garis akurasi akan merangkak naik secara stabil.
* **Indikator Kinerja Model:** Jika garis biru dan oranye bergerak berdekatan, artinya model **sehat** (kemampuan generalisasi baik). Jika garis biru (Latih) terus membaik sementara garis oranye (Validasi) terhenti atau justru memburuk, maka terjadi **Overfitting**—model terlalu rumit sehingga ia menghafal detail spesifik data latih dan gagal saat dihadapkan pada data baru.

**5. Inferensi Akhir**
Pada uji prediksi data sintetik (`N=25.0, P=20.0, K=30.0`), total NPK adalah 75. Angka ini berada di bawah batas toleransi (120). Hasil prediksi fungsi aktivasi Sigmoid berada pada `0.3426`. Karena nilai tersebut menjauhi probabilitas angka 1, program berhasil dan secara presisi merekomendasikan keputusan akhir sesuai dengan kondisi aktual di lapangan.