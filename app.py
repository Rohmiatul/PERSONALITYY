import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier

# --- Judul Aplikasi ---
st.title("Aplikasi Prediksi Kepribadian")
st.write("Masukkan data untuk memprediksi apakah seseorang adalah Ekstrovert atau Introvert.")

# --- Input Data Baru ---
st.header("Masukkan Data Baru")

# Menggunakan st.number_input untuk input numerik
new_Social_event_attendance = st.number_input(
    "Jumlah Kegiatan sosial:",
    min_value=0.0,
    max_value=100.0, # Sesuaikan dengan rentang data Anda
    value=0.0,
    step=1.0,
    help="Masukkan jumlah kegiatan sosial yang dihadiri."
)
new_Going_outside = st.number_input(
    "Jumlah Kegiatan bermain di luar:",
    min_value=0.0,
    max_value=100.0, # Sesuaikan dengan rentang data Anda
    value=0.0,
    step=1.0,
    help="Masukkan jumlah kegiatan yang dilakukan di luar ruangan."
)
new_Friends_circle_size = st.number_input(
    "Jumlah pertemanan:",
    min_value=0.0,
    max_value=100.0, # Sesuaikan dengan rentang data Anda
    value=0.0,
    step=1.0,
    help="Masukkan ukuran lingkaran pertemanan Anda."
)

# --- Tombol Prediksi ---
if st.button("Prediksi Kepribadian"):
    try:
        # --- Asumsi: Model dan Imputer sudah dilatih dan dimuat ---
        # Di sini Anda perlu memuat model dan imputer yang sudah dilatih.
        # Contoh:
        # import pickle
        # with open('imputer.pkl', 'rb') as f:
        #     imputer = pickle.load(f)
        # with open('knn_model.pkl', 'rb') as f:
        #     knn = pickle.load(f)

        # Jika Anda belum memiliki model yang disimpan, Anda bisa melatihnya di sini
        # Contoh sederhana (pastikan Anda memiliki data pelatihan yang sesuai):
        # Ini hanya contoh, Anda harus melatih model Anda dengan data yang relevan
        # dan menyimpannya, lalu memuatnya di sini.
        data_dummy = pd.DataFrame({
            'Social_event_attendance': [10, 2, 8, 3, 15, 1],
            'Going_outside': [5, 1, 7, 2, 10, 0],
            'Friends_circle_size': [20, 5, 15, 8, 30, 3],
            'Personality_code': [1, 0, 1, 0, 1, 0] # 1: Extrovert, 0: Introvert
        })
        X_dummy = data_dummy[['Social_event_attendance', 'Going_outside', 'Friends_circle_size']]
        y_dummy = data_dummy['Personality_code']

        imputer = SimpleImputer(strategy='mean')
        imputer.fit(X_dummy) # Fit imputer dengan data pelatihan Anda

        knn = KNeighborsClassifier(n_neighbors=3) # Sesuaikan n_neighbors Anda
        knn.fit(imputer.transform(X_dummy), y_dummy) # Latih KNN dengan data pelatihan Anda

        # Buat DataFrame dari input baru
        new_data_df = pd.DataFrame(
            [[new_Social_event_attendance, new_Going_outside, new_Friends_circle_size]],
            columns=['Social_event_attendance', 'Going_outside', 'Friends_circle_size']
        )

        # Impute data baru
        new_data_df_imputed = imputer.transform(new_data_df)
        new_data_df_imputed = pd.DataFrame(new_data_df_imputed, columns=new_data_df.columns)

        # Lakukan prediksi
        predicted_code = knn.predict(new_data_df_imputed)[0]

        # Konversi hasil prediksi ke label asli
        label_mapping = {1: 'Extrovert', 0: 'Introvert'}
        predicted_label = label_mapping.get(predicted_code, 'Tidak diketahui')

        st.subheader("Hasil Prediksi:")
        st.success(f"Prediksi Personality adalah: **{predicted_label}**")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
        st.info("Pastikan model dan imputer Anda telah dimuat dengan benar atau dilatih.")

st.markdown(
    """
    ---
    Aplikasi ini memprediksi kepribadian (Ekstrovert/Introvert) berdasarkan input Anda.
    """
)
