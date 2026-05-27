import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
# import model_ppo_mu, heuristic_solver_mu disini

# ==================================================
# CONFIG & NAVBAR (LAYOUT ATAS)
# ==================================================
st.set_page_config(layout="wide", page_title="JSSP AI Deployment Engine")

st.markdown("""
    <div style='background-color:#0e1117; padding:15px; border-radius:10px; margin-bottom:25px;'>
        <h2 style='color:white; margin:0;'>📊 JSSP Optimization Dashboard: Deep RL vs Classical Heuristics</h2>
        <p style='color:#a3a8b4; margin:0;'>Deployment System berbasis Proximal Policy Optimization (PPO) & Graph Neural Network (GNN)</p>
    </div>
""", unsafe_allow_html=True)

# ==================================================
# PEMBAGIAN KOLOM (KIRI & KANAN)
# ==================================================
col_kiri, col_kanan = st.columns([1, 2]) # Kolom kanan lebih lebar untuk grafik

# --------------------------------------------------
# KOLOM KIRI: INPUT AREA
# --------------------------------------------------
with col_kiri:
    st.subheader("🛠️ Input Konfigurasi JSSP")
    
    # Pilihan Metode Input
    tab_upload, tab_manual = st.tabs(["📁 Upload CSV/Text", "✍️ Input Manual"])
    
    instance_data = None
    
    with tab_upload:
        uploaded_file = st.file_uploader("Pilih file spesifikasi JSSP", type=["csv", "txt"])
        if uploaded_file is not None:
            # Fungsi parser CSV kamu ditaruh di sini
            st.success("File berhasil diunggah!")
            
    with tab_manual:
        num_jobs = st.number_input("Jumlah Job", min_value=2, max_value=10, value=3)
        num_machines = st.number_input("Jumlah Mesin", min_value=2, max_value=10, value=3)
        
        st.markdown("---")
        st.write("### 🎛️ Atur Rute Urutan Mesin (Isi dengan ID Mesin: 0, 1, 2...)")
        # Template default urutan mesin
        default_machines = pd.DataFrame(
            [[m for m in range(num_machines)] for j in range(num_jobs)],
            columns=[f"Step {i+1}" for i in range(num_machines)],
            index=[f"Job {j}" for j in range(num_jobs)]
        )
        df_machines_input = st.data_editor(default_machines, key="mach_edit")
 
        st.write("### ⏱️ Atur Durasi Waktu Pengerjaan (Menit/Detik)")
        # Template default durasi waktu
        default_durations = pd.DataFrame(
            [[5 for m in range(num_machines)] for j in range(num_jobs)],
            columns=[f"Step {i+1}" for i in range(num_machines)],
            index=[f"Job {j}" for j in range(num_jobs)]
        )
        df_durations_input = st.data_editor(default_durations, key="dur_edit")

    st.markdown("---")
    btn_optimize = st.button("🚀 Jalankan Optimasi Penjadwalan", type="primary")

# --------------------------------------------------
# KOLOM KANAN: TAMPILAN HASIL (GANTT & METRIK)
# --------------------------------------------------
with col_kanan:
    st.subheader("📈 Hasil Analisis & Komparasi Penjadwalan")
    
    if btn_optimize:
        # [BACKEND LOGIC] Sesi pembacaan numpy array dari input manual
        matrix_mesin = df_machines_input.to_numpy()
        matrix_durasi = df_durations_input.to_numpy()
        
        # 1. PROSES SIMULASI (Bypass ke model PPO dan Heuristik kamu)
        # Sesi ini nantinya akan menerima return dari model PPO/Heuristik aslimu
        score_ppo = 14  # Sesuai dengan hasil dummy Gantt Chart di bawah (0 sampai 14)
        score_spt = 18
        score_fifo = 22
        score_mwr = 19
        score_mor = 20
        
        # 2. TAMPILKAN METRIK KOMPARASI MAKESPAN
        st.write("### ⏱️ Perbandingan Hasil Makespan (Total Waktu)")
        
        # Menampilkan metrik utama dalam kolom berdampingan
        metrics_col = st.columns(3)
        metrics_col[0].metric(label="🤖 Model AI (PPO+GNN)", value=f"{score_ppo} mnt", delta="Paling Optimal", delta_color="inverse")
        metrics_col[1].metric(label="⏱️ Heuristik SPT", value=f"{score_spt} mnt", delta=f"+{score_spt-score_ppo} mnt")
        metrics_col[2].metric(label="⏳ Heuristik FIFO", value=f"{score_fifo} mnt", delta=f"+{score_fifo-score_ppo} mnt")
        
        # Menampilkan alternatif pembanding tambahan (MWR, MOR) dalam bentuk tabel ekspansi agar rapi
        with st.expander("🔍 Lihat Hasil Heuristik Lainnya"):
            st.write(f"- **Most Work Remaining (MWR):** {score_mwr} menit")
            st.write(f"- **Most Operations Remaining (MOR):** {score_mor} menit")
        
        # 3. MEMBUAT GANTT CHART (Sumbu Y = Mesin, Warna = Job)
        st.write("### 📅 Visualisasi Gantt Chart (Sumbu Y = Mesin)")

        # Data koordinat plotting hasil optimasi
        df_gantt_fixed = [
            # Job 0 pengerjaannya: M1 (durasi 2), M3 (durasi 4), M2 (durasi 5)
            dict(Task="Machine 1", Start='2026-05-26 00:00:00', Finish='2026-05-26 00:02:00', Resource='Job 0'),
            dict(Task="Machine 3", Start='2026-05-26 00:05:00', Finish='2026-05-26 00:09:00', Resource='Job 0'),
            dict(Task="Machine 2", Start='2026-05-26 00:09:00', Finish='2026-05-26 00:14:00', Resource='Job 0'),
            
            # Job 1 pengerjaannya: M3 (durasi 5), M1 (durasi 2), M2 (durasi 2)
            dict(Task="Machine 3", Start='2026-05-26 00:00:00', Finish='2026-05-26 00:05:00', Resource='Job 1'),
            dict(Task="Machine 1", Start='2026-05-26 00:05:00', Finish='2026-05-26 00:07:00', Resource='Job 1'),
            dict(Task="Machine 2", Start='2026-05-26 00:07:00', Finish='2026-05-26 00:09:00', Resource='Job 1'),
            
            # Job 2 pengerjaannya: M2 (durasi 8), M3 (durasi 1), M1 (durasi 4)
            dict(Task="Machine 2", Start='2026-05-26 00:00:00', Finish='2026-05-26 00:08:00', Resource='Job 2'),
            dict(Task="Machine 3", Start='2026-05-26 00:09:00', Finish='2026-05-26 00:10:00', Resource='Job 2'),
            dict(Task="Machine 1", Start='2026-05-26 00:10:00', Finish='2026-05-26 00:14:00', Resource='Job 2'),
        ]

        # Buat grafik Plotly Gantt
        fig = ff.create_gantt(
            df_gantt_fixed, 
            index_col='Resource', 
            show_colorbar=True, 
            group_tasks=True,      
            title="Optimal Job Shop Schedule (PPO Model Result)"
        )

        # Membalikkan sumbu Y agar terurut dari Mesin paling kecil (M1) di bagian atas
        fig.update_yaxes(autorange="reversed")

        st.plotly_chart(fig, use_container_width=True)
        st.caption("💡 Tip: Arahkan kursor ke ujung kanan atas grafik untuk memunculkan ikon kamera, lalu klik untuk mengunduh Gantt Chart (PNG).")
        
        # 4. REVISI: TOMBOL DOWNLOAD URUTAN JOB (Ubah Target ke df_gantt_fixed)
        st.write("### 📑 Ekstraksi Tabel Urutan Kerja")
        
        # Mengubah data list dict tadi langsung menjadi DataFrame terstruktur
        df_csv_result = pd.DataFrame(df_gantt_fixed)
        
        # Konversi waktu kembali ke bentuk indeks integer relatif agar mudah dibaca di CSV hasil akhir
        df_csv_result['Start_Minutes'] = pd.to_datetime(df_csv_result['Start']).dt.minute
        df_csv_result['End_Minutes'] = pd.to_datetime(df_csv_result['Finish']).dt.minute
        
        # Pilih kolom-kolom yang esensial saja untuk disuguhkan ke user pabrik
        df_clean_output = df_csv_result[['Resource', 'Task', 'Start_Minutes', 'End_Minutes']].rename(
            columns={'Resource': 'Job ID', 'Task': 'Machine ID', 'Start_Minutes': 'Start Time', 'End_Minutes': 'End Time'}
        ).sort_values(by=['Machine ID', 'Start Time'])
        
        # Tampilkan cuplikan tabel hasil di web dashboard
        st.dataframe(df_clean_output, use_container_width=True, hide_index=True)
        
        # Generate trigger download button
        csv_data = df_clean_output.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Tabel Urutan Jadwal Detail (CSV)",
            data=csv_data,
            file_name='hasil_jadwal_terbaik_ppo.csv',
            mime='text/csv',
        )
    else:
        st.info("Silakan masukkan data konfigurasi JSSP di kolom kiri dan klik tombol 'Jalankan Optimasi Penjadwalan' untuk melihat visualisasi.")