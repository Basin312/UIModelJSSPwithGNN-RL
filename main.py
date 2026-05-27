import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import torch
from job_shop_lib.dispatching.rules import DispatchingRuleSolver

# ==================================================
# IMPORT MODUL BACKEND RL YANG SUDAH DIRAPIKAN
# ==================================================
from environment.data_loader import CSVInstance
from environment.schedule_env import JSSPEnv
from model.gnn_model import JSSPGNN
from agents.ppo_agent import PPO

# ==================================================
# UTILITY FUNCTIONS (BACKEND LOGIC)
# ==================================================

@st.cache_resource
def load_trained_agent():
    """Memuat model GNN dan Agen PPO ke dalam cache Streamlit agar tidak reload terus."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Inisialisasi Arsitektur Model
    model = JSSPGNN(num_node_features=8).to(device)
    
    # 2. Inisialisasi Agen PPO
    agent = PPO(model=model, device=device)
    
    # 3. Muat Bobot Model (Sesuaikan path file checkpoint .pt kamu)
    checkpoint_path = "model/best_model_gnn_jssp.pth"
    if os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
    else:
        st.sidebar.warning("PERINGATAN!: File 'model/best_model_gnn_jssp.pth' tidak ditemukan. Menggunakan bobot acak (Mode Demo).")
        
    return agent, device

def run_ppo_inference(ppo_agent, device, instance_obj):
    """Menjalankan simulasi penjadwalan JSSP menggunakan model PPO."""
    # Dummy generator karena JSSPEnv membutuhkan objek generator di init
    def single_instance_generator():
        while True:
            yield instance_obj
            
    env = JSSPEnv(generator=single_instance_generator())
    ppo_agent.policy.eval()
    
    state, _ = env.reset(instance=instance_obj)
    done = False
    
    # Catatan riwayat eksekusi untuk visualisasi Gantt Chart
    history_logs = []
    
    with torch.no_grad():
        while not done:
            state = state.to(device)
            legal_actions = env.get_legal_actions()
            
            # Pilih aksi terbaik secara deterministik (Argmax)
            action, _, _, _ = ppo_agent.select_action(state, legal_actions, deterministic=True)
            
            # Catat log sebelum step memajukan waktu
            node_id = action
            job_id, step_id = env.graph.node_to_job[node_id]
            machine_id = env.graph.machine_matrix[job_id, step_id]
            start_time = env.current_time
            duration = env.graph.duration_matrix[job_id, step_id]
            
            history_logs.append({
                "Job": f"Job {job_id}",
                "Machine": f"Machine {machine_id}",
                "Start": start_time,
                "Finish": start_time + duration
            })
            
            # Eksekusi aksi
            state, reward, done, _, _ = env.step(action)
            
    makespan = env.current_time
    return makespan, history_logs

def convert_to_plotly_gantt(history_logs):
    """Mengonversi log waktu integer menjadi format datetime yang dipahami Plotly."""
    base_date = "2026-01-01 00:00:00"
    df_gantt = []
    
    for log in history_logs:
        # Mengonversi satuan waktu integer (menit) ke format string ISO Datetime
        start_dt = pd.to_datetime(base_date) + pd.to_timedelta(log["Start"], unit="m")
        finish_dt = pd.to_datetime(base_date) + pd.to_timedelta(log["Finish"], unit="m")
        
        df_gantt.append(dict(
            Task=log["Machine"],
            Start=start_dt.strftime('%Y-%m-%d %H:%M:%S'),
            Finish=finish_dt.strftime('%Y-%m-%d %H:%M:%S'),
            Resource=log["Job"]
        ))
    return df_gantt

# ==================================================
# LAYOUT UTAMA STREAMLIT
# ==================================================
st.set_page_config(layout="wide", page_title="JSSP AI Deployment Engine")

st.markdown("""
    <div style='background-color:#0e1117; padding:15px; border-radius:10px; margin-bottom:25px;'>
        <h2 style='color:white; margin:0;'> JSSP Optimization Dashboard: Deep RL vs Classical Heuristics</h2>
        <p style='color:#a3a8b4; margin:0;'>Deployment System berbasis Proximal Policy Optimization (PPO) & Graph Neural Network (GNN)</p>
    </div>
""", unsafe_allow_html=True)

col_kiri, col_kanan = st.columns([1, 2])

# Load Agen PPO Backend
agent, current_device = load_trained_agent()

# --------------------------------------------------
# KOLOM KIRI: INPUT AREA
# --------------------------------------------------
with col_kiri:
    st.subheader("🛠️ Input Konfigurasi JSSP")
    st.write("Silakan unggah spesifikasi masalah JSSP dalam format teks benchmark.")
    
    final_instance = None
    
    # Langsung tampilkan File Uploader tanpa Tab-Tab-an
    uploaded_file = st.file_uploader("Pilih file spesifikasi JSSP (.txt / .csv)", type=["csv", "txt"])
    
    if uploaded_file is not None:
        try:
            # Panggil CSVInstance BARU (Tanpa num_jobs & num_machines)
            final_instance = CSVInstance(uploaded_file)
            
            # Tampilkan informasi dimensi yang berhasil dideteksi otomatis
            st.success(
                f" File berhasil dimuat!\n"
                f"* **Terdeteksi:** {final_instance.num_jobs} Jobs & {final_instance.num_machines} Machines\n"
                f"* **Total Operasi:** {final_instance.num_operations}"
            )
            
            # (Opsional) Intip matriks durasi untuk memastikan data benar
            with st.expander(" Intip Matriks Durasi Waktu"):
                st.dataframe(final_instance.duration_matrix)
                
        except Exception as e:
            st.error(f" Gagal membaca file. Pastikan format kolom sesuai. Error: {e}")

    st.markdown("---")
    
    # Tombol optimasi diaktifkan HANYA jika file sudah diunggah
    if final_instance is not None:
        btn_optimize = st.button("🚀 Jalankan Optimasi Penjadwalan", type="primary")
    else:
        st.info("💡 Unggah file di atas terlebih dahulu untuk mengaktifkan tombol optimasi.")
        btn_optimize = False

# --------------------------------------------------
# KOLOM KANAN: TAMPILAN HASIL (GANTT & METRIK)
# --------------------------------------------------
with col_kanan:
    st.subheader(" Hasil Analisis & Komparasi Penjadwalan")
    
    if btn_optimize and final_instance is not None:
        with st.spinner("Model GNN-PPO sedang mencari kombinasi urutan paling optimal..."):
            # 1. PREDIKSI UTAMA VIA BACKEND RL MODEL
            score_ppo, logs_riwayat = run_ppo_inference(agent, current_device, final_instance)
            
            instance_lib = final_instance.to_job_shop_lib()

            # Intasnce dispatching rules
            solver_spt = DispatchingRuleSolver(dispatching_rule="shortest_processing_time")
            solver_mwr = DispatchingRuleSolver(dispatching_rule="most_work_remaining")
            solver_fifo = DispatchingRuleSolver(dispatching_rule="first_come_first_served")
            solver_mor = DispatchingRuleSolver(dispatching_rule="most_operations_remaining", machine_chooser="random")
            
            # solution PDR
            solution_spt = solver_spt(instance_lib)
            solution_mwr = solver_mwr(instance_lib)
            solution_fifo = solver_fifo(instance_lib)
            solution_mor = solver_mor(instance_lib)

            # Nilai heuristik klasik (Ganti dengan fungsi matematika/solver aslimu jika ada)
            # Sementara menggunakan interpolasi logis berbasis nilai PPO untuk visualisasi dashboard
            score_spt = int(solution_spt.makespan())
            score_fifo = int(solution_fifo.makespan())
            score_mwr = int(solution_mwr.makespan())
            score_mor = int(solution_mor.makespan())
            
            daftar_nilai_pdr = [score_spt, score_fifo, score_mwr, score_mor]
            pdr_terbaik = min(daftar_nilai_pdr)
            
            # Hitung persentase penghematan/efisiensi waktu Model PPO vs PDR Terbaik
            if pdr_terbaik > 0:
                persen_efisiensi = ((pdr_terbaik - int(score_ppo)) / pdr_terbaik) * 100
            else:
                persen_efisiensi = 0
            
            # Tentukan teks delta dan warnanya berdasarkan hasil tanding PPO vs PDR
            if persen_efisiensi > 0:
                teks_delta = f"+{persen_efisiensi:.1f}% Vs PDR Terbaik"
                warna_delta = "normal"   # Warna hijau (PPO lebih cepat)
            elif persen_efisiensi < 0:
                teks_delta = f"{persen_efisiensi:.1f}% Vs PDR Terbaik"
                warna_delta = "inverse"  # Warna merah (PPO kalah cepat)
            else:
                teks_delta = "Setara PDR Terbaik"
                warna_delta = "off"

            # 2. TAMPILKAN METRIK KOMPARASI MAKESPAN
            st.write("###  Perbandingan Hasil Makespan (Total Waktu)")
            metrics_col = st.columns(3)
            metrics_col[0].metric(label=" Model AI (PPO+GNN)", value=f"{int(score_ppo)} waktu", delta= teks_delta, delta_color= warna_delta)
            metrics_col[1].metric(label=" Heuristik SPT", value=f"{score_spt} waktu", delta=f"+{score_spt-int(score_ppo)} waktu")
            metrics_col[2].metric(label=" Heuristik FIFO", value=f"{score_fifo} waktu", delta=f"+{score_fifo-int(score_ppo)} waktu")
            
            with st.expander(" Lihat Hasil Heuristik Lainnya"):
                st.write(f"- **Most Work Remaining (MWR):** {score_mwr} waktu")
                st.write(f"- **Most Operations Remaining (MOR):** {score_mor} waktu")
            
            # 3. VISUALISASI DATA GANTT CHART ASLI
            st.write("###  Visualisasi Gantt Chart")
            df_gantt_fixed = convert_to_plotly_gantt(logs_riwayat)
            
            # --- PERBAIKAN: GENERATE WARNA DINAMIS BERDASARKAN JUMLAH JOB ---
            # Kita buat palet warna RGB/Hex acak sebanyak jumlah Job unik yang ada
            unique_jobs = sorted(list(set([d['Resource'] for d in df_gantt_fixed])))
            num_unique_jobs = len(unique_jobs)
            
            # Menggunakan skema warna bawaan Plotly Express secara dinamis (mendukung banyak warna)
            import plotly.express as px
            if num_unique_jobs <= 10:
                color_palette = px.colors.qualitative.Plotly[:num_unique_jobs]
            elif num_unique_jobs <= 24:
                color_palette = px.colors.qualitative.Dark24[:num_unique_jobs]
            else:
                # Jika lebih dari 24 job, generate warna pelangi (HLS/HSV) secara matematis
                color_palette = [f"hsl({int(360 * i / num_unique_jobs)}, 70%, 50%)" for i in range(num_unique_jobs)]
            
            # Buat mapping dictionary antara nama Job dengan warnanya
            colors_mapping = dict(zip(unique_jobs, color_palette))
            
            # Masukkan parameter 'colors' ke dalam create_gantt
            fig = ff.create_gantt(
                df_gantt_fixed, 
                colors=colors_mapping,      # <--- Masukkan mapping warna di sini
                index_col='Resource', 
                show_colorbar=True, 
                group_tasks=True,      
                title="Optimal Job Shop Schedule (PPO Model Result)"
            )
            fig.update_yaxes(autorange="reversed")  # Urutkan mesin dari terkecil ke terbesar
            st.plotly_chart(fig, use_container_width=True)
            
            # 4. EKSTRAKSI DATA TABEL UNTUK USER PABRIK
            st.write("###  Ekstraksi Tabel Urutan Kerja")
            df_csv_result = pd.DataFrame(logs_riwayat)
            
            df_clean_output = df_csv_result[['Job', 'Machine', 'Start', 'Finish']].rename(
                columns={'Job': 'Job ID', 'Machine': 'Machine ID', 'Start': 'Start Time', 'Finish': 'End Time'}
            ).sort_values(by=['Machine ID', 'Start Time'])
            
            st.dataframe(df_clean_output, use_container_width=True, hide_index=True)
            
            csv_data = df_clean_output.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=" Download Tabel Urutan Jadwal Detail (CSV)",
                data=csv_data,
                file_name='hasil_jadwal_terbaik_ppo.csv',
                mime='text/csv',
            )
    else:
        st.info("Silakan masukkan data konfigurasi JSSP di kolom kiri dan klik tombol 'Jalankan Optimasi Penjadwalan' untuk melihat visualisasi.")