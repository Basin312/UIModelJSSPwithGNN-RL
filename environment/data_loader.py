import numpy as np
from job_shop_lib import JobShopInstance, Operation

class CSVInstance:
    def __init__(self, filepath_or_string):
        self.duration_matrix = []
        self.machines_matrix = []
        
        # 1. Handle Input File / String
        if hasattr(filepath_or_string, 'read'):
            content = filepath_or_string.read()
            file_string = content.decode("utf-8") if isinstance(content, bytes) else content
            lines = file_string.strip().split('\n')
        elif '\n' in str(filepath_or_string):
            lines = filepath_or_string.strip().split('\n')
        else:
            with open(filepath_or_string, 'r') as f:
                lines = f.readlines()
                
        self.clean_text_data = "\n".join([line.strip() for line in lines if line.strip()])
                
        # 2. Proses Parsing Data Mentah
        for line in lines:
            if not line.strip():
                continue
            parts = list(map(int, line.split()))
            
            # Memisahkan pasangan [Mesin, Durasi]
            machines = parts[0::2]   
            durations = parts[1::2]  
            
            self.machines_matrix.append(machines)
            self.duration_matrix.append(durations)
            
        # 3. DETEKSI DIMENSI OTOMATIS SECARA MATEMATIS
        self.duration_matrix = np.array(self.duration_matrix)
        self.machines_matrix = np.array(self.machines_matrix)
        
        # Ukuran baris = Jumlah Job, Ukuran kolom = Jumlah Mesin
        self.num_jobs = int(self.duration_matrix.shape[0])
        self.num_machines = int(self.duration_matrix.shape[1])
        self.num_operations = self.num_jobs * self.num_machines
        self.max_duration = float(np.max(self.duration_matrix))

    def to_job_shop_lib(self):
        jobs_list = []
        for i in range(self.num_jobs):
            current_job_operations = []
            for j in range(self.num_machines):
                machine_id = int(self.machines_matrix[i, j])
                duration = int(self.duration_matrix[i, j])
                current_job_operations.append(Operation(machine_id, duration))
            jobs_list.append(current_job_operations)
            
        return JobShopInstance(jobs=jobs_list, name="Uploaded_Instance")