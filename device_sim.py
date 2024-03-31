import numpy as np
from glob import glob

Name_TMP_dir = "TMP_dir\\"
Name_TMP_dir_out = "TMP_dir_out\\"
n_in_thr = 16
Dict = {}

def Info_reader(Name_TMP_dir):
    Filenames = glob(Name_TMP_dir + "*.tmp")
    for File in Filenames:
        New_file = File.strip()
        i_C, k, j_C = [New_file[i] for i in [20, 22, 24]] # Connected with .tmp filename format 
        Dict[i_C + "_" + k + "_" + j_C] = File
    return Filenames


Filenames = Info_reader(Name_TMP_dir)

N_A_row, K_max, N_B_cols = [int(i)+1 for i in max(Dict.keys()).split("_")]
N_of_departures = N_A_row*K_max*N_B_cols

def Matrix_reader_numpy(INPUT):
    Start_flag = "start"
    Finish_flag = "end"
    start_index = False
    Matrix_names = []
    AB_Matrix = []
    TMP_Matrix = []
    for String in INPUT:
        if Start_flag in String:
            Matrix_names += String.strip().split()[0]
            start_index = True
        elif start_index :
            if Finish_flag in String :
                start_index = False
                AB_Matrix += [TMP_Matrix]
                TMP_Matrix = []
            else :
                TMP_string = String.strip().split()
                TMP_Matrix += [[float(value) for value in TMP_string]]
        else :
            continue
    return np.array(AB_Matrix[0], dtype = float), np.array(AB_Matrix[1], dtype = float)

def Beauty_writer_2(OUTPUT, C_sliced):
    OUTPUT.write("start\n")
    for i_C in range(np.size(C_sliced, 0)):
        for j_C in range(np.size(C_sliced, 1)):
            OUTPUT.write(str(C_sliced[i_C][j_C]) + " ")
        OUTPUT.write("\n")
    OUTPUT.write("end\n")


for i_A in range(N_A_row):#N_A_row
    for j_B in range(N_B_cols):#N_B_cols
        C_sliced = np.zeros((n_in_thr, n_in_thr))
        for k in range(K_max):#K_max
            with open(Dict[str(i_A) + "_" + str(k) + "_" + str(j_B)], "r") as INPUT:
                A_tmp, B_tmp = Matrix_reader_numpy(INPUT)
            C_sliced_tmp = np.matmul(A_tmp, B_tmp) # numpy перемножение матриц
        C_sliced = C_sliced + C_sliced_tmp
        with open(Name_TMP_dir_out + "device2host_" + str(i_A) + "_" + str(j_B) + ".tmp", "w") as OUTPUT:
            Beauty_writer_2(OUTPUT, C_sliced)

        
            
