import numpy as np

def Matrix_slicer(Base_Matrix, n_in_thr):
    N_row, N_col = len(Base_Matrix), len(Base_Matrix[0])
    n_of_row_matrix = int(N_row // n_in_thr) if N_row % n_in_thr == 0 else int(N_row // n_in_thr) + 1
    n_of_col_matrix = int(N_col // n_in_thr) if N_col % n_in_thr == 0 else int(N_col // n_in_thr) + 1
    Sliced_Matrix = []
    for i_row in range(n_of_row_matrix):
        for i_col in range(n_of_col_matrix):
            Matrix_part = [[0 for i in range(n_in_thr)] for j in range(n_in_thr)]
            for i in range(n_in_thr):
                for j in range(n_in_thr):
                    i_for_base, j_for_base = i_row*n_in_thr + i, i_col*n_in_thr + j
                    if i_for_base < N_row : 
                        if j_for_base < N_col :
                            Matrix_part[i][j] = Base_Matrix[i_for_base][j_for_base]
                        else:
                            break
                    else:
                        break
            Sliced_Matrix += [Matrix_part]
    return Sliced_Matrix, n_of_row_matrix, n_of_col_matrix

n_in_thr = 16

def Beauty_writer(Matrix, Name_of_matrix, OUTPUT_file):
    OUTPUT_file.write(Name_of_matrix + " start\n")
    for i in range(n_in_thr):
        for j in range(n_in_thr):
            OUTPUT_file.write(str(Matrix[i][j])+" ")
        OUTPUT_file.write("\n")
    OUTPUT_file.write(Name_of_matrix + " end\n")

def Road_to_device(A_part, B_part, i_C, k, j_C):
    with open("TMP_dir\\host2device_" + str(i_C) + "_" + str(k) + "_" + str(j_C) + ".tmp", "w") as OUTPUT:
        Beauty_writer(A_part, "A", OUTPUT)
        Beauty_writer(B_part, "B", OUTPUT)
n = 32

A, B = np.random.rand(n, n), np.random.rand(n, n)
A_sliced, N_A_rows, N_A_cols = Matrix_slicer(A, n_in_thr)
B_sliced, N_B_rows, N_B_cols = Matrix_slicer(B, n_in_thr)

N_of_departures = N_A_rows * N_B_cols if N_A_cols == N_B_cols else 0

if N_of_departures > 0 :
    for i_A in range(N_A_rows):
        for j_B in range(N_B_cols):
            for k in range(N_A_cols): # or range of N_B_cols numbers
                Road_to_device(A_sliced[i_A * N_A_cols + k], B_sliced[k * N_B_cols + j_B], i_A, k, j_B)
else:
    raise("*Numbers of A columes and numbers of B rows must be equal!*")


