#include <iostream>
#include <math.h>

extern "C" { 
#include <lapacke.h>
}

using namespace std;

int main(int argc, char *argv[])
{
    lapack_int N = 4;
    // use full matrix
    double matrix_ex_full[16] = {-2,0,0.5,0,    0,0.5,-2,0.7,    0.5, -2, 0.5, 0,     0,0.7,0,-1};
    double evals_full[4];
    lapack_int test1 = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'N', 'U', N, matrix_ex_full,N, evals_full);
    cout << "success = " <<test1 << endl;
    for (int i = 0;i<4;i++)
        cout << evals_full[i] << endl;
    // use upper triagonal only
    double matrix_ex_uppertri[10] = {-2, 0, 0.5, 0,    0.5, -2, 0.7,    0.5, 0,   -1};
    double evals_uppertri[4];


    int test2 = LAPACKE_dspev(LAPACK_COL_MAJOR, 'N', 'L', N, matrix_ex_uppertri, evals_uppertri,NULL,N);
    cout << "success = " <<test2 << endl;
    for (int i = 0;i<4;i++)
        cout << evals_uppertri[i] << endl;
    return 0;
}
