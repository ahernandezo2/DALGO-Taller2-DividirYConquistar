import numpy as np

def strassen_matrix_mult(A,B):
    n = A.shape[0]
    #print(n)
    if n == 1:
        return A*B #caso base en que las matrices ya son 1x1

    #en otro caso hay que dividir las dos matrices en 8 sub matrices de tamaño n//2 x n//2
    A11 = A[:n//2, :n//2] #primeras filas y primeras columnas
    A12 = A[:n//2, n//2:] # primeras filas y últimas columnas
    A21 = A[n//2:, :n//2] #últimas filas, primeras columnas
    A22 = A[n//2:, n//2:] #últimas y últimas
    #print(A11,A12,A21,A22)
    B11 = B[:n//2, :n//2] #primeras filas y primeras columnas
    B12 = B[:n//2, n//2:] # primeras filas y últimas columnas
    B21 = B[n//2:, :n//2] #últimas filas, primeras columnas
    B22 = B[n//2:, n//2:] #últimas y últimas

    # se hacen las operaciones que definió Strassen, pero se le debe hacer 
    # strassen a las operaciones para que en algún momento de la recurrencia sean triviales
    M1 = strassen_matrix_mult(A11 + A22, B11 + B22)
    M2 = strassen_matrix_mult(A21 + A22, B11)
    M3 = strassen_matrix_mult(A11, B12 - B22)
    M4 = strassen_matrix_mult(A22, B21  - B11)
    M5 = strassen_matrix_mult(A11 + A12, B22)
    M6 = strassen_matrix_mult(A21 - A11, B11 + B12)
    M7 = strassen_matrix_mult(A12 - A22, B21 + B22)

    #ahora se crean los 4 componentes la matriz solución de una forma definida por Strassen
    S11 = M1 + M4 - M5 + M7
    S12 = M3 + M5
    S21 = M2 + M4
    S22 = M1 - M2 + M3 + M6

    #ahora que se tienen las 4 componentes de la matriz solución, hay que unirlos
    S = np.vstack((np.hstack((S11, S12)), np.hstack((S21, S22))))
    return S

A = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12],
              [13, 14, 15, 16]])

B = np.array([[16, 15, 14, 13],
              [12, 11, 10, 9],
              [8, 7, 6, 5],
              [4, 3, 2, 1]])

C = strassen_matrix_mult(A, B)

print(C)