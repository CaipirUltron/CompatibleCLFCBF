import numpy as np

dim = 4
A = np.array([[1,3, 2,1],
              [0,2, 1,3],
              [2,1,-1,2],
              [3,0,-1,3]])

b = np.array([0,
              4,
              5,
              9])

detA = np.linalg.det(A)
print("\nO determinante de A é " + str(detA) + ".\n")
if detA == 0:
    raise Exception("O sistema é indeterminado ou impossível.\n")

A1 = A.copy()
A1[:,0] = b
detA1 = np.linalg.det(A1)
x1 = detA1/detA

A2 = A.copy()
A2[:,1] = b
detA2 = np.linalg.det(A2)
x2 = detA2/detA

A3 = A.copy()
A3[:,2] = b
detA3 = np.linalg.det(A3)
x3 = detA3/detA

A4 = A.copy()
A4[:,3] = b
detA4 = np.linalg.det(A4)
x4 = detA4/detA

determinants = [ detA1, detA2, detA3, detA4 ]
solution = [ x1, x2, x3, x4 ]

# determinants = []
# Alist = []
# solution = []
# for i in range(dim):
#     Anew = A.copy()
#     Anew[:,i] = b
#     detAnew = np.linalg.det(Anew)
#     Alist.append( Anew )
#     determinants.append( detAnew )
#     solution.append( detAnew/detA )

print("Os determinantes das matrizes de Cramer são: " + str(determinants) + "\n")
print("A solução do sistema é: " + str(solution) + "\n")