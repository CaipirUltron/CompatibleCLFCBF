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

determinants = []
Alist = []
solution = []
for i in range(dim):
    Anew = A.copy()
    Anew[:,i] = b
    detAnew = np.linalg.det(Anew)
    Alist.append( Anew )
    determinants.append( detAnew )
    solution.append( detAnew/detA )

print("Os determinantes das matrizes de Cramer são: " + str(determinants) + "\n")
print("A solução do sistema é: " + str(solution) + "\n")