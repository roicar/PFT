def 2mm(n):
A = te.placeholder(((n, n),), name="A", dtype="float32")
B = te.placeholder(((n, n),), name="B", dtype="float32")
C = te.placeholder(((n, n),), name="C", dtype="float32")
D = te.placeholder(((n, n),), name="D", dtype="float32")
E = te.placeholder(((n, n),), name="E", dtype="float32")
k = te.reduce_axis((0, n), name="k")
C = te.compute((n, n), lambda i, j: te.sum(A[i][k] * B[k][j], axis=k), name="C")
K = te.reduce_axis((0, n), name="K")
E = te.compute((n, n), lambda M, J: te.sum(C[M][K] * D[K][J], axis=K), name="E")
return [A, B, C, D, E]
