def trmm(n):
A = te.placeholder(((n + 1, n + 1),), name="A", dtype="float32")
x = te.placeholder((n + 1,), name="x", dtype="float32")
y = te.placeholder((n + 1,), name="y", dtype="float32")
k = te.reduce_axis((0, n + 1), name="k")
B = te.compute((n + 1, n + 1), lambda i, j: te.sum(alpha*A[k][i] * B[k][j], axis=k), name="B")
return [A, x, y]
