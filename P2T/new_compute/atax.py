def atax(n):
A = te.placeholder(((n, n),), name="A", dtype="float32")
x = te.placeholder((n,), name="x", dtype="float32")
y = te.placeholder((n,), name="y", dtype="float32")
j = te.reduce_axis((0, n), name="j")
tmp = te.compute((n), lambda i: te.sum(A[i][j] * x[j], axis=j), name="tmp")
return [A, x, y]
