def mvt(n):
A = te.placeholder(((n + 1, n + 1),), name="A", dtype="float32")
x1 = te.placeholder((n + 1,), name="x1", dtype="float32")
x2 = te.placeholder((n + 1,), name="x2", dtype="float32")
y_1 = te.placeholder((n + 1,), name="y_1", dtype="float32")
y_2 = te.placeholder((n + 1,), name="y_2", dtype="float32")
j = te.reduce_axis((0, n + 1), name="j")
x1 = te.compute((n + 1), lambda i: te.sum(x1[i] + A[i][j] * y_1[j], axis=j), name="x1")
return [A, x1, x2, y_1, y_2]
