def bicg(n):
A = te.placeholder(((n + 1, n + 1),), name="A", dtype="float32")
p = te.placeholder((n + 1,), name="p", dtype="float32")
q = te.placeholder((n + 1,), name="q", dtype="float32")
s = te.placeholder((n + 1,), name="s", dtype="float32")
j = te.reduce_axis((0, n + 1), name="j")
s = te.compute((n + 1), lambda j: te.sum(r[i] * A[i][j], axis=j), name="s")
return [A, p, q, s]
