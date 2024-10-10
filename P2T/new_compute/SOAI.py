def SOAI(n):
A = te.placeholder((n + 1,), name="A", dtype="float32")
B = te.placeholder((n + 1,), name="B", dtype="float32")
C = te.placeholder((3*n + 1,), name="C", dtype="float32")
t1 = te.reduce_axis((0, n + 1), name="t1")

A1 = te.compute(
    (3*n + 1,), 
    lambda i:
        te.if_then_else(
            tvm.tir.all(0 <= i <= n), 
            A[i], 
            0
    ), 
    name='A1'
)

B1 = te.compute(
    (5*n + 1,), 
    lambda i:
        te.if_then_else(
            tvm.tir.all(2*n<= i <= 3*n), 
            B[i - 2*n], 
            0
    ), 
    name='B1'
)
C = te.compute((3*n + 1), lambda t2: te.sum(A1[3*t1] * B1[2*n - 2*t1 + t2], axis=t1), name="C")
return [A, B, C]
