def polynomialmul(n):
a = te.placeholder((n + 1,), name="a", dtype="float32")
b = te.placeholder((n + 1,), name="b", dtype="float32")
c = te.placeholder((2*n + 1,), name="c", dtype="float32")
t2 = te.reduce_axis((0, n + 1), name="t2")

a1 = te.compute(
    (3*n + 1,), 
    lambda i:
        te.if_then_else(
            tvm.tir.all(0 <= i - n <= n), 
            a[i - n], 
            0
    ), 
    name='a1'
)
c = te.compute((2*n + 1), lambda t1: te.sum(a1[t1 + t2] * b[n - t2], axis=t2), name="c")
return [a, b, c]
