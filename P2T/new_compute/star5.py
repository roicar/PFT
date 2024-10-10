def star5(n):
a = te.placeholder(((n, n),), name="a", dtype="float32")
b = te.placeholder(((n, n),), name="b", dtype="float32")

b = te.compute(
    ((n - 2, n - 2), 
    lambda i, j:
        (
        (a[(1 + i) - 1][(1 + j)] + a[(1 + i)][(1 + j) + 1] + a[(1 + i)][(1 + j)] + a[(1 + i) + 1][(1 + j)] + a[(1 + i)][(1 + j) - 1]) / 5.0
    ), 
    name='b'
)
return [a, b]
