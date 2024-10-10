import tvm
from tvm import te, auto_scheduler
import numpy as np
import sys


def box25(N):
    target = tvm.target.Target("llvm -mcpu=core-avx2")
    ctx = tvm.device(target.kind.name, 0)
    A = te.placeholder((N, N), name='A')
    
    
    k =te.reduce_axis((0,N),"k")

    A_out = te.compute((N,N),lambda i,j:te.if_then_else(tvm.tir.all(i>=2,i<=N-3,j>=2,j<=N-3),
            (A[i-2][j-2] + A[i-2][j-1] + A[i-2][j] + A[i-2][j+1] + A[i-2][j+2]
          + A[i-1][j-2] + A[i-1][j-1] + A[i-1][j] + A[i-1][j+1] + A[i-1][j+2]
          + A[i][j-2] + A[i][j-1] + A[i][j] + A[i][j+1] + A[i][j+2]
          + A[i+1][j-2] + A[i+1][j-1] + A[i+1][j] + A[i+1][j+1] + A[i+1][j+2]
          + A[i+2][j-2] + A[i+2][j-1] + A[i+2][j] + A[i+2][j+1] + A[i+2][j+2]) / 25.0
,A[i,j]))
    
    
    s = te.create_schedule(A_out.op)
    print(tvm.lower(s,[A,A_out],simple_mode=True))
    func = tvm.build(s,[A,A_out],target=target )
    
    
    A_np = np.random.uniform(size=(N,N)).astype(np.float32)
    print(A_np)
    dev = tvm.cpu()
    A_tvm = tvm.nd.array(A_np, ctx)
    
    
    A_out = tvm.nd.empty((N,N), device=dev)


    func(A_tvm,A_out)
    print(A_out)
    
    evaluator = func.time_evaluator(func.entry_name, ctx, repeat = 10, min_repeat_ms=50)
    rel = evaluator(A_tvm,A_out).results
    print("Execution time of this operator: max:%.10f s   median:%.10f s   min:%.10f s"
            % (np.max(rel) , np.median(rel) , np.min(rel)) ) 
    

    
if __name__ == '__main__':
    
    box25(1024)                  
                 
    
    
    
    