import tvm
from tvm import te, auto_scheduler
import numpy as np
import sys

def seidel(N):
    target = tvm.target.Target("llvm -mcpu=core-avx2")
    ctx = tvm.device(target.kind.name, 0)
    A = te.placeholder((N, N), name='A')
    
    A1 = te.compute((N-4,N-4),lambda i,j:(A[i][j+2] + A[i+1][j+2] + A[i+2][j+2] + A[i+3][j+2] + A[i+4][j+2]+ A[i+2][j] + A[i+2][j+1] + A[i+2][j+3]+A[i+2][j+4])/9)
    
    s = te.create_schedule(A1.op)
    print(tvm.lower(s,[A,A1],simple_mode=True))
    func = tvm.build(s,[A,A1],target=target) 
    
    A_np = np.random.uniform(size=(N,N)).astype(np.float32)
    print(A_np)
    dev = tvm.cpu()
    A_tvm = tvm.nd.array(A_np, ctx)
    
    A_out = tvm.nd.empty((N-4,N-4), device=dev)
    func(A_tvm,A_out)
    print(A_out)    
    evaluator = func.time_evaluator(func.entry_name, ctx, repeat = 10, min_repeat_ms=50)
    rel = evaluator(A_tvm,A_out).results
    print("Execution time of this operator: max:%.10f s   median:%.10f s   min:%.10f s"
            % (np.max(rel) , np.median(rel) , np.min(rel)) )
    
if __name__ == '__main__':
    N = int(sys.argv[1])
    seidel(N)