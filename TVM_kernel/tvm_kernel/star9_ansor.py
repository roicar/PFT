import tvm
from tvm import te, auto_scheduler
import numpy as np
import sys


@auto_scheduler.register_workload
def star9(N):
    target = tvm.target.Target("llvm -mcpu=core-avx2")
    ctx = tvm.device(target.kind.name, 0)
    A = te.placeholder((N, N), name='A')

    A_out =  te.compute((N,N),lambda i,j:te.if_then_else(tvm.tir.all(i>=2,i<=N-3,j>=2,j<=N-3),( A[i-2][j] + A[i-1][j] + A[i][j] + A[i+1][j] + A[i+2][j]+ A[i][j-2] + A[i][j-1] + A[i][j+1]+A[i][j+2])/9,A[i][j]))
    
    return [A,A_out]



def mytuner(n_trial,N):
    target = tvm.target.Target("llvm -mcpu=core-avx2")
    ctx = tvm.device(target.kind.name, 0)
    dev = tvm.cpu()
    task = tvm.auto_scheduler.SearchTask(func=star9, args=(N,), target=target)
    
    print("Computational DAG:")
    print(task.compute_dag)
    
    log_file = "star9.json"
    tune_option = auto_scheduler.TuningOptions(
        early_stopping=400,
        num_measure_trials=1000,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )
    
    task.tune(tune_option) 
    sch, args = task.apply_best(log_file)
    print("Lowered TIR:")
    print(tvm.lower(sch, args, simple_mode=True))
    func = tvm.build(sch, args, target)
    
    
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
    N = int(sys.argv[1])   
    mytuner(10,N)

    