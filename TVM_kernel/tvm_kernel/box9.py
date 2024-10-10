import tvm
from tvm import te, auto_scheduler
import numpy as np
import sys


@auto_scheduler.register_workload
def seidel_9(N):
    A = te.placeholder((N, N), name='A')
    
    A_out = te.compute((N,N),lambda i,j:te.if_then_else(tvm.tir.all(i>=1,i<=N-2,j>=1,j<=N-2),(A[i-1,j+1]+ A[i-1][j] +A[i-1][j+1] + A[i][j-1] + A[i][j] + A[i][j+1]+ A[i+1][j-1]+A[i+1][j]+A[i+1][j+1] )/9,A[i][j]))
    
    return [A,A_out]


def mytuner(n_trial,N):
    target = tvm.target.Target("llvm -mcpu=core-avx2")
    ctx = tvm.device(target.kind.name, 0)
    dev = tvm.cpu()
    task = tvm.auto_scheduler.SearchTask(func=seidel_9, args=(N,), target=target)
    
    print("Computational DAG:")
    print(task.compute_dag)
    
    log_file = "seidel_9.json"
    tune_option = auto_scheduler.TuningOptions(
        early_stopping=400,
        num_measure_trials=500,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )
    
    task.tune(tune_option) 
    sch, args = task.apply_best(log_file)
    print("Lowered TIR:")
    print(tvm.lower(sch, args, simple_mode=True))
    func = tvm.build(sch, args, target)
    
      
    A_np = np.random.uniform(size=(N,N)).astype(np.float32)
    dev = tvm.cpu()
    A_tvm = tvm.nd.array(A_np, ctx)
    A_out = tvm.nd.empty((N,N), device=dev)
    
    func(A_tvm,A_out)
    print(A_tvm,A_out)
    
    evaluator = func.time_evaluator(func.entry_name, ctx, repeat = 10, min_repeat_ms=500)
    rel = evaluator(A_tvm,A_out).results
    print("Execution time of this operator: max:%.10f s   median:%.10f s   min:%.10f s"
          % (np.max(rel) , np.median(rel) , np.min(rel) ))
    print("Equivalent python schedule:")
    print(task.print_best(log_file))

if __name__ == '__main__':
    N = int(sys.argv[1])   
    mytuner(10,N)
