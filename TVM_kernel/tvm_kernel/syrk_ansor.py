import tvm
from tvm import te,auto_scheduler
import numpy as np
import sys

@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def syrk(M,N,beta,alpha):
    A = te.placeholder((N, M), name='A')
    C = te.placeholder((N, N), name='C')
    k = te.reduce_axis((0,M),name='k')
    tmp1 = te.compute((N,N),lambda i,j:te.if_then_else(j<=i,C[i,j]*beta,0))
    tmp2 = te.compute((N,N),lambda i,j:te.sum(te.if_then_else(j<=i,A[i,k]*alpha*A[j,k],0),axis=k))
    C1 =te.compute((N,N),lambda i,j:tmp1[i,j]+tmp2[i,j])
    return [A,C,C1]


def mytuner(n_trial,M,N,beta,alpha):
    target = tvm.target.Target("llvm -mcpu=core-avx2")
    ctx = tvm.device(target.kind.name, 0)
    task = tvm.auto_scheduler.SearchTask(func=syrk, args=(M, N,alpha, beta), target=target)
     # Inspect the computational graph
    print("Computational DAG:")
    print(task.compute_dag)
    
    log_file = "syrk.json"
    tune_option = auto_scheduler.TuningOptions(
        early_stopping = 400,
        num_measure_trials=1000,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )
    
     # Run auto-tuning (search)
    task.tune(tune_option)
    sch, args = task.apply_best(log_file)
    print("Lowered TIR:")
    print(tvm.lower(sch, args, simple_mode=True))
    func = tvm.build(sch, args, target)
    dev=tvm.cpu()
    
    a_np = np.random.uniform(size=(N,M)).astype(np.float32)
    c_np = np.random.uniform(size=(N,N)).astype(np.float32)

    a_tvm = tvm.nd.array(a_np, ctx)
    c_tvm = tvm.nd.array(c_np, ctx)
    c_out = tvm.nd.empty((N,N), device=dev)
    func(a_tvm,c_tvm,c_out)
    print(c_out)
    
    evaluator = func.time_evaluator(func.entry_name, ctx, repeat = 10, min_repeat_ms=500)
    rel = evaluator(a_tvm,c_tvm,c_out).results
    print("Execution time of this operator: max:%.10f s   median:%.10f s   min:%.10f s"
            % (np.max(rel) , np.median(rel), np.min(rel) ) )

    print("Equivalent python schedule:")
    print(task.print_best(log_file))
    
  
if __name__ == '__main__':
    M = 1024
    N = 1024
    alpha = 32412
    beta = 2123
    mytuner(10,M,N,beta,alpha)
    