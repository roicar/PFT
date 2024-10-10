import tvm
from tvm import te,auto_scheduler
import numpy as np
import sys
import time

@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def trmm(M,N,alpha):
    A = te.placeholder((M, M), name='A')
    B = te.placeholder((M, N), name='B')
    k =te.reduce_axis((0,M),'k')
    B1 = te.compute((M,N),lambda i,j:te.sum(te.if_then_else(k>=i+1,A[k,i]*B[k,j],B[i,j]),axis=k))
    B1 = te.compute((M,N),lambda i,j:alpha*B1[i,j])
    
    return [A,B,B1]


def mytuner(n_trial,M,N,alpha):
    target = tvm.target.Target("llvm -mcpu=core-avx2")
    ctx = tvm.device(target.kind.name, 0)
    dev=tvm.cpu()
    task = tvm.auto_scheduler.SearchTask(func=trmm, args=(M,N,alpha), target=target)
     # Inspect the computational graph
    print("Computational DAG:")
    print(task.compute_dag)
    
    log_file = "trmm.json"
    tune_option = auto_scheduler.TuningOptions(
        early_stopping = 400,
        num_measure_trials=100,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )
    
     # Run auto-tuning (search)
    task.tune(tune_option)
    sch, args = task.apply_best(log_file)
    print("Lowered TIR:")
    print(tvm.lower(sch, args, simple_mode=True))
    
    func = tvm.build(sch, args, target)

    a_np = np.tril(np.random.rand(M, M).astype(np.float32))
    b_np = np.random.uniform(size=(M, N)).astype(np.float32)
    np_out = alpha*np.dot(a_np.T,b_np)
    print(np_out)
    start_transfer = time.time()
    a_tvm = tvm.nd.array(a_np, ctx)
    b_tvm = tvm.nd.array(b_np, ctx)
    
    out = tvm.nd.empty((M, N), device=dev)
    
    func(a_tvm,b_tvm,out)
    end_transfer = time.time()
    
    all_time = end_transfer - start_transfer
    print("func(a_tvm, b_tvm, c_tvm) 的执行时间为:", all_time, "秒")
    print(out)
    evaluator = func.time_evaluator(func.entry_name, ctx, repeat = 10, min_repeat_ms=500)
    rel = evaluator(a_tvm, b_tvm,out).results
    print("Execution time of this operator: max:%.10f s   median:%.10f s   min:%.10f s"
            % (np.max(rel) , np.median(rel), np.min(rel) ) )

    print("Equivalent python schedule:")
    print(task.print_best(log_file))

if __name__ == '__main__':
    M = int(sys.argv[1])
    N = int(sys.argv[2])
    mytuner(10,M,N,32412)