import tvm
from tvm import te, auto_scheduler
import numpy as np
import sys
import time

@auto_scheduler.register_workload
def bigc(M,N):

    A = te.placeholder((M,N), name='A')
    r = te.placeholder((M,), name='r')
    p = te.placeholder((N,), name='p')
    
    k = te.reduce_axis((0,M),'k')
    j = te.reduce_axis((0,N),'j')
    s = te.compute((M,), lambda i:  te.sum(r[k] * A[k, i], axis=k))
    q = te.compute((N,), lambda i:  te.sum(A[i, j] * p[j], axis=j))

    return [A,r,p,s,q]
    
def mytuner(n_trial,M,N):
    target = tvm.target.Target("llvm -mcpu=core-avx2")
    ctx = tvm.device(target.kind.name, 0)
    dev = tvm.cpu()

    # Pass k as an API parameter
    task = tvm.auto_scheduler.SearchTask(func=bigc, args=(M,N), target=target)
    
    print("Computational DAG:")
    print(task.compute_dag)

    log_file = "bigc.json"
    tune_option = auto_scheduler.TuningOptions(
        early_stopping=400,
        num_measure_trials=10,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )
    task.tune(tune_option)  # Pass k as an API parameter
    sch, args = task.apply_best(log_file)
    print("Lowered TIR:")
    print(tvm.lower(sch, args, simple_mode=True))
    func = tvm.build(sch, args, target)
    
    a_np = np.random.uniform(size=(M,N)).astype(np.float32)
    r_np = np.random.uniform(size=(M,)).astype(np.float32)
    p_np = np.random.uniform(size=(N,)).astype(np.float32)
    s_np = np.zeros(M,).astype(np.float32)
    q_np = np.zeros(N,).astype(np.float32)
    
    
    s_np_out = np.dot(a_np.T, r_np)
    q_np_out = np.dot(a_np,p_np)
    
    
    start_transfer =time.time()
    a_tvm = tvm.nd.array(a_np, ctx)
    r_tvm = tvm.nd.array(r_np, ctx)
    p_tvm = tvm.nd.array(p_np, ctx)
    s_tvm = tvm.nd.array(s_np, ctx)
    q_tvm = tvm.nd.array(q_np, ctx)

    func(a_tvm,r_tvm,p_tvm,s_tvm,q_tvm)
    
    end_transfer = time.time()
    
    all_time = end_transfer - start_transfer
    print("func(a_tvm, b_tvm, c_tvm) 的执行时间为:", all_time, "秒")
    print(s_np_out,s_tvm,q_np_out,q_tvm)
    
    evaluator = func.time_evaluator(func.entry_name, ctx, repeat=10, min_repeat_ms=50)
    rel = evaluator(a_tvm,r_tvm,p_tvm,s_tvm,q_tvm).results
    print("Execution time of this operator: max:%.10f s   median:%.10f s   min:%.10f s"
          % (np.max(rel) , np.median(rel) , np.min(rel) ))

    print("Equivalent python schedule:")
    print(task.print_best(log_file))
    

if __name__ == '__main__':
    M = int(sys.argv[1])
    N = int(sys.argv[2])
    mytuner(10,M,N)
