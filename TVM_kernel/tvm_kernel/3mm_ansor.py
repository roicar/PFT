import tvm
from tvm import te, auto_scheduler
import numpy as np
import sys

@auto_scheduler.register_workload
def continuous_matrix_multiply3(NI, NJ, NK, NL):
    
         # 定义TVM数组
    A = te.placeholder((NI, NK), name='A')
    B = te.placeholder((NK, NJ), name='B')
    C = te.placeholder((NJ, NI), name='C')
    D = te.placeholder((NI, NL), name='D')
      
    k =te.reduce_axis((0,NK),"k")
    t = te.reduce_axis((0,NI),"t")
    x =te.reduce_axis((0,NJ),"x")
    
        
    E = te.compute((NI, NJ), lambda i, j: te.sum( A[i, k] * B[k, j], axis=k))
    
    F = te.compute((NJ, NL), lambda i, j: te.sum( C[i, t] * D[t, j], axis=t))
    
    G = te.compute((NI, NL), lambda i, j: te.sum(E[i, x] * F[x, j], axis=x))
    
    return [A, B, C, D, G]


def mytuner(n_trial, NI, NJ, NK, NL):
    target = tvm.target.Target("llvm -mcpu=core-avx2")
    ctx = tvm.device(target.kind.name, 0)
    dev = tvm.cpu()
        # Pass k as an API parameter
    task = tvm.auto_scheduler.SearchTask(func=continuous_matrix_multiply3, args=(NI, NJ, NK, NL), target=target)
    
    print("Computational DAG:")
    print(task.compute_dag)

    log_file = "continuous_matrix_multiply3.json"
    tune_option = auto_scheduler.TuningOptions(
        early_stopping=400,
        num_measure_trials=1000,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )
    task.tune(tune_option)  # Pass k as an API parameter
    sch, args = task.apply_best(log_file)
    print("Lowered TIR:")
    print(tvm.lower(sch, args, simple_mode=True))
    
    func = tvm.build(sch, args, target)
    
    a_np = np.random.uniform(size=(NI, NK)).astype(np.float32)
    b_np = np.random.uniform(size=(NK, NJ)).astype(np.float32)
    c_np = np.random.uniform(size=(NJ, NI)).astype(np.float32)
    d_np = np.random.uniform(size=(NI, NL)).astype(np.float32)
    
    g_np_out = np.dot(np.dot(a_np,b_np),np.dot(c_np,d_np))
    
    a_tvm = tvm.nd.array(a_np, ctx)
    b_tvm = tvm.nd.array(b_np, ctx)
    c_tvm = tvm.nd.array(c_np, ctx)
    d_tvm = tvm.nd.array(d_np, ctx)
    
    g_out= tvm.nd.empty((NI, NL), device=dev)
    func(a_tvm, b_tvm, c_tvm, d_tvm, g_out)
    np.testing.assert_allclose(g_np_out, g_out.numpy(), rtol=1e-3)
    
    
    evaluator = func.time_evaluator(func.entry_name, ctx, repeat=10, min_repeat_ms=500)
    rel = evaluator(a_tvm, b_tvm, c_tvm, d_tvm, g_out).results
    print("Execution time of this operator: max:%.10f s   median:%.10f s   min:%.10f s"
          % (np.max(rel) , np.median(rel) , np.min(rel) ))

    print("Equivalent python schedule:")
    print(task.print_best(log_file))


if __name__ == '__main__':
    NI = int(sys.argv[1])
    NJ = int(sys.argv[2])
    
    NK = int(sys.argv[3])
    NL = int(sys.argv[4])
    mytuner(10, NI, NJ, NK, NL)
