import tvm
from tvm import te, auto_scheduler
import numpy as np
import sys
import time

@auto_scheduler.register_workload
def continuous_matrix_multiply(NI, NJ, NK, NL):
    Target = tvm.target.Target(target="llvm -mcpu=core-avx2")
    ctx = tvm.device(Target.kind.name, 0)

    A = te.placeholder((NI, NK), name='A')
    B = te.placeholder((NK, NJ), name='B')
    C = te.placeholder((NJ, NL), name='C')
    D = te.placeholder((NI, NL), name='D')
    k =te.reduce_axis((0,NK),name="k")
    t = te.reduce_axis((0,NK),"t")
    # 定义中间矩阵 tmp
    tmp = te.compute((NI, NJ), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k))

    D = te.compute((NI, NL), lambda i, j: te.sum(tmp[i,t]*C[t,j],axis=t))

    return [A, B, C, D]


def mytuner(n_trial, NI, NJ, NK, NL):
    target = tvm.target.Target("llvm -mcpu=core-avx2")
    ctx = tvm.device(target.kind.name, 0)
    dev = tvm.cpu()

    # Pass k as an API parameter
    task = tvm.auto_scheduler.SearchTask(func=continuous_matrix_multiply, args=(NI, NJ, NK, NL), target=target)

    print("Computational DAG:")
    print(task.compute_dag)

    log_file = "2mm.json"
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
    a_np = np.random.uniform(size=(NI, NK)).astype(np.float32)
    b_np = np.random.uniform(size=(NK, NJ)).astype(np.float32)
    c_np = np.random.uniform(size=(NJ, NL)).astype(np.float32)
    
    d_np_out = np.dot(np.dot(a_np,b_np), c_np)
    
    start_transfer = time.time()
    a_tvm = tvm.nd.array(a_np, ctx)
    b_tvm = tvm.nd.array(b_np, ctx)
    c_tvm = tvm.nd.array(c_np, ctx)
    
    d_out = tvm.nd.empty((NI, NL), device=dev)
    func(a_tvm, b_tvm, c_tvm, d_out)
    end_transfer = time.time()
    
    all_time = end_transfer - start_transfer
    print("func(a_tvm, b_tvm, c_tvm) 的执行时间为:", all_time, "秒")
    
    
    
    # np.testing.assert_allclose(d_np_out, d_out.numpy(), rtol=1e-3)
    
    evaluator = func.time_evaluator(func.entry_name, ctx, repeat=10, min_repeat_ms=500)
    rel = evaluator(a_tvm, b_tvm, c_tvm, d_out).results
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
