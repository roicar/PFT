import tvm
from tvm import te, auto_scheduler
import numpy as np
import sys

@auto_scheduler.register_workload
def atax(M,N):

    # 定义TVM数组
    A = te.placeholder((M, N), name='A')
    x = te.placeholder((N,), name='x')
    y = te.placeholder((N,), name='y')
    k = te.reduce_axis((0,N),"k")
    t = te.reduce_axis((0,N),"t")
        # 中间矩阵 tmp
    tmp = te.compute((M,), lambda i: te.sum(A[i, k] * x[k], axis=k))

    # 更新 y
    y = te.compute((N,), lambda i:te.sum(A[t, i] * tmp[t], axis=t))
    
    
    return [A, x,y]


def mytuner(n_trial,M,N):
    target = tvm.target.Target("llvm -mcpu=core-avx2")
    ctx = tvm.device(target.kind.name, 0)
    dev = tvm.cpu()

    # Pass k as an API parameter
    task = tvm.auto_scheduler.SearchTask(func=atax, args=(M,N), target=target)
    
    print("Computational DAG:")
    print(task.compute_dag)

    log_file = "atax.json"
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
    
    a_np = np.random.uniform(size=(M,N)).astype(np.float32)
    x_np = np.random.uniform(size=(N,)).astype(np.float32)
    y_np = np.random.uniform(size=(N,)).astype(np.float32)
    
    y_np_out = y_np
    tmp = np.dot(a_np, x_np)
    y_np_out = np.dot(a_np.T, tmp)
    
    
    
    
    a_tvm = tvm.nd.array(a_np, ctx)
    x_tvm = tvm.nd.array(x_np, ctx)
    y_tvm = tvm.nd.array(y_np, ctx)
    
    func(a_tvm,x_tvm,y_tvm)
    np.testing.assert_allclose(y_np_out, y_tvm.numpy(), rtol=1e-3)
    
    
    evaluator = func.time_evaluator(func.entry_name, ctx, repeat=10, min_repeat_ms=500)
    rel = evaluator(a_tvm, x_tvm,y_tvm).results
    print("Execution time of this operator: max:%.10f s   median:%.10f s   min:%.10f s"
          % (np.max(rel) , np.median(rel) , np.min(rel) ))

    print("Equivalent python schedule:")
    print(task.print_best(log_file))



if __name__ == '__main__':
    M = int(sys.argv[1])
    N = int(sys.argv[2])
    mytuner(10,M,N)

