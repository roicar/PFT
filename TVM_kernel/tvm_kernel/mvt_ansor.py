import tvm
from tvm import te, auto_scheduler
import numpy as np
import sys

@auto_scheduler.register_workload
def mvt(N):
    A = te.placeholder((N, N), name='A')
    y_1 = te.placeholder((N,), name='y_1')
    y_2 = te.placeholder((N,), name='y_2')
    x1 = te.placeholder((N,), name='x1')
    x2 = te.placeholder((N,), name='x2')
    j = te.reduce_axis((0,N),'j')
    k = te.reduce_axis((0,N),'k')
    x1 = te.compute((N,), lambda i: te.sum(A[i, j] * y_1[j], axis=j), name='x1')
    x2 = te.compute((N,), lambda i: te.sum(A[k, i] * y_2[k],axis=k), name='x2')
 
    return [A,y_1,y_2,x1,x2]


def mytuner(n_trial,N):
    target = tvm.target.Target("llvm -mcpu=core-avx2")
    ctx = tvm.device(target.kind.name, 0)
    dev = tvm.cpu()
    task = tvm.auto_scheduler.SearchTask(func=mvt, args=(N,), target=target)
    
    print("Computational DAG:")
    print(task.compute_dag)
    
    log_file = "mvt.json"
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
    y1_np = np.random.uniform(size=(N,)).astype(np.float32)
    y2_np = np.random.uniform(size=(N,)).astype(np.float32)
    x1_np = np.random.uniform(size=(N,)).astype(np.float32)
    x2_np = np.random.uniform(size=(N,)).astype(np.float32)
    
    x1_np_out = x1_np
    x1_np_out = np.dot(A_np,y1_np)
    x2_np_out = x2_np
    x2_np_out = np.dot(A_np.T,y2_np)
    
    dev = tvm.cpu()
    
    A_tvm = tvm.nd.array(A_np, ctx)
    y1_tvm = tvm.nd.array(y1_np, ctx)
    y2_tvm = tvm.nd.array(y2_np, ctx)
    x1_tvm =tvm.nd.array(x1_np, ctx)
    x2_tvm = tvm.nd.array(x2_np, ctx)

    func(A_tvm,y1_tvm,y2_tvm,x1_tvm,x2_tvm)
    
    print(x1_np_out,x1_tvm,x2_np_out,x2_tvm)
    
    
    
    evaluator = func.time_evaluator(func.entry_name, ctx, repeat=10, min_repeat_ms=500)
    rel = evaluator(A_tvm,y1_tvm,y2_tvm,x1_tvm,x2_tvm).results
    print("Execution time of this operator: max:%.10f s   median:%.10f s   min:%.10f s"
          % (np.max(rel) , np.median(rel) , np.min(rel) ))

    print("Equivalent python schedule:")
    print(task.print_best(log_file))
    
    
    
if __name__ == '__main__':
    N = int(sys.argv[1])   
    mytuner(10,N)