import tvm
from tvm import te,auto_scheduler
import numpy as np
import sys

@auto_scheduler.register_workload 
def soai(n):
    A = te.placeholder((n,),name="A",dtype="float32")
    B = te.placeholder((n,),name="B",dtype="float32")
    t= te.reduce_axis((0,n),"t")
    A1 = te.compute((3*n,),lambda i:te.if_then_else(i<n,A[i],0),name='A1')
    B1 = te.compute((5*n,),lambda i:te.if_then_else(tvm.tir.all(i>=2*n,i<3*n),B[i-2*n],0),name='B1')
    C= te.compute((3*n,),lambda p:te.sum(A1[3*t]*B1[p-2*t+2*n],axis=t),name="C")
    return [A,B,C]
  
   
def mytuner(degree):
    target = tvm.target.Target("llvm -mcpu=core-avx2")
    ctx = tvm.device(target.kind.name, 0)
    dev=tvm.cpu()
    task = tvm.auto_scheduler.SearchTask(func=soai, args=(degree,), target=target)
     # Inspect the computational graph
    print("Computational DAG:")
    print(task.compute_dag)
    
    log_file = "soai_padding.json"
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

    
    a_np = np.random.uniform(size=(degree,)).astype(np.float32)
    b_np = np.random.uniform(size=(degree,)).astype(np.float32)

    a_tvm = tvm.nd.array(a_np, ctx)
    b_tvm = tvm.nd.array(b_np, ctx)
    c_tvm = tvm.nd.empty((3*degree,), device=dev)
    
    func(a_tvm,b_tvm,c_tvm)
    evaluator = func.time_evaluator(func.entry_name, ctx, repeat = 10, min_repeat_ms=500)
    rel = evaluator(a_tvm, b_tvm,c_tvm).results
    print("Execution time of this operator: max:%.10f s   median:%.10f s   min:%.10f s"
            % (np.max(rel), np.median(rel), np.min(rel) ) )
    print("Equivalent python schedule:")
    print(task.print_best(log_file))

if __name__ == '__main__':
    
    degree = int(sys.argv[1])
    mytuner(degree)
    