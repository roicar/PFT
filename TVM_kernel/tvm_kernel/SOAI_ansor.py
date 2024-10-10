import tvm
from tvm import te,auto_scheduler
import numpy as np
import sys
import time

@auto_scheduler.register_workload 
def soai(n):
    A = te.placeholder((n,),name="A",dtype="float32")
    B = te.placeholder((n,),name="B",dtype="float32")
    t= te.reduce_axis((0,n),"t")
    C= te.compute((3*n,),lambda p:te.sum(te.if_then_else(tvm.tir.all(p-2*t>=0,p-2*t<n,3*t<n),A[3*t]*B[p-2*t],0),axis=t),name="C")
    return [A,B,C]
  
   
def mytuner(degree):
    target = tvm.target.Target("llvm -mcpu=core-avx2")
    ctx = tvm.device(target.kind.name, 0)
    dev=tvm.cpu()
    task = tvm.auto_scheduler.SearchTask(func=soai, args=(degree,), target=target)
     # Inspect the computational graph
    print("Computational DAG:")
    print(task.compute_dag)
    
    log_file = "soai.json"
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

    
    a_np = np.random.uniform(size=(degree,)).astype(np.float32)
    b_np = np.random.uniform(size=(degree,)).astype(np.float32)

    a_tvm = tvm.nd.array(a_np, ctx)
    b_tvm = tvm.nd.array(b_np, ctx)
    c_tvm = tvm.nd.empty((3*degree,), device=dev)
    start_transfer = time.time()
    func(a_tvm,b_tvm,c_tvm)
    end_transfer = time.time()
    
    all_time = end_transfer - start_transfer
    print("func(a_tvm, b_tvm, c_tvm) 的执行时间为:", all_time, "秒")
    
    evaluator = func.time_evaluator(func.entry_name, ctx, repeat = 10, min_repeat_ms=500)
    rel = evaluator(a_tvm, b_tvm,c_tvm).results
    print("Execution time of this operator: max:%.10f s   median:%.10f s   min:%.10f s"
            % (np.max(rel), np.median(rel), np.min(rel) ) )
    print("Equivalent python schedule:")
    print(task.print_best(log_file))

if __name__ == '__main__':
    
    degree = int(sys.argv[1])
    mytuner(degree)
    