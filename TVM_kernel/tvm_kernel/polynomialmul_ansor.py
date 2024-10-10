import tvm
from tvm import te,auto_scheduler
import numpy as np
import sys

@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def polymm(degreeA,degreeB):
    N = degreeA+degreeB+1
    A = te.placeholder((degreeA+1,),name="A",dtype="float32")
    B = te.placeholder((degreeB+1,),name="B",dtype="float32")

    k= te.reduce_axis((0,degreeB+1),"k")
    out = te.compute((N,),lambda i:te.sum(te.if_then_else(i-k>=0,A[i-k],0)*B[k],axis=k,where=(i-k<=degreeA)),name="out")


    return [A,B,out]


def mytuner(n_trial,degreeA,degreeB):
    target = tvm.target.Target("llvm -mcpu=core-avx2")
    ctx = tvm.device(target.kind.name, 0)
    dev=tvm.cpu()
    task = tvm.auto_scheduler.SearchTask(func=polymm, args=(degreeA, degreeB), target=target)
     # Inspect the computational graph
    print("Computational DAG:")
    print(task.compute_dag)
    
    log_file = "polymm.json"
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

    
    a_np = np.random.uniform(size=(degreeA+1,)).astype(np.float32)
    b_np = np.random.uniform(size=(degreeB+1,)).astype(np.float32)
    out_np = np.convolve(a_np,b_np,mode="full")
    a_tvm = tvm.nd.array(a_np, ctx)
    b_tvm = tvm.nd.array(b_np, ctx)
    c_tvm = tvm.nd.empty(out_np.shape, device=dev)
    
    func(a_tvm,b_tvm,c_tvm)
    # np.testing.assert_allclose(out_np, c_tvm.numpy(), rtol=1e-3)
    evaluator = func.time_evaluator(func.entry_name, ctx, repeat = 10, min_repeat_ms=500)
    rel = evaluator(a_tvm, b_tvm,c_tvm).results
    print("Execution time of this operator: max:%.10f s   median:%.10f s   min:%.10f s"
            % (np.max(rel), np.median(rel), np.min(rel) ) )
    print("Equivalent python schedule:")
    print(task.print_best(log_file))

if __name__ == '__main__':
    
    degreeA = int(sys.argv[1])
    degreeB = int(sys.argv[2])
    mytuner(10,degreeA,degreeB)
    