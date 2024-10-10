import tvm
from tvm import te,auto_scheduler
import numpy as np
import sys


@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def gemm(NI,NJ,NK,beta,alpha):
    A = te.placeholder((NI,NK ), name='A')
    B = te.placeholder((NK,NJ ), name='B')
    C =te.placeholder((NI,NJ),name='C')
    k = te.reduce_axis((0,NK),'k')
    
    C1 = te.compute((NI,NJ),lambda i ,j:beta*C[i,j])
    C2 = te.compute((NI,NJ),lambda i ,j:te.sum(alpha*A[i,k]*B[k,j],axis=k))
    C2 = te.compute((NI,NJ),lambda i ,j:C2[i,j]+C1[i,j])
    return [A,B,C,C2]


def mytuner(n_trial,NI,NJ,NK,beta,alpha):
    target = tvm.target.Target("llvm -mcpu=core-avx2")
    ctx = tvm.device(target.kind.name, 0)
    task = tvm.auto_scheduler.SearchTask(func=gemm, args=(NI,NJ,NK,beta,alpha), target=target)
     # Inspect the computational graph
    print("Computational DAG:")
    print(task.compute_dag)
    
    log_file = "gemm.json"
    tune_option = auto_scheduler.TuningOptions(
        early_stopping = 400,
        num_measure_trials=500,
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
    
     
    a_np = np.random.uniform(size=(NI,NK )).astype(np.float32)
    b_np = np.random.uniform(size=(NK,NJ)).astype(np.float32)
    c_np = np.random.uniform(size=(NI,NJ)).astype(np.float32)
    
    c_np_out = alpha*np.dot(a_np,b_np)+beta*c_np
    
    a_tvm = tvm.nd.array(a_np, ctx)
    b_tvm = tvm.nd.array(b_np, ctx)
    c_tvm = tvm.nd.array(c_np, ctx)
    c_out = tvm.nd.empty((NI,NJ), device=dev)
    
    func(a_tvm,b_tvm,c_tvm,c_out)
    np.testing.assert_allclose(c_np_out, c_out.numpy(), rtol=1e-3)
    print(c_np_out,c_out)
    
    
    
    evaluator = func.time_evaluator(func.entry_name, ctx, repeat = 10, min_repeat_ms=500)
    rel = evaluator(a_tvm, b_tvm,c_tvm,c_out).results
    print("Execution time of this operator: max:%.10f ns   median:%.10f ns   min:%.10f ns"
            % (np.max(rel), np.median(rel), np.min(rel)) )

    print("Equivalent python schedule:")
    print(task.print_best(log_file))
    
if __name__ == '__main__':
    NI = int(sys.argv[1])
    NJ = int(sys.argv[2])
    NK = int(sys.argv[3])
    beta = int(sys.argv[4])
    alpha = int(sys.argv[5])
    mytuner(10,NI,NJ,NK,beta,alpha)
