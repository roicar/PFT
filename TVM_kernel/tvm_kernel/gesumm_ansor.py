import tvm
from tvm import te,auto_scheduler
import numpy as np
import sys

@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def gesumm(N,beta,alpha):
    A = te.placeholder((N,N), name='A')
    B = te.placeholder((N,N ), name='B')
    x =te.placeholder((N, ), name='x') 

    
    j =te.reduce_axis((0,N),'j')  

    y = te.compute((N,),lambda i:te.sum(alpha*A[i,j]*x[j]+beta*B[i,j]*x[j],axis=j))

    
    return [A,B,x,y]


def mytuner(n_trial,N,beta,alpha):
    target = tvm.target.Target("llvm -mcpu=core-avx2")
    ctx = tvm.device(target.kind.name, 0)
    task = tvm.auto_scheduler.SearchTask(func=gesumm, args=(N,beta,alpha), target=target)
     # Inspect the computational graph
    print("Computational DAG:")
    print(task.compute_dag)
    
    log_file = "gesumm.json"
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
    
    dev=tvm.cpu()
     
    a_np = np.random.uniform(size=(N,N )).astype(np.float32)
    b_np = np.random.uniform(size=(N,N)).astype(np.float32)
    x_np = np.random.uniform(size=(N,)).astype(np.float32)
    
    y= alpha*np.dot(a_np,x_np)+beta*np.dot(b_np,x_np)
    print(y)
    y_out = tvm.nd.empty((N,), device=dev)
    
    a_tvm = tvm.nd.array(a_np, ctx)
    b_tvm = tvm.nd.array(b_np, ctx)
    x_tvm = tvm.nd.array(x_np, ctx)

    
    func(a_tvm,b_tvm,x_tvm,y_out)
    print(y_out)
    
    
    evaluator = func.time_evaluator(func.entry_name, ctx, repeat = 10, min_repeat_ms=50)
    rel = evaluator(a_tvm,b_tvm,x_tvm,y_out).results
    print("Execution time of this operator: max:%.3f ns   median:%.3f ns   min:%.3f ns"
            % (np.max(rel) * 1000000000, np.median(rel) * 1000000000, np.min(rel) * 1000000000) )

    print("Equivalent python schedule:")
    print(task.print_best(log_file))
    


if __name__ == '__main__':
    N = int(sys.argv[1])
    mytuner(10,N,1.2,1.5)
    
    
    