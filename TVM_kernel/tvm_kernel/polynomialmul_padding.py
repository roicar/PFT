import tvm
from tvm import te,auto_scheduler
import sys
import time
import numpy as np
from tvm import tir

@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def polynomialmul_padding2(degree):
    #degree是多项式的最高次数
    n = degree
    
    #创建定义A,B大小
    A = te.placeholder((n+1,),name="A",dtype="float32")
    B = te.placeholder((n+1,),name="B",dtype="float32")
    t = te.reduce_axis((0,n+1),"t")
    
    padding_size = 2*n
    padding_A_shape = n+1+padding_size
    
    A1 = te.compute((padding_A_shape,),lambda i: te.if_then_else(tvm.tir.all(i >= n, i <=2*n),A[i - n],0.0), name="A1")
    
    C = te.compute((2*n+1,),lambda p:te.sum(A1[t+p]*B[n-t],axis=t))
    
    return [A,B,C]  


def mytuner(n_trial,degree):
    target = tvm.target.Target("cuda")
    ctx = tvm.device(target.kind.name, 0)
    dev = tvm.cuda()
    task = tvm.auto_scheduler.SearchTask(func=polynomialmul_padding2, args=(degree,), target=target)
     # Inspect the computational graph
    print("Computational DAG:")
    print(task.compute_dag)
    
    log_file = "polynomialmul_padding_pluto.json"
    tune_option = auto_scheduler.TuningOptions(
        early_stopping = 400,
        num_measure_trials=100,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )
    
    task.tune(tune_option)
    sch, args = task.apply_best(log_file)
    print("Lowered TIR:")
    print(tvm.lower(sch, args, simple_mode=True))
    
    func = tvm.build(sch, args, target)
    
    a_np = np.random.randint(0,10,size=(degree+1,)).astype('float32')
    b_np = np.random.randint(0,10,size=(degree+1,)).astype('float32')
    c_np = np.convolve(a_np,b_np,mode="full")
 
    
    start_transfer = time.time()
    a_tvm = tvm.nd.array(a_np, ctx)
    b_tvm = tvm.nd.array(b_np, ctx)
    c_tvm = tvm.nd.empty(c_np.shape, device=dev)
    
    func(a_tvm,b_tvm,c_tvm)
    end_transfer = time.time()
    
    all_time = end_transfer - start_transfer
    print("func(a_tvm, b_tvm, c_tvm) 的执行时间为:", all_time, "秒")

    print(c_tvm)
    
    evaluator = func.time_evaluator(func.entry_name, ctx, repeat = 10, min_repeat_ms=500)
    rel = evaluator(a_tvm, b_tvm,c_tvm).results
    print("Execution time of this operator: max:%.10f s   median:%.10f s   min:%.10f s"
            % (np.max(rel), np.median(rel), np.min(rel) ) )
    print("Equivalent python schedule:")
    print(task.print_best(log_file))    
    
    

    

if __name__ == '__main__':
    degree = int(sys.argv[1])
    mytuner(10,degree)
    
    