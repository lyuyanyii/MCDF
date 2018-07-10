import os
import multiprocessing
import time


def get_GPU_status( gpu_id:int, worker_id:int ):
    #assert (0 <= gpu_id and gpu_id <= 3)
    if not (0 <= gpu_id and gpu_id <= 3):
        raise Exception('Nikola only has 4 GPUs.')
    status_file = 'nvidia_info/{}.txt'.format( worker_id )
    os.system( "rm {}".format( status_file ) )
    os.system( "nvidia-smi >> {}".format(status_file) )
    with open( status_file, 'r' ) as f:
        lines = f.readlines()
        status_info = lines[ 8 + gpu_id * 3 ]
        gpu_mem, gpu_util = -1, -1
        for i in status_info.split(' '):
            if 'MiB' in i and gpu_mem == -1:
                gpu_mem = int(i.replace('MiB', ''))
            if '%' in i:
                gpu_util = int(i.replace('%', ''))
        return gpu_mem, gpu_util

    raise Exception('get_GPU_status fails.')

def worker( jobs_que, gpu_que, worker_id ):
    while not jobs_que.empty():
        gpu_id = gpu_que.get()
        mem, usage = get_GPU_status( gpu_id, worker_id )
        print(mem,usage)
        if 12000 - mem >= 3000 and usage < 50:
            try:
                pid = os.fork()

                if pid == 0:
                    job_shell = jobs_que.get()
                    job_shell = 'CUDA_VISIBLE_DEVICES={} '.format(gpu_id) + job_shell + ' >> output.txt'
                    print( job_shell + ' START' )
                    print( 'Lock on gpu {}'.format( gpu_id ) )
                    os.system( job_shell )
                    return
                else:
                    time.sleep( 180 )
                    print( 'Release gpu {}'.format( gpu_id ) )
                    gpu_que.put( gpu_id )
            except:
                gpu_que.put( gpu_id )
                pass
        else:
            gpu_que.put( gpu_id )

if __name__ == '__main__':
    jobs = []
    for alpha in [0.5]:
        for percent in [100]:
            for seed in [0]:
                for dis in ['Gaussian', 'Poisson', 'Laplace', 'Blankout']:
                    jobs.append('python3 main.py --arch Densenet124 --save-folder result/cifar10sub_seed{seed}/alpha_{alp}_percent_{per}_{dis}/ --lr 0.1 --dataset subcifar10 -b 64 --alpha {alp} --epoch 300 --sub-percent {per} --seed {seed} --distribution {dis}'.format( alp=alpha, per=percent, seed=seed, dis=dis ))

    jobs_que = multiprocessing.Queue( len(jobs) )
    for i in jobs:
        jobs_que.put( i )

    gpu_que = multiprocessing.Queue( 4 )
    for i in range(4):
        gpu_que.put( i )

    p_list = []
    for i in range(10):
        p = multiprocessing.Process(target=worker,args=(jobs_que,gpu_que,i))
        p.start()
        p_list.append( p )
    try:
        for p in p_list:
            p.join()
        print("Finish")
    except KeyboardInterrupt:
        for p in p_list:
            p.terminate()
            p.join()
        print("Keyboard interrupt in subcifar")
    finally:
        print("-------Cleaning up Main-------")
    jobs_que.close()
    gpu_que.close()

