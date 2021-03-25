from time import perf_counter as timer
from dask.distributed import LocalCluster
from subprocess import run

if __name__ == '__main__':
     cluster = LocalCluster()
     address = cluster.scheduler_address

     tic = timer()
     run(['./bin/pympa',
          'data/short_test/travel_times', 'data/short_test/templates',
          'data/short_test/templates.zmap', 'data/short_test/continuous',
          'data/short_test/settings.yaml', 'data/short_test/output.stats', address])
     toc = timer()
     print(f"Elapsed time: {toc - tic:.2f} seconds.")
