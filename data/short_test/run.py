from time import perf_counter as timer
from subprocess import run

if __name__ == '__main__':
     tic = timer()
     run(['./bin/pympa',
          'data/short_test/travel_times', 'data/short_test/templates',
          'data/short_test/templates.zmap', 'data/short_test/continuous',
          'data/short_test/settings.yaml', 'data/short_test/outputs.stats'])
     toc = timer()
     print(f"Elapsed time: {toc - tic:.2f} seconds.")
