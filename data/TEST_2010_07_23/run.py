from time import perf_counter as timer
import runpy

tic = timer()
runpy.run_module('pympa', run_name='__main__')
toc = timer()
print(f"Elapsed time: {toc - tic:.2f} seconds.")
