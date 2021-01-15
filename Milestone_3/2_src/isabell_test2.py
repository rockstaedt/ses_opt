import concurrent.futures
import time

start = time.perf_counter()
secs = [5, 4, 3, 2, 1]

for i in secs:
    print(f'Sleeping {i} second(s) ...')
    time.sleep(i)
    print (f'Done Sleeping...{i}')
        
finish = time.perf_counter()
print(f'Finished in {round(finish-start, 2)} second(s)')