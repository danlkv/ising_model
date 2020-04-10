from multiprocessing.dummy import Pool

pool = Pool(processes=32)

y = pool.map(lambda x: x**2, range(2000))

print('Done map:', y)
