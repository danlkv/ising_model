from loguru import logger as log
import sys
from multiprocessing.dummy import Pool
import platform
import psutil

log.remove()
print_format = '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green>\
 [{extra[node]}:R{extra[rank]} Mem:{extra[mempc]}%]\
 <level>{level}\t\
| <cyan>{function}</cyan>:<cyan>{line}</cyan>\
 - {message}</level>\
'
log = log.bind(rank=1, node=platform.node())
log = log.patch(lambda record: record['extra'].update(mempc=psutil.virtual_memory().percent))

format = '{time}|{level}\t|{message}'
log.add(sys.stderr, format=print_format, colorize=True)

log.info('Welcome hello')
log.debug('Welcome hello')
log.add(open('testlog.log', 'w+'), level='DEBUG')

pool = Pool(processes=4)

@log.catch(reraise=True)
def broken(x):
	return x/0


#y = pool.map(broken, range(4))
