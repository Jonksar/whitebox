"""
  --------------------------------------------------
  File Name : utils.py
  Creation Date : 26-05-2018
  Last Modified : 2019-10-20 Sun 11:26 am
  Created By : Joonatan Samuel
  --------------------------------------------------
"""

from __future__ import print_function

import itertools
import time
import sys, os

def first(the_iterable, condition = lambda x: True):
    for i in the_iterable:
        if condition(i):
            return i

def flatten(lst):
    return [item for sublist in lst for item in sublist]

def dict_merge(*args):
    return dict(itertools.chain(*[d.items() for d in args]))

def sign(x):
    return 1 if x > 0 else -1

def itemsetter(*items):
    if len(items) == 1:
        item = items[0]
        def g(obj, value):
            obj[item] = value
    else:
        def g(obj, *values):
            for item, value in zip(items, values):
                obj[item] = value
    return g

class Timer:
    def __init__(self, desc=None):
        self.t = time.time()
        self.desc = desc

    def __enter__(self, *args, **kwargs): pass
    def __exit__(self, *args, **kwargs):
        print(( "this" if self.desc is None else self.desc )+ " took %6.4f seconds" % (time.time() - self.t  ))

class Colorizer:
    @staticmethod
    def red(prt): return "\033[91m{}\033[00m" .format(prt)
    @staticmethod
    def green(prt): return "\033[92m{}\033[00m" .format(prt)
    @staticmethod
    def yellow(prt): return "\033[93m{}\033[00m" .format(prt)
    @staticmethod
    def lightPurple(prt): return "\033[94m{}\033[00m" .format(prt)
    @staticmethod
    def purple(prt): return "\033[95m{}\033[00m" .format(prt)
    @staticmethod
    def cyan(prt): return "\033[96m{}\033[00m" .format(prt)
    @staticmethod
    def lightGray(prt): return "\033[97m{}\033[00m" .format(prt)
    @staticmethod
    def black(prt): return "\033[98m{}\033[00m" .format(prt)

    @staticmethod
    def error_text(string):
        items = string.split(' ')
        for i, item in enumerate(items):
            if 'error' in item.lower():
                items[i] = Colorizer.red(items[i])
            elif 'warning' in item.lower():
                items[i] = Colorizer.yellow(items[i])
            elif 'info' in item.lower():
                items[i] = Colorizer.cyan(items[i])
        return ' '.join(items)

def eprint(*args, **kwargs):
    args = list(map(lambda item: Colorizer.error_text(item) if isinstance( item, str ) else item, list(args) ))
    print(*args, file=sys.stderr, **kwargs)

def keyboard_interrupt(func, *args, **kwards):
    def wrapped(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except KeyboardInterrupt as e:
            eprint("\nInterrupted function '%s' with %s" % ( Colorizer.lightGray( func.__name__ ) , Colorizer.green("KeyboardInterrupt")))

    return wrapped

if __name__ == '__main__':
    eprint( '[ERROR] this is a error example')
    eprint( '[WARNING] this is a warning example')
    eprint( '[INFO] this is an info example')

    @keyboard_interrupt
    def sleeper_function():
        eprint('[INFO] interrupt this with KeyboardInterrupt')
        time.sleep(25)

    sleeper_function()
