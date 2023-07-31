from functools import wraps
from timeit import default_timer


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = default_timer()
        result = f(*args, **kw)
        te = default_timer()
        print(f'{f.__name__} took: {te-ts:,.2f} sec')
        return result
    return wrap
