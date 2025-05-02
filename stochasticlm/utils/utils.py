from functools import reduce

def compose(*fns):
    def __inner__(*args):
        return reduce(lambda acc, fn: (fn(*acc) if isinstance(acc, tuple) else fn(acc)), reversed(fns), args)
    return __inner__

