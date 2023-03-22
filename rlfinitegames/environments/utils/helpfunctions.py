
""" helper functions for environments"""
from functools import reduce

def rgetattr(obj, attr, *args):
    """get aatrbiutes from an object recursively"""
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return reduce(_getattr, [obj] + attr.split('.'))
