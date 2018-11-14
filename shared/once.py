import pickle
import pathlib
cache_dir = 'cache'
def once(func, *args, **kwargs):
    args_as_string = ', '.join(str(e) for e in args)
    key = f"{func.__name__}({args_as_string})"
    filename = cache_dir + '/' + key.replace('/', '_') + '.once'
    try:
        file = open(filename, 'rb')
        print("found " + filename + " in cache")
        value = pickle.load(file)
        file.close()
        return value
    except FileNotFoundError:
        value = func(*args, **kwargs)
        pathlib.Path(cache_dir).mkdir(parents=False, exist_ok=True)
        file = open(filename, 'wb')
        pickle.dump(value, file)
        print("write " + filename + " to cache")
        return value