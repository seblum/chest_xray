import glob
import shutil
import os


def timer(func):
    """Decorator to print how long a function runs."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        t_total = time.time() - start
        print(f"{func.__name__} took {round(t_total,2)}s")
        return result

    return wrapper


def data_count(classtype, filenames):
    if classtype == 0:
        return sum(map(lambda x: "NORMAL" in x, filenames))
    elif classtype == 1:
        return sum(map(lambda x: "PNEUMONIA" in x, filenames))


@timer
def move_files(filelist, dest):
    files = glob.glob(f"{dest}*")
    for f in files:
        shutil.rmtree(f)
    for f in filelist:
        part = f.split(os.path.sep)[-2]
        if not os.path.exists(f"{dest}{part}/"):
            os.makedirs(f"{dest}{part}/")
        shutil.copy(f, f"{dest}{part}/")
