class ProgressBar:
    """
    A progress bar wrapper that acts like tqdm but supports custom callbacks.
    """
    _callback = None

    @classmethod
    def set_callback(cls, callback):
        """
        Set a callback function for progress updates.
        The callback should have the signature:
            callback(current: int, total: int, desc: str)
        """
        cls._callback = callback

    @classmethod
    def get_callback(cls):
        return cls._callback

    def __init__(self, iterable=None, desc="", total=None, **kwargs):
        self.iterable = iterable
        self.desc = desc
        self.total = total if total is not None else (len(iterable) if hasattr(iterable, "__len__") else None)
        self.kwargs = kwargs
        
        callback = self.__class__._callback
        self.use_tqdm = (callback is None)
        if self.use_tqdm:
            from tqdm import tqdm
            self.tqdm_pbar = tqdm(iterable, desc=desc, total=total, **kwargs)
        else:
            self.tqdm_pbar = None
            self.current = 0
            if callback is not None:
                callback(0, self.total, self.desc)

    def __iter__(self):
        if self.use_tqdm:
            yield from self.tqdm_pbar
        else:
            if self.iterable is not None:
                for item in self.iterable:
                    yield item
                    self.update(1)

    def update(self, n=1):
        if self.use_tqdm:
            if self.tqdm_pbar is not None:
                self.tqdm_pbar.update(n)
        else:
            self.current += n
            callback = self.__class__._callback
            if callback is not None:
                callback(self.current, self.total, self.desc)

    def close(self):
        if self.use_tqdm and self.tqdm_pbar is not None:
            self.tqdm_pbar.close()

    def __enter__(self):
        if self.use_tqdm and self.tqdm_pbar is not None:
            self.tqdm_pbar.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.use_tqdm and self.tqdm_pbar is not None:
            return self.tqdm_pbar.__exit__(exc_type, exc_val, exc_tb)
