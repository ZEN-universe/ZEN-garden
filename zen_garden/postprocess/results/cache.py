import functools


class ConditionalCache:
    """
    Decorator to conditionally cache method results based on an instance flag.
    If the flag is True, the method results are cached using functools.cache.
    Otherwise, the method is called normally without caching.

    ## Usage
    ```
    class MyClass:
        def __init__(self, enable_cache: bool):
            self.enable_cache = enable_cache

        @ConditionalCache('enable_cache')
        def my_method(self, ...):
            # method implementation
            pass
    ```
    """

    def __init__(self, flag_name: str):
        self.flag_name = flag_name

    def __call__(self, func):
        self.func = func
        functools.update_wrapper(self, func)
        return self

    def __get__(self, instance, owner):
        if instance is None:
            return self

        # Check if caching is enabled for this instance and get the bound method
        enable_cache = getattr(instance, self.flag_name, False)
        bound_func = self.func.__get__(instance, owner)

        # If caching is disabled, return the original bound method
        if not enable_cache:
            return bound_func

        # Check if the cached version already exists
        cache_attr = f"__cached_{self.func.__name__}"
        cached_func = getattr(instance, cache_attr, None)
        if cached_func is not None:
            return cached_func

        # Create and store the cached version of the method
        cached_func = functools.cache(bound_func)
        setattr(instance, cache_attr, cached_func)
        return cached_func
