"""
Numba JIT 加速工具模块。

提供可选的 JIT 编译加速。无 Numba 环境时自动降级为纯 Python，不影响功能。

用法：
    from src.lattice_reduction._native.jit_compat import njit, jit_available

    @njit(cache=True)
    def hot_function(...):
        ...

cache=True: 编译结果缓存到磁盘（~/.cache/numba/），后续启动无需重新编译。
"""

try:
    from numba import njit as _njit
    jit_available = True

    def njit(func=None, **kwargs):
        """Numba njit 装饰器，支持 @njit 或 @njit(cache=True)。"""
        if func is not None:
            return _njit(func, **kwargs)
        return lambda f: _njit(f, **kwargs)

except ImportError:
    jit_available = False

    def njit(func=None, **kwargs):
        """无 Numba 时的降级装饰器。"""
        if func is not None:
            return func
        return lambda f: f
