#! encoding="utf-8"
from .tsit5 import Tsit5Solver
from .dopri5 import Dopri5Solver
from .fixed_grid import Euler, Midpoint, RK4
from .fixed_adams import AdamsBashforth, AdamsBashforthMoulton
from .adams import VariableCoefficientAdamsBashforth
from .misc import _check_inputs

SOLVERS = {
    'explicit_adams': AdamsBashforth,
    'fixed_adams': AdamsBashforthMoulton,
    'adams': VariableCoefficientAdamsBashforth,
    'tsit5': Tsit5Solver,
    'dopri5': Dopri5Solver,
    'euler': Euler,
    'midpoint': Midpoint,
    'rk4': RK4,
}


def odeint(func, y0, t, rtol=1e-7, atol=1e-9, method=None, options=None):
    """对一个ODE系统进行积分

    解决了一阶ODE非刚性系统的初值问题:
        ```
        dy/dt = func(t, y), y(t[0]) = y0
        ```
    其中y是任意形状的张量。

    输出的dtypes和数值精度基于输入“y0”的dtypes。

    Args:
        func: 将持有状态“y”和标量张量`t`的张量映射到关于时间（time）的状态导数的张量的函数。

        y0: N-D 张量在时间点`t [0]`给出的`y`的起始值。dtype可以是浮点数或复数。
        
        t: 1-D Tensor，持有一系列时间点来解决“y”。初始时间点应该是此序列的第一个元素，并且每个时
        间必须大于前一个时间。可能有任何浮点dtype。转换为float64 dtype的Tensor。

        rtol: 可选的float64 Tensor，指定相对误差的上限，`y`的每个元素。

        atol: 可选的float64 Tensor，指定绝对误差的上限，`y`的每个元素。
        method: 字符串，指定积分的方法。
        options: 可选字典，为指定的积分方法配置选项。只有在明确设置`method`时才能提供。
        name: 可选，该步操作的名称.

    Returns:
        y: 张量，第一维对应不同的时间点。包含了`t`中每个所需时间点的y的求解值，其中，第一维的第一个元素是初始值`y0`。


    Raises:
        ValueError: 非法的`method`.
        TypeError: 无`method`有`options`, 或者`t`/`y0`的dtype非法。
    """

    tensor_input, func, y0, t = _check_inputs(func, y0, t)

    if options is None:
        options = {}
    elif method is None:
        raise ValueError('未指定`method`，`options`无效')

    if method is None:
        method = 'dopri5'

    solver = SOLVERS[method](func, y0, rtol=rtol, atol=atol, **options)
    solution = solver.integrate(t)

    if tensor_input:
        solution = solution[0]
    return solution
