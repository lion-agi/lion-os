import logging
import time

def optimize(function, param_grid, *args, **kwargs):
    """
    Optimize a given function or process to improve its performance or efficiency.

    Args:
        function (callable): The function to optimize.
        param_grid (dict): A dictionary where keys are parameter names and values are lists of parameter values to try.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        dict: A dictionary containing the optimal parameter values and their corresponding performance measurements.
    """
    best_params = None
    best_performance = float('inf')
    performance_log = []

    for params in param_grid:
        start_time = time.time()
        result = function(*args, **{**kwargs, **params})
        end_time = time.time()
        performance = end_time - start_time

        performance_log.append({
            "params": params,
            "performance": performance
        })

        if performance < best_performance:
            best_performance = performance
            best_params = params

    logging.info(f"Optimization result: Best params: {best_params}, Best performance: {best_performance}")
    return {
        "best_params": best_params,
        "best_performance": best_performance,
        "performance_log": performance_log
    }
