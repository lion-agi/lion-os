import logging

def evaluate(function, expected_output, *args, **kwargs):
    """
    Evaluate the output of a given function or operation to determine its quality or correctness.

    Args:
        function (callable): The function to evaluate.
        expected_output (Any): The expected output of the function.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        dict: A dictionary containing the evaluation result and any discrepancies.
    """
    try:
        actual_output = function(*args, **kwargs)
        result = {
            "function": function.__name__,
            "expected_output": expected_output,
            "actual_output": actual_output,
            "is_correct": actual_output == expected_output,
            "discrepancies": None if actual_output == expected_output else {
                "expected": expected_output,
                "actual": actual_output
            }
        }
        logging.info(f"Evaluation result: {result}")
        return result
    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        return {
            "function": function.__name__,
            "expected_output": expected_output,
            "actual_output": None,
            "is_correct": False,
            "discrepancies": str(e)
        }
