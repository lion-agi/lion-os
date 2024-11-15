def test_operations_imports():
    """Test that all operations can be imported from the operations package"""
    from lion.operations import brainstorm, plan, select

    assert callable(brainstorm)
    assert callable(plan)
    assert callable(select)
