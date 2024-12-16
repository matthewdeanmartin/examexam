from examexam.take_exam import is_valid


def test_is_valid():

    assert is_valid("1", 5, 1)[0]
    assert is_valid("1,2", 5, 2)[0]
    assert is_valid("1,2,3,4", 5, 4)[0]

    assert not is_valid("", 5, 0)[0]
    assert not is_valid("", 5, 3)[0]
    # too big
    assert not is_valid("10", 5, 3)[0]
    # too few
    assert not is_valid("1", 5, 3)[0]
    # several out of range
    assert not is_valid("11,11,11", 5, 3)[0]
