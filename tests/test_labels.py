from copra_grading.labels import derive_class_label


def test_below_lower_threshold_is_class_1():
    assert derive_class_label(5.9) == "1"


def test_exactly_at_lower_threshold_is_class_1():
    assert derive_class_label(6.0) == "1"


def test_just_above_lower_threshold_is_class_2():
    assert derive_class_label(6.1) == "2"


def test_mid_range_is_class_2():
    assert derive_class_label(10.0) == "2"


def test_just_below_upper_threshold_is_class_2():
    assert derive_class_label(13.9) == "2"


def test_exactly_at_upper_threshold_is_class_3():
    assert derive_class_label(14.0) == "3"


def test_above_upper_threshold_is_class_3():
    assert derive_class_label(20.0) == "3"
