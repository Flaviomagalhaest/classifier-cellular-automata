from src.helpers import distance_two_points


def test_distance_two_points_correctly():
    assert distance_two_points((2, 2), (1, 2)) == 1
    assert distance_two_points((2, 2), (1, 3)) == 1.4142
    assert distance_two_points((2, 2), (2, 3)) == 1
    assert distance_two_points((2, 2), (2, 0)) == 2
    assert distance_two_points((2, 2), (1, 4)) == 2.2361
    assert distance_two_points((2, 2), (2, 4)) == 2
    assert distance_two_points((2, 2), (4, 4)) == 2.8284
    assert distance_two_points((2, 2), (0, 4)) == 2.8284
