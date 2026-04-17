import pytest
import numpy as np

from services.orientation import get_orientation_vector, should_count_crossing, get_angular_error


class TestGetOrientationVector:
    def test_basic_vector(self):
        kp = np.array([[10.0, 0.0], [0.0, 0.0]])  # head right of tail
        vec = get_orientation_vector(kp)
        assert vec is not None
        np.testing.assert_allclose(vec, [1.0, 0.0], atol=1e-6)

    def test_normalized(self):
        kp = np.array([[3.0, 4.0], [0.0, 0.0]])
        vec = get_orientation_vector(kp)
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-6

    def test_none_on_degenerate(self):
        kp = np.array([[5.0, 5.0], [5.0, 5.0]])  # head == tail
        assert get_orientation_vector(kp) is None

    def test_none_on_short_array(self):
        assert get_orientation_vector(np.array([[1.0, 2.0]])) is None

    def test_none_on_none_input(self):
        assert get_orientation_vector(None) is None


class TestShouldCountCrossing:
    def test_aligned_vectors_accepted(self):
        track_dir = np.array([0.0, -1.0])   # moving up
        orient = np.array([0.0, -1.0])      # head pointing up
        assert should_count_crossing(track_dir, orient, threshold_deg=60.0) == True

    def test_perpendicular_rejected(self):
        track_dir = np.array([0.0, -1.0])   # moving up
        orient = np.array([1.0, 0.0])       # head pointing right (90°)
        assert should_count_crossing(track_dir, orient, threshold_deg=60.0) == False

    def test_opposite_rejected(self):
        track_dir = np.array([0.0, -1.0])   # moving up
        orient = np.array([0.0, 1.0])       # head pointing down (180°)
        assert should_count_crossing(track_dir, orient, threshold_deg=60.0) == False

    def test_fallback_true_on_none_orient(self):
        track_dir = np.array([0.0, -1.0])
        assert should_count_crossing(track_dir, None, threshold_deg=60.0) == True

    def test_fallback_true_on_none_track(self):
        orient = np.array([0.0, -1.0])
        assert should_count_crossing(None, orient, threshold_deg=60.0) == True

    def test_threshold_boundary(self):
        # 45° angle
        orient = np.array([1.0, -1.0]) / np.sqrt(2)
        track_dir = np.array([0.0, -1.0])
        assert should_count_crossing(track_dir, orient, threshold_deg=46.0) == True
        assert should_count_crossing(track_dir, orient, threshold_deg=44.0) == False


class TestGetAngularError:
    def test_identical_vectors(self):
        v = np.array([1.0, 0.0])
        assert get_angular_error(v, v) == pytest.approx(0.0, abs=1e-5)

    def test_opposite_vectors(self):
        # Should return min(180, 0) = 0 because of head/tail symmetry
        v1 = np.array([1.0, 0.0])
        v2 = np.array([-1.0, 0.0])
        assert get_angular_error(v1, v2) == pytest.approx(0.0, abs=1e-5)

    def test_90_degree_error(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        assert get_angular_error(v1, v2) == pytest.approx(90.0, abs=1e-5)

    def test_none_inputs(self):
        assert get_angular_error(None, np.array([1.0, 0.0])) is None
        assert get_angular_error(np.array([1.0, 0.0]), None) is None
