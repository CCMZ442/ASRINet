import unittest
import numpy as np
from metrics import boundary_counts, erode_square, summarize


class MetricsTests(unittest.TestCase):
    def test_square_identity(self):
        m = np.zeros((200, 200), dtype=np.uint8); m[40:160, 40:160] = 1
        r = boundary_counts(m, m)
        self.assertEqual(r['boundary_radius'], 6)
        self.assertEqual(r['boundary_union'], 120**2-108**2)
        self.assertEqual(r['boundary_intersection'], r['boundary_union'])

    def test_frame(self):
        m = np.ones((200, 200), dtype=np.uint8)
        self.assertEqual(boundary_counts(m, m)['boundary_union'], 200**2-188**2)

    def test_empty(self):
        m = np.zeros((20, 20), dtype=np.uint8)
        self.assertEqual(boundary_counts(m, m)['boundary_union'], 0)

    def test_disjoint(self):
        p = np.zeros((20, 20), dtype=np.uint8); p[1:4, 1:4] = 1
        g = np.zeros_like(p); g[12:15, 12:15] = 1
        self.assertEqual(boundary_counts(p, g)['boundary_intersection'], 0)
        self.assertEqual(boundary_counts(p, g)['boundary_union'], 16)

    def test_ignore_band(self):
        p = np.ones((20, 20), dtype=np.uint8); g = p.copy(); g[10, 10] = 255
        self.assertEqual(boundary_counts(p, g)['boundary_valid_pixels'], 400-9)

    def test_thin(self):
        p = np.zeros((20, 20), dtype=np.uint8); p[:, 10] = 1
        self.assertEqual(boundary_counts(p, p)['boundary_union'], 20)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            boundary_counts(np.full((20,20), 2), np.zeros((20,20)))

    def test_pooling(self):
        r = dict(class_index=0, tp=3, fp=1, fn=2, tn=4,
                 boundary_intersection=2, boundary_union=4)
        s = summarize([r, r], ['a'])
        self.assertEqual(s['class_mean_iou'], .5)
        self.assertEqual(s['boundary_mean_iou'], .5)
        self.assertEqual(s['precision'], .75)

    def test_scipy_reference(self):
        from scipy.ndimage import binary_erosion
        rng = np.random.default_rng(314)
        for radius in [1, 2, 6]:
            m = rng.uniform(size=(21,23)) > .15
            np.testing.assert_array_equal(erode_square(m, radius),
                binary_erosion(m, structure=np.ones((3,3)), iterations=radius, border_value=0))


if __name__ == '__main__':
    unittest.main()
