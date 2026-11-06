"""Numerical checks for training-only affine replay."""
import unittest
import numpy as np
from reevaluate_black_box_affine import affine_metrics, predict, training_data
from reevaluate_black_box_affine_test import frozen_predictions, test_score


class AffineTests(unittest.TestCase):
    def test_test_predictions_use_frozen_training_fit(self):
        from sklearn.preprocessing import StandardScaler
        y_train = np.array([-2., -1., 0., 1., 2.])
        metrics = affine_metrics(y_train, 2*y_train+5)
        p_test = np.array([9., 11.])
        fitted = frozen_predictions(p_test, metrics)
        np.testing.assert_allclose(fitted, [2., 3.])
        scaler = StandardScaler().fit(np.array([[-1.], [1.]]))
        self.assertAlmostEqual(test_score(np.array([2., 3.]), fitted, scaler), 1.)
        self.assertLess(test_score(np.array([20., 30.]), fitted, scaler), 0.)
        # Scoring arbitrary held-out targets must not change the coefficients.
        self.assertEqual(metrics['slope'], .5)
        self.assertEqual(metrics['intercept'], -2.5)

    def test_recovers_scale_offset_and_negative_slope(self):
        y = np.linspace(-3, 3, 51)
        result = affine_metrics(y, -2*y+7)
        self.assertAlmostEqual(result['slope'], -.5)
        self.assertAlmostEqual(result['intercept'], 3.5)
        self.assertAlmostEqual(result['affine_train_r2'], 1)
        self.assertLess(result['train_r2'], 0)

    def test_constant_prediction_uses_training_mean(self):
        result = affine_metrics(np.array([1., 2., 3.]), np.ones(3)*8)
        self.assertEqual(result['slope'], 0)
        self.assertEqual(result['intercept'], 2)
        self.assertAlmostEqual(result['affine_train_r2'], 0)

    def test_matches_lstsq_with_intercept(self):
        rng = np.random.RandomState(123)
        p = rng.randn(100)
        y = 1.7*p - 2 + rng.randn(100)*.4
        a, b = np.linalg.lstsq(np.column_stack([p, np.ones(100)]), y, rcond=None)[0]
        result = affine_metrics(y, p)
        self.assertAlmostEqual(result['slope'], a)
        self.assertAlmostEqual(result['intercept'], b)

    def test_tiny_prediction_scale(self):
        y = np.linspace(-1, 1, 20)
        self.assertAlmostEqual(affine_metrics(y, y*1e-100)['affine_train_r2'], 1)

    def test_expression_mapping_and_constant(self):
        X = np.array([[1., 2.], [3., 4.]])
        np.testing.assert_allclose(predict('square(x0) + sin(x1)', X), X[:, 0]**2+np.sin(X[:, 1]))
        np.testing.assert_allclose(predict('2.5', X), [2.5, 2.5])

    def test_training_scalers_do_not_see_held_out_targets(self):
        from sklearn.model_selection import train_test_split
        X = np.arange(80.).reshape(40, 2)
        y = np.arange(40.)
        task = dict(seed=10000, run_index=2, max_samples=10)
        _, test = train_test_split(np.arange(40), train_size=.75, test_size=.25, random_state=10002)
        original = training_data(X, y, task)
        y[test] += 1e6
        changed = training_data(X, y, task)
        np.testing.assert_array_equal(original[0], changed[0])
        np.testing.assert_array_equal(original[1], changed[1])


if __name__ == '__main__':
    unittest.main()
