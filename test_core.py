"""Small algorithm tests use synthetic arrays, not paper results or patient data."""
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
import io_utils as R
import linear as L
import metrics
import pipeline as P
import predict
import train as U


class CoreTests(unittest.TestCase):
    def test_final_training_modes(self):
        U.DEV = torch.device('cpu')
        U.self_test()

    def test_breslow_gradient_and_training_isolation(self):
        rng = np.random.default_rng(22)
        n, g = 40, 200
        x = rng.normal(size=(n, 1, g))
        d = dict(cancer='unified', clinical=rng.normal(size=(n, 3)),
                 prior=np.linspace(.1, 1., g), time=np.repeat(np.arange(1., 21.), 2),
                 event=np.tile([0., 1.], 20))
        fit, held = np.arange(30), np.arange(30, 40)
        state, off, _ = P.fit_linear(d, x, fit, [held], (1, 20, 1., .5, .3, .1))
        changed, other_x = copy.deepcopy(d), x.copy()
        other_x[held] += 100
        changed['clinical'][held] += 100
        changed['time'][held] += 100
        changed['event'][held] = 1 - changed['event'][held]
        other, other_off, _ = P.fit_linear(changed, other_x, fit, [held], (1, 20, 1., .5, .3, .1))
        np.testing.assert_array_equal(state['coef'], other['coef'])
        np.testing.assert_array_equal(off, other_off)
        r = rng.normal(size=n)
        cox = L.CoxData(np.eye(n), d['time'], d['event'])
        numeric = [(cox.loss(r + v) - cox.loss(r - v)) / 2e-6 for v in np.eye(n) * 1e-6]
        np.testing.assert_allclose(cox.loss_grad(r)[1], numeric, atol=1e-8)

    def test_auc_pairs_ties_and_undefined_horizons(self):
        r = np.array([1., 1., 0., -1.])
        t, e = np.array([5., 20., 70., 150.]), np.array([1, 1, 0, 0])
        got = metrics.horizon_auc(r, t, e)
        self.assertEqual([x['pairs'] for x in got], [3, 4, 2])
        self.assertAlmostEqual(metrics.pooled_auc(r, t, e), 8.5 / 9)
        self.assertIsNone(metrics.pooled_auc(r, t, np.zeros(4)))
        self.assertEqual(metrics.summary([1., 3., None]), dict(n_defined=2, mean=2., sample_sd=np.sqrt(2)))

    def test_zero_reference_ig_signed_and_complete(self):
        import interpret as I
        rng = np.random.default_rng(9)
        nets = [dict(w1=rng.normal(size=(5, 8)).astype(np.float32) * .1, b1=np.full(5, .1),
                     w2=rng.normal(size=(3, 5)).astype(np.float32) * .1, b2=np.full(3, .1),
                     w3=rng.normal(size=(1, 3)).astype(np.float32) * .1,
                     risk_mean=np.array(.2), risk_sd=np.array(.8)) for _ in range(5)]
        weights = [{k: v.detach().numpy().copy() for k, v in U.Network(w).state_dict().items()} for w in nets]
        z = rng.normal(size=(4, 8)).astype(np.float32)
        ig, native, reference, audit = I.neural_ig(z, nets, dict(gate=-.3, weights=weights))
        np.testing.assert_allclose(ig.sum(1), native - reference, atol=1e-4, rtol=1e-3)
        self.assertEqual(len(audit), 5)
        self.assertTrue(np.ptp(ig) > 0 and np.all(reference == reference[0]))
        zero = I.neural_ig(z, [], None)
        np.testing.assert_array_equal(zero[0], np.zeros_like(z))

    def test_source_fitting(self):
        import pretrain19 as B
        import pretrain19b as A
        B.DEV = torch.device('cpu')
        rng = np.random.default_rng(3)
        x = rng.normal(size=(60, 10)).astype(np.float32)
        t, e, g = np.arange(1., 61.), np.tile([0., 1.], 30), np.repeat(['A', 'B', 'C'], 20)
        a, b = B.split(t, e, g)
        weights, info, history = A.fit('real', 0, 1e-5, x, t, e, g, a, b, steps=3, patience=10)
        self.assertEqual(len(history), 3)
        self.assertGreater(float(weights['risk_sd']), 0)
        self.assertTrue(np.isfinite(info['best_val_loss']))

    def test_checkpoint_hash_rejected_before_unpickle(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.pkl'
            path.write_bytes(b'not a trusted pickle')
            with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                predict.load_checkpoint(path, '0' * 64)

    def test_cohort_split_overlap_rejected(self):
        d = dict(time=np.arange(4), train_01=np.array([0, 1]),
                 val_01=np.array([1]), test_01=np.array([3]), x=np.zeros((4, 1, 2)))
        with self.assertRaises(AssertionError):
            P.split_data(d, 1)


if __name__ == '__main__':
    unittest.main()
