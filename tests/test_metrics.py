import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from talk2dino_jittor.metrics import ConfusionMatrix

class TestMetrics(unittest.TestCase):
    def test_hand_counted_confusion_and_ignore(self):
        m = ConfusionMatrix(3)
        m.update(np.array([0,1,1,2,0]), np.array([0,0,1,2,255]))
        self.assertEqual(m.matrix.tolist(), [[1,1,0],[0,1,0],[0,0,1]])
        np.testing.assert_allclose(m.compute()["iou"], [0.5,0.5,1.0])
        self.assertAlmostEqual(m.compute()["miou"], 2/3)

    def test_absent_class_is_ignored_in_mean(self):
        m=ConfusionMatrix(3)
        m.update(np.array([0,1]),np.array([0,1]))
        self.assertAlmostEqual(m.compute()["miou"],1.0)
        self.assertTrue(np.isnan(m.compute()["iou"][2]))

    def test_invalid_predictions_raise(self):
        with self.assertRaises(ValueError):
            ConfusionMatrix(2).update(np.array([2]), np.array([0]))

    def test_accumulation_matches_combined_batch(self):
        pred=np.array([0,1,2,1]); gt=np.array([0,2,2,1])
        a,b=ConfusionMatrix(3),ConfusionMatrix(3)
        a.update(pred[:2],gt[:2]); a.update(pred[2:],gt[2:])
        b.update(pred,gt)
        np.testing.assert_array_equal(a.matrix,b.matrix)

if __name__=="__main__":
    unittest.main()
