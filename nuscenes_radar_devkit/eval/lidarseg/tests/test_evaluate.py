import os
import unittest

import numpy as np

from nuscenes_radar_devkit.nuscenes import NuScenes
from nuscenes_radar_devkit.eval.lidarseg.evaluate import LidarSegEval
from nuscenes_radar_devkit.eval.lidarseg.utils import ConfusionMatrix


class TestLidarSegEval(unittest.TestCase):

    def setUp(self):
        # TODO remove!
        os.environ['NUSCENES'] = '/data/sets/nuscenes'

        assert 'NUSCENES' in os.environ, 'Set NUSCENES env. variable to enable tests.'
        self.nusc = NuScenes(version='v1.0-mini', dataroot=os.environ['NUSCENES'], verbose=False)

        self.result_path = os.environ['NUSCENES']
        self.evaluator = LidarSegEval(self.nusc, self.result_path, eval_set='mini_val', verbose=False)
        self.cm = ConfusionMatrix(num_classes=3, ignore_idx=0)

    def test_get_confusion_matrix(self):
        """

        """
        test_gtru = np.array([0, 0, 0, 2, 2, 2])
        test_pred = np.array([2, 0, 0, 2, 2, 2])

        """
        empty_cm = np.zeros((self.evaluator.num_classes, self.evaluator.num_classes))
        correct_cm = np.insert(empty_cm, [0,], [[2, 0, 1],
                                                  [0, 0, 0],
                                                  [0, 0, 3]], axis=1)
        print(correct_cm)
        """
        test_cm = self.evaluator._get_confusion_matrix(test_gtru, test_pred)
        np.testing.assert_equal(test_cm[:3, :3], [[2, 0, 1],
                                                  [0, 0, 0],
                                                  [0, 0, 3]])

        # TODO check that error is thrown if class index is out of bounds

    def test_get_confusion_matrix_invalid_class(self):
        """
        Tests that an error is thrown if an input contains class indices which are not within the range of valid class
        indices (i.e. 0 to N, where N is the number of classes).
        """
        test_gtru = np.array([0, 0, 0, 2, 2, 2])
        test_pred = np.array([2, 0, 0, 2, 2, 20])

        self.assertRaises(AssertionError, self.evaluator._get_confusion_matrix, test_gtru, test_pred)


if __name__ == '__main__':
    test = TestLidarSegEval()
    test.setUp()
    test.test_get_confusion_matrix()
    test.test_get_confusion_matrix_invalid_class()
