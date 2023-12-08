# coding = utf-8

import unittest
import numpy as np
from reader.dump_reader import DumpReader
from static.tetrahedral import q_tetra
from utils.logging import get_logger_handle

logger = get_logger_handle(__name__)

READ_TEST_FILE_PATH = "tests/sample_test_data"


class TestTetrahedral(unittest.TestCase):
    """
    Test class for tetrahedral
    """
    def setUp(self) -> None:
        super().setUp()
        self.test_file_3D = f"{READ_TEST_FILE_PATH}/test_additional_columns.dump"

    def test_tetrahedral(self) -> None:
        """
        Test tetrahedral works properly for 3D system
        """
        logger.info(f"Starting test tetrahedral using {self.test_file_3D}...")
        readdump = DumpReader(self.test_file_3D, 3)
        readdump.read_onefile()
        tetrahedral = q_tetra(readdump.snapshots)
        np.testing.assert_almost_equal(
            np.array([[0.79553053, 0.87693344, 0.80694122, 0.8021498 , 0.78445042,
            0.82128644, 0.8267357 , 0.82825242, 0.85446152, 0.78723782,
            0.91187236, 0.95982658, 0.82207   , 0.89368492, 0.92176236,
            0.81613609, 0.88312225, 0.78026667, 0.78700256, 0.9084372 ,
            0.76013874, 0.84776699, 0.92610592, 0.86596065, 0.83644922,
            0.93345373, 0.7424407 , 0.85787562, 0.84222525, 0.90899426,
            0.81283177, 0.88652565, 0.89460427, 0.8145703 , 0.83746283,
            0.81115578, 0.81614046, 0.90185306, 0.87648113, 0.91973314,
            0.81851143],
            [0.78023514, 0.78866256, 0.85469208, 0.83678135, 0.78683257,
            0.80899596, 0.85186268, 0.78381519, 0.86064959, 0.90620284,
            0.92179804, 0.80363381, 0.80844456, 0.83152024, 0.8870659 ,
            0.82157773, 0.92530334, 0.86496757, 0.82228795, 0.86166149,
            0.74768952, 0.89838833, 0.77258799, 0.94369645, 0.8946883 ,
            0.81438987, 0.82619657, 0.85071086, 0.8121586 , 0.86868307,
            0.86678466, 0.84724192, 0.80091934, 0.86722429, 0.85641036,
            0.89014217, 0.82300112, 0.84661378, 0.85579741, 0.85526542,
            0.83750587]]),
            tetrahedral[:, ::200])
        logger.info(f"Finishing test tetrahedral using {self.test_file_3D}...")
