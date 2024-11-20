# coding = utf-8

import os
import unittest

import numpy as np
import pandas as pd

from PyMatterSim.neighbors.calculate_neighbors import Nnearests
from PyMatterSim.reader.dump_reader import DumpReader
from PyMatterSim.reader.reader_utils import DumpFileType
from PyMatterSim.static.vector import (divergence_curl, local_vector_alignment,
                                       participation_ratio, phase_quotient,
                                       vector_decomposition_sq, vector_fft_corr,
                                       vibrability)
from PyMatterSim.utils.logging import get_logger_handle
from PyMatterSim.utils.wavevector import choosewavevector

logger = get_logger_handle(__name__)

READ_TEST_FILE_PATH = "tests/sample_test_data"


class TestVector(unittest.TestCase):
    """
    Test class for Structure factor
    """
    # ,,phase_quotient,divergence_curl,vibrability,vector_fft_corr

    def setUp(self) -> None:
        super().setUp()
        self.test_file_2d = f"{READ_TEST_FILE_PATH}/2d/2ddump.s.v.atom"
        self.test_file_3d = f"{READ_TEST_FILE_PATH}/3d/3ddump.s.v.atom"

        self.input_vp = DumpReader(self.test_file_2d, ndim=2)
        self.input_vp.read_onefile()
        self.input_v = DumpReader(
            self.test_file_2d,
            ndim=2,
            filetype=DumpFileType.LAMMPSVECTOR,
            columnsids=[5, 6]
        )
        self.input_v.read_onefile()

    def test_participation_ratio(self) -> None:
        """
        Test participation_ratio
        """
        logger.info(f"Starting test using {self.test_file_2d}...")

        v1 = self.input_v.snapshots.snapshots[0].positions
        v2 = self.input_v.snapshots.snapshots[9].positions
        pr1 = participation_ratio(v1)
        pr2 = participation_ratio(v2)
        np.testing.assert_almost_equal(pr1, 0.521491080361457)
        np.testing.assert_almost_equal(pr2, 0.4829625233406623)
        logger.info(
            f"Finishing test gyration_tensor using {self.test_file_2d}...")

    def test_local_vector_alignment(self) -> None:
        """
        Test local_vector_alignment
        """
        logger.info(f"Starting test using {self.test_file_2d}...")

        Nnearests(self.input_vp.snapshots, N=6, ppp=np.array(
            [1, 1]), fnfile='neighborlist.dat')
        neighborfile = 'neighborlist.dat'
        v1 = self.input_v.snapshots.snapshots[0].positions
        v2 = self.input_v.snapshots.snapshots[9].positions
        lva1 = local_vector_alignment(v1, neighborfile)
        lva2 = local_vector_alignment(v2, neighborfile)

        np.testing.assert_almost_equal(
            lva1[:10],
            np.array([
                -0.11154581, -0.01004038, -0.01522551, 0.09333871, -0.10149098,
                0.25865722, -0.05835916, -0.28434863, -0.01074069, -0.79158841])
        )
        np.testing.assert_almost_equal(
            lva2[:10],
            np.array([
                0.00992724, -0.39541864, -0.06977997, 0.02699935, -0.05977384,
                -0.06719096, 0.06022695, 0.0664444, -0.1649534, -0.27471856])
        )

        os.remove(neighborfile)
        logger.info(
            f"Finishing test gyration_tensor using {self.test_file_2d}...")

    def test_phase_quotient(self) -> None:
        """
        Test phase_quotient
        """
        logger.info(f"Starting test using {self.test_file_2d}...")

        Nnearests(self.input_vp.snapshots, N=6, ppp=np.array(
            [1, 1]), fnfile='neighborlist.dat')
        neighborfile = 'neighborlist.dat'
        v1 = self.input_v.snapshots.snapshots[0].positions
        v2 = self.input_v.snapshots.snapshots[9].positions
        pq1 = phase_quotient(v1, neighborfile)
        pq2 = phase_quotient(v2, neighborfile)

        np.testing.assert_almost_equal(pq1, -0.05594896698423584)
        np.testing.assert_almost_equal(pq2, -0.02978854969507169)

        os.remove(neighborfile)
        logger.info(
            f"Finishing test gyration_tensor using {self.test_file_2d}...")

    def test_divergence_curl_2d(self) -> None:
        """
        Test divergence_curl for 2d
        """
        logger.info(f"Starting test using {self.test_file_2d}...")
        ppp = np.array([1, 1])
        Nnearests(
            self.input_vp.snapshots,
            N=6,
            ppp=ppp,
            fnfile='neighborlist.dat')
        neighborfile = 'neighborlist.dat'

        v1 = self.input_v.snapshots.snapshots[0].positions
        divergence1 = divergence_curl(
            self.input_vp.snapshots.snapshots[0], v1, ppp, neighborfile)
        np.testing.assert_almost_equal(
            divergence1[:10],
            np.array([
                -0.29712397, -0.16234507, -0.0749001, 0.35241748, -0.082197,
                -0.01734587, 0.34023927, -0.22716885, 0.28200287, -0.14452179
            ])
        )
        os.remove(neighborfile)
        logger.info(
            f"Finishing test gyration_tensor using {self.test_file_2d}...")

    def test_divergence_curl_3d(self) -> None:
        """
        Test divergence_curl for 3d
        """
        logger.info(f"Starting test using {self.test_file_3d}...")
        self.input_vp = DumpReader(self.test_file_3d, ndim=3)
        self.input_vp.read_onefile()

        self.input_v = DumpReader(
            self.test_file_3d,
            ndim=3,
            filetype=DumpFileType.LAMMPSVECTOR,
            columnsids=[6, 7, 8]
        )
        self.input_v.read_onefile()

        ppp = np.array([1, 1, 1])
        Nnearests(
            self.input_vp.snapshots,
            N=12,
            ppp=ppp,
            fnfile='neighborlist.dat')
        neighborfile = 'neighborlist.dat'
        v1 = self.input_v.snapshots.snapshots[0].positions

        divergence, curl = divergence_curl(
            self.input_vp.snapshots.snapshots[0], v1, ppp, neighborfile)

        np.testing.assert_almost_equal(
            divergence[:10],
            np.array([
                0.13229511, 0.06307015, -0.2782819, -0.00341545, -0.40276813,
                -0.18070488, 0.38035838, -0.27656106, -0.13100627, -0.2772078
            ])
        )
        np.testing.assert_almost_equal(
            curl[:10],
            np.array([
                [0.1027198, 0.05574753, -0.00101901],
                [0.04840527, -0.20696524, 0.08992696],
                [0.26607367, 0.33976341, 0.21631076],
                [-0.27889849, 0.01681247, 0.1275699],
                [0.05798374, -0.04301222, 0.11948525],
                [-0.09619516, -0.06255371, 0.03827239],
                [-0.0609074, -0.08368192, -0.14454036],
                [0.02275234, -0.11861828, -0.46957482],
                [0.20737365, 0.17262808, 0.11304505],
                [-0.15270774, -0.00511331, -0.0221363]])
        )

        os.remove(neighborfile)
        logger.info(
            f"Finishing test gyration_tensor using {self.test_file_3d}...")

    def test_vibrability(self) -> None:
        """
        Test vibrability
        """
        logger.info("Starting test vibrability...")

        a = np.array([[2, 3, 4, 1, 3, 3, 2, 2, 2, 2],
                      [3, 2, 2, 2, 3, 3, 3, 2, 3, 3],
                      [2, 2, 3, 3, 1, 2, 4, 1, 4, 3],
                      [2, 1, 2, 2, 3, 1, 1, 2, 3, 4],
                      [3, 2, 3, 4, 3, 2, 2, 2, 1, 4],
                      [4, 3, 4, 1, 2, 2, 4, 2, 2, 3],
                      [3, 4, 3, 1, 1, 1, 2, 2, 3, 1],
                      [3, 1, 2, 3, 2, 4, 2, 4, 4, 1],
                      [2, 2, 2, 4, 3, 4, 3, 3, 3, 2],
                      [1, 1, 1, 1, 1, 1, 2, 1, 2, 3]])
        eigenfrequencies, eigenvectors = np.linalg.eig(a)
        eigenfrequencies = np.abs(eigenfrequencies.real)
        eigenfrequencies = np.sqrt(eigenfrequencies)
        eigenvectors = eigenvectors.real
        vb = vibrability(eigenfrequencies, eigenvectors, 10)

        np.testing.assert_almost_equal(
            vb,
            np.array([
                0.42386879, 0.2696528, 0.4320155, 0.57338379, 1.14279844,
                0.41539947, 0.36564011, 0.2999141, 0.09689469, 0.18054036
            ])
        )
        logger.info("Finishing test vibrability...")

    def test_vector_decomposition_sq(self) -> None:
        """
        Test vector_decomposition_sq
        """
        logger.info(f"Starting test using {self.test_file_2d}...")

        v1 = self.input_v.snapshots.snapshots[0].positions
        _, ave_sqresults1 = vector_decomposition_sq(
            self.input_vp.snapshots.snapshots[0],
            choosewavevector(2, 10),
            v1
        )

        v2 = self.input_v.snapshots.snapshots[9].positions
        _, ave_sqresults2 = vector_decomposition_sq(
            self.input_vp.snapshots.snapshots[9],
            choosewavevector(2, 10),
            v2
        )

        np.testing.assert_almost_equal(ave_sqresults1[['Sq_T', "Sq_L"]].values,
                                       np.array([[0.32717711, 0.10718685],
                                                 [0.12571704, 0.58979643],
                                                 [0.45099315, 0.33494711],
                                                 [0.074935, 0.05214448],
                                                 [0.46892341, 0.43781952]]))
        np.testing.assert_almost_equal(ave_sqresults1["Sq"].values, np.array(
            [0.43436396, 0.71551347, 0.78594025, 0.12707948, 0.90674292]))
        np.testing.assert_almost_equal(ave_sqresults2[['Sq_T', "Sq_L"]].values,
                                       np.array([[0.22552786, 1.52405984],
                                                 [0.4014555, 0.73126886],
                                                 [0.22552013, 0.44160593],
                                                 [0.37419256, 0.03427435],
                                                 [0.45553099, 0.41599951]]))
        np.testing.assert_almost_equal(ave_sqresults2["Sq"].values, np.array(
            [1.74958771, 1.13272437, 0.66712606, 0.4084669, 0.8715305]))
        logger.info(f"Finishing test vector_decomposition_sq using {self.test_file_2d}...")

    def test_vector_fft_corr(self) -> None:
        """
        Test vector_fft_corr
        """
        logger.info(f"Starting test using {self.test_file_2d}...")
        qvector = choosewavevector(2, 27, False)

        v = []
        for n in range(self.input_v.snapshots.nsnapshots):
            temp = self.input_v.snapshots.snapshots[n].positions
            v.append(temp)
        v = np.array(v)

        alldata = vector_fft_corr(
            self.input_vp.snapshots,
            qvector,
            v,
            outputfile="test")
        spectra = pd.read_csv("test.spectra.csv")
        T_fft = alldata["T_FFT"].values  # np.load("test.T_FFT.npy")
        L_fft = alldata["L_FFT"].values  # np.load("test.L_FFT.npy")
        np.testing.assert_almost_equal(spectra[["Sq_T", "Sq_L"]].values,
                                       np.array([[0.47630587, 0.60613897],
                                                [0.36717133, 0.56802924],
                                                [0.35249389, 0.28962447],
                                                [0.45956127, 0.41762113],
                                                [0.4186975, 0.42035541],
                                                [0.1724058, 0.43946091],
                                                [0.34672458, 0.4990829],
                                                [0.59251935, 0.23891557],
                                                [0.28245196, 0.38044027],
                                                [0.53871987, 0.40569833],
                                                [0.30052022, 0.50367928],
                                                [0.36510082, 0.4478105],
                                                [0.43244991, 0.51579779],
                                                [0.37179696, 0.29852751]]))
        np.testing.assert_almost_equal(
            T_fft[0, 3:],
            np.array([
                1., 0.33282675, 0.10380247, 0.06949692, 0.16854733,
                0.46965734, 0.01265494, -0.18949574, 0.52572456, 0.17707337
            ])
        )
        np.testing.assert_almost_equal(
            L_fft[0, 3:],
            np.array([
                1., -0.08931861, 0.18748259, 0.14045109, 0.20976937,
                -0.36043121, 0.04820545, -0.19739207, -1.01346368, 0.06020561
            ])
        )

        os.remove("test.spectra.csv")
        os.remove("test.T_FFT.npy")
        os.remove("test.L_FFT.npy")
        os.remove("test.FFT.npy")
        logger.info(f"Finishing test vector_decomposition_sq using {self.test_file_2d}...")
