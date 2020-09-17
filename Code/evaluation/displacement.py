import numpy as np
import logging

logger = logging.getLogger('displacement')


def _read_file(file_name):
    with open(file_name, 'r') as f:
        return np.array([[float(dim) for dim in line.split()[:2]] for line in f])


def calculate_displacement_score(base_files):
    base_files.sort()
    prev = _read_file(base_files[0])
    for file in base_files[1:]:
        curr = _read_file(file)

        displacement = np.linalg.norm((prev / np.linalg.norm(prev)) -
                                      (curr[:len(prev)] / np.linalg.norm(curr)), axis=1)
        logger.debug(displacement.shape)
        logger.debug(displacement)
        logger.info(displacement.mean())
