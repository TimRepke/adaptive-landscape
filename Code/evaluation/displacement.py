import numpy as np
import logging
from evaluation import _read_file

logger = logging.getLogger('displacement')


def calculate_displacement_score(base_files):
    base_files.sort()
    prev = _read_file(base_files[0])
    for file in base_files[1:]:
        curr = _read_file(file)
        logger.debug(f'Loaded file: {file}')
        displacement = np.linalg.norm((prev[:, [0, 1]] / np.linalg.norm(prev[:, [0, 1]])) -
                                      (curr[:len(prev), [0, 1]] / np.linalg.norm(curr[:, [0, 1]])), axis=1)
        logger.info(f'Number of compared points: {displacement.shape[0]}')
        logger.debug(displacement)

        logger.info(f'Displacement per digit: ' +
                    str({digit: np.mean(displacement[prev[:, 2] == digit]) for digit in range(10)}))

        logger.info(f'Total displacement: {displacement.mean():.7f}')
        prev = curr


# TODO copy from thesne (loss function)
def movement_penalty(Ys, N):
    penalties = []
    for t in range(len(Ys) - 1):
        penalties.append(T.sum((Ys[t] - Ys[t + 1]) ** 2))

    return T.sum(penalties) / (2 * N)