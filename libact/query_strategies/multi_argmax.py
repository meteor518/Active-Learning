import numpy as np

def multi_argmax(values, n_instances=1):
    """
    Selects the indices of the n_instances highest values.

    :param values:
        Contains the values to be selected from.
    :type values:
        numpy.ndarray of shape = (n_samples, 1)

    :param n_instances:
        Specifies how many indices to return.
    :type n_instances:
        int

    :returns:
      - **max_idx** *(numpy.ndarray of shape = (n_samples, 1))* --
        Contains the indices of the n_instances largest values.

    """
    assert n_instances <= len(values), 'n_instances must be less or equal than the size of utility'

    max_idx = np.argpartition(-values, n_instances - 1, axis=0)[:n_instances]
    return max_idx