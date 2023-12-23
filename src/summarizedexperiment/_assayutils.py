import itertools

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def merge_assays(assays):
    _all_keys = [list(x.keys() for x in assays)]

    _set = set()
    for k_idx in range(len(_all_keys)):
        kx = _all_keys[k_idx]
        for ky in kx:
            if ky in _set:
                ky = f"{ky}_{k_idx}"

            _set.add(ky)

    _new_all_keys = list(_set)

    _all_assays = [list(x.values()) for x in assays]
    _all_assays = list(itertools.chain.from_iterable(_all_assays))
    return dict(zip(_new_all_keys, _all_assays))
