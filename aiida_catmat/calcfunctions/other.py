"""Other utilities"""

import collections

from aiida.orm import Dict
from aiida.engine import calcfunction


def dict_merge(dct: dict, merge_dct: dict) -> None:
    """Taken from https://gist.github.com/angstwad/bf22d1822c38a92ec0a9
    Recursive dict merge. Inspired by ``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.

    Args:
        dct (dict): dict onto which the merge is executed
        merge_dct (dict): dict merged into dict
    """
    for k in merge_dct.keys():
        if (k in dct and isinstance(dct[k], dict) and isinstance(merge_dct[k], collections.Mapping)):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]


@calcfunction
def aiida_dict_merge(to_dict: Dict, from_dict: Dict) -> Dict:
    """Merges two `AiiDA` dictionaries.

    Args:
        to_dict (Dict): ``Dict`` onto which the merge is executed
        from_dict (Dict): ``Dict`` which will be merged into the input.

    Returns:
        Dict: The resulting ``Dict``
    """
    to_dict = to_dict.get_dict()

    if isinstance(from_dict, Dict):
        from_dict = from_dict.get_dict()

    dict_merge(to_dict, from_dict)

    return Dict(dict=to_dict)


#EOF
