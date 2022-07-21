"""from aiida-quantumespresso plugin """

from aiida.common import AttributeDict

from aiida.orm import Dict
from aiida.engine.processes import PortNamespace


def wrap_bare_dict_inputs(port_namespace, inputs):
    """Wrap bare dictionaries in `inputs` in a `Dict` node if dictated by the corresponding port in given namespace.
    :param port_namespace: a `PortNamespace`
    :param inputs: a dictionary of inputs intended for submission of the process
    :return: a dictionary with all bare dictionaries wrapped in `Dict` if dictated by the port namespace
    """

    wrapped = {}

    for key, value in inputs.items():

        if key not in port_namespace:
            wrapped[key] = value
            continue

        port = port_namespace[key]

        if isinstance(port, PortNamespace):
            wrapped[key] = wrap_bare_dict_inputs(port, value)
        elif port.valid_type == Dict and isinstance(value, dict):
            wrapped[key] = Dict(dict=value)
        else:
            wrapped[key] = value

    return wrapped


def prepare_process_inputs(process, inputs):
    """Prepare the inputs for submission for the given process, according to its spec.
    That is to say that when an input is found in the inputs that corresponds to an input port in the spec of the
    process that expects a `Dict`, yet the value in the inputs is a plain dictionary, the value will be wrapped in by
    the `Dict` class to create a valid input.
    :param process: sub class of `Process` for which to prepare the inputs dictionary
    :param inputs: a dictionary of inputs intended for submission of the process
    :return: a dictionary with all bare dictionaries wrapped in `Dict` if dictated by the process spec
    """
    prepared_inputs = wrap_bare_dict_inputs(process.spec().inputs, inputs)
    return AttributeDict(prepared_inputs)


# EOF
