# Reference absl.flags in code with the complete name (verbose).
import absl.flags
from doctor.who import jodie

_FOO = absl.flags.DEFINE_string(...)

list_character = ["a", "b", "c"]


def add_value_into_list(list_character: list, value: str = "add_value") -> list:
    """
    Append value to list
    :param list_character: list character need to append value
    :param value: value append
    :return: New list after append value
    """
    return

print("list_character: ", list_character)