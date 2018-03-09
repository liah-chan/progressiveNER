
Miscellaneous utility function
import collections
import operator
import os
import time
import datetime
import shutil

def order_dictionary(dictionary, mode, reverse=False):
    Order a dictionary by 'key' or 'value'.
    mode should be either 'key' or 'value'

    if mode =='key':
        return collections.OrderedDict(sorted(dictionary.items(),
                                              key=operator.itemgetter(0),
                                              reverse=reverse))
    elif mode =='value':
        return collections.OrderedDict(sorted(dictionary.items(),
                                              key=operator.itemgetter(1),
                                              reverse=reverse))
    elif mode =='key_value':
        return collections.OrderedDict(sorted(dictionary.items(),
                                              reverse=reverse))
    elif mode =='value_key':
        return collections.OrderedDict(sorted(dictionary.items(),
                                              key=lambda x: (x[1], x[0]),
                                              reverse=reverse))
    else:
        raise ValueError("Unknown mode. Should be 'key' or 'value'")

def reverse_dictionary(dictionary):
    if type(dictionary) is collections.OrderedDict:
        return collections.OrderedDict([(v, k) for k, v in dictionary.items()])
    else:
        return {v: k for k, v in dictionary.items()}

def merge_dictionaries(*dict_args):
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def pad_list(old_list, padding_size, padding_value):
    assert padding_size >= len(old_list)
    return old_list + [padding_value] * (padding_size-len(old_list))

def get_basename_without_extension(filepath):
    return os.path.basename(os.path.splitext(filepath)[0])

def create_folder_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_current_milliseconds():
    return(int(round(time.time() * 1000)))


def get_current_time_in_seconds():
    return(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))

def get_current_time_in_miliseconds():
    return(get_current_time_in_seconds() + '-' + str(datetime.datetime.now().microsecond))


def convert_configparser_to_dictionary(config):
    my_config_parser_dict = {s:dict(config.items(s)) for s in config.sections()}
    return my_config_parser_dict


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)