#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yaml as y
import json
from collections import OrderedDict
from multiprocessing import Manager
import os
import fnmatch
import sys
import datetime
import re

# matchID imports
from log import err


def init():
    global manager, jobs, inmemory, log, conf, levCache, jobs_list, conf_update
    manager = Manager()
    inmemory = {}
    jobs = {}
    jobs_list = manager.dict()
    levCache = {}
    log = None
    conf_update = None

def guess_type(value):
    if (value == "None"):
        return None
    if (value == "False"):
        return False
    if (value == "True"):
        return True
    try:
        return int(value)
    except:
        pass
    try:
        return float(value)
    except:
        pass
    try:
        return json.loads(value.decode('utf8'))
    except:
        pass
    try:
        return str(value)
    except:
        pass
    return value

def ordered_load(stream, Loader=y.Loader, object_pairs_hook=OrderedDict, tag='!ENV'):
    class OrderedLoader(Loader):
        pass

    def constructor_env_variables(loader, node):
        """
        Extracts the environment variable from the node's value
        :param yaml.Loader loader: the yaml loader
        :param node: the current node in the yaml
        :return: the parsed string that contains the value of the environment
        variable
        """
        value = loader.construct_scalar(node)
        match = pattern.findall(value)  # to find all env variables in line
        if match:
            full_value = value
            for g in match:
                full_value = guess_type(full_value.replace(
                    '${{{}}}'.format(g), os.environ.get(g, g)
                ))
            return full_value
        return value

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    # pattern for global vars: look for ${word}
    pattern = re.compile('.*?\${(\w+)}.*?')

    OrderedLoader.add_implicit_resolver(tag, pattern, None)
    OrderedLoader.add_constructor(tag,constructor_env_variables)

    OrderedLoader.add_constructor(
        y .resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping
        )

    return y.load(stream, OrderedLoader)


def deepupdate(original, update):
    """
    Recursively update a dict.
    Subdict's won't be overwritten but also updated.
    """
    for key, value in original.items():
        # python3 for key, value in original.items():
        if key not in update:
            update[key] = value
        elif isinstance(value, dict):
            deepupdate(value, update[key])
    return update


def check_conf(cfg, project, source):
    for key in list(["recipes", "datasets", "connectors"]):
        if (key in list(cfg.keys())):
            for obj in cfg[key]:
                cfg[key][obj]["source"] = source
                cfg[key][obj]["project"] = project

    return cfg


def read_conf():
    global conf, conf_update
    try:
        elapsed = (datetime.datetime.now() - conf_update).seconds
    except:
        elapsed = 100
    if (elapsed < 5):
        return
    conf_update = datetime.datetime.now()
    try:
        conf_dir = conf["global"]["conf"]
    except:
        conf_dir = "conf"

    cfg = {"global": {"projects": {}}}

    cfg = read_conf_dir(conf_dir, cfg)

    try:
        projects = next(os.walk(cfg["global"]["paths"]["projects"]))[1]
        for project in projects:
            project = os.path.join(cfg["global"]["paths"]["projects"], project)
            cfg = read_conf_dir(project, cfg)

    except:
        print(err())

    conf = cfg


def read_conf_dir(conf_dir, cfg):
    project = os.path.basename(conf_dir)
    cfg["global"]["projects"][project] = {"path": conf_dir, "files": {}}
    for root, dirnames, filenames in os.walk(conf_dir):
        # print root,dirnames,filenames
        subpath = root.replace(
            conf_dir + "/", "") if (conf_dir != root) else ""
        for filename in fnmatch.filter(filenames, '*.yml'):
            conf_file = os.path.join(root, filename)
            filename = os.path.join(subpath, filename)
            cfg["global"]["projects"][project][
                "files"][filename] = "not checked"

            with open(conf_file) as reader:
                try:
                    update = ordered_load(reader)
                    update = check_conf(update, project, filename)
                    cfg = deepupdate(cfg, update)
                    cfg["global"]["projects"][project][
                        "files"][filename] = "yaml is ok"
                except:
                    cfg["global"]["projects"][project]["files"][
                        filename] = "yaml is ko - " + err()
    return cfg


class Configured(object):

    def __init__(self, family=None, name=None):
        self.name = name
        self.family = family
        try:
            self.conf = conf[family][name]
        except:
            sys.exit("Ooops: {} not found in {} conf".format(
                self.name, self.family))
