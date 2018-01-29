#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import yaml as y
from collections import OrderedDict
from multiprocessing import Manager
import os, fnmatch, sys

# matchID imports
from log import err


def init():
	global manager, jobs, inmemory, log, conf, levCache, jobs_list
	manager = Manager()
	inmemory={}
	jobs = {}
	jobs_list = manager.dict()
	levCache={}
	log = None

def ordered_load(stream, Loader=y.Loader, object_pairs_hook=OrderedDict):
	class OrderedLoader(Loader):
		pass
	def construct_mapping(loader, node):
		loader.flatten_mapping(node)
		return object_pairs_hook(loader.construct_pairs(node))
	OrderedLoader.add_constructor(
		y .resolver.BaseResolver.DEFAULT_MAPPING_TAG,
		construct_mapping)
	return y.load(stream, OrderedLoader)


def deepupdate(original, update):
	"""
    Recursively update a dict.
    Subdict's won't be overwritten but also updated.
    """
	for key, value in original.iteritems():
	# python3 for key, value in original.items():
		if key not in update:
			update[key] = value
		elif isinstance(value, dict):
			deepupdate(value, update[key])
	return update

def check_conf(cfg,project,source):
	for key in list(["recipes","datasets","connectors"]):
		if (key in cfg.keys()):
			for obj in cfg[key]:
				cfg[key][obj]["source"]=source
				cfg[key][obj]["project"]=project

	return cfg

def read_conf():
	global conf
	try:
		conf_dir=conf["global"]["conf"]
	except:
		conf_dir="conf"

	cfg={"global":{"projects":{}}}

	cfg=read_conf_dir(conf_dir,cfg)

	try:
		projects=next(os.walk(cfg["global"]["paths"]["projects"]))[1]
		for project in projects:
			project=os.path.join(cfg["global"]["paths"]["projects"],project)
			cfg=read_conf_dir(project,cfg)

	except:
		print err()

	conf=cfg

def read_conf_dir(conf_dir,cfg):
	project=os.path.basename(conf_dir)
	cfg["global"]["projects"][project] = {"path": conf_dir,"files":{}}
	for root, dirnames, filenames in os.walk(conf_dir):
		#print root,dirnames,filenames
		subpath=root.replace(conf_dir+"/","") if (conf_dir != root) else ""
		for filename in fnmatch.filter(filenames, '*.yml'):
			conf_file=os.path.join(root, filename)
			filename=os.path.join(subpath,filename)
			cfg["global"]["projects"][project]["files"][filename]="not checked"

			with open(conf_file) as reader:
				try:
					update=ordered_load(reader)
					update=check_conf(update,project,filename)
					cfg=deepupdate(cfg,update)
					cfg["global"]["projects"][project]["files"][filename]="yaml is ok"
				except:
					cfg["global"]["projects"][project]["files"][filename]="yaml is ko - "+err()
	return cfg

