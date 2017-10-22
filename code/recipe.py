#!/usr/bin/env python2
# -*- coding: utf-8 -*-


### import basics
import sys, os, io, fnmatch, re, datetime, hashlib, unicodedata, shutil

import traceback
from cStringIO import StringIO
import yaml as y
import json
import itertools
import simplejson
from collections import Iterable
from collections import OrderedDict
from pandas.io.json import json_normalize
from collections import deque


#### interact with datasets
import gzip
#from pandasql import sqldf
import elasticsearch
from elasticsearch import Elasticsearch, helpers
import pandas as pd

#### parallelize
import concurrent.futures
#import threading
from multiprocessing import Process, Manager

import uuid

### datascience dependecies
#dataprep with pandas
import automata
import random
import numpy as np
#ml dependencies
from sklearn.utils import shuffle
import sklearn.ensemble
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib
from numpy import array

#ngram with nltk
from nltk.util import ngrams
#from nltk.tokenize import WhitespaceTokenizer

# geodistance computation
from geopy.distance import vincenty
# from decimal import *
# from fuzzywuzzy import fuzz, process
# from fastcomp import compare
import jellyfish

### api
from flask import Flask,jsonify,Response, abort,request
from flask_restplus import Resource,Api,reqparse
from werkzeug.utils import secure_filename
from werkzeug.serving import run_simple
from werkzeug.wsgi import DispatcherMiddleware
import parsers


global manager, jobs, inmemory, log, conf

manager = Manager()
inmemory={}
jobs = {}



def err():
	#exc_info=sys.exc_info()
	exc_type, exc_obj, exc_tb = sys.exc_info()
	return "{} : {} line {}".format(str(exc_type),exc_obj,exc_tb.tb_lineno)
	#return "{}".format(traceback.print_exception(*exc_info))

def fwf_format(row,widths,sep=""):
	return sep.join([row[col].ljust(widths[i]-len(sep)) for i,col in enumerate(row.keys())])

def to_fwf(df, fname, widths=None,sep="",header=False,names=None,append=False,encoding="utf8",log=None):
	if (log == None):
		wlog = log
	mode = "w"
	try:
		wdf=df[names]
		if (sep == None):
			sep = ""
		wdf=wdf.apply(lambda row: fwf_format(row,widths,sep),axis=1)
		if header:
			header=sep.join([unicode(col).ljust(widths[i]-len(sep)) for i,col in enumerate(names)])
		else:
			header=None
		if append == True:
			mode = "a"
		with open(fname,mode) as f:
			if (header == None):
				np.savetxt(f,wdf.values,fmt="%s")
			else:
				np.savetxt(f,wdf.values,header=header.encode(encoding),fmt="%s",comments="")
		return
	except:
		if (log != None):
			log.write("Ooops : problem while writing fwf to {}: {}".format(fname,err()))
		else:
			raise

def parsedate(x="",format="%Y%m%d"):
	try:
		return datetime.datetime.strptime(x,self.args["format"])
	except:
		return x

def WHERE( back = 0 ):
    frame = sys._getframe( back + 1 )
    return "{}".format(frame.f_code.co_name)
    # return "%s/%s %s()" % ( os.path.basename( frame.f_code.co_filename ),
                        # frame.f_lineno, frame.f_code.co_name )

def jsonDumps(j=None,encoding='utf8'):
    return simplejson.dumps(j, ensure_ascii=False, encoding=encoding,ignore_nan=True)

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

def geopoint(geopoint):
	try:
		return tuple(float(x) for x in geopoint.replace("POINT(","").replace(")","").split(" ")[::-1])
	except:
		return ""

def distance(a,b):
	try:
		return round(10*vincenty(geopoint(a),geopoint(b)).kilometers)/10
	except:
		return ""



def replace_regex(x,regex):
	if (type(x)==str) | (type(x)==unicode):
		for r in regex:
			x=r[0].sub(r[1],x)
	elif (type(x)==list):
		x=[replace_regex(z,regex) for z in x]
	elif (type(x)==dict):
		x=dict((k,replace_regex(v,regex)) for (k,v) in x.items())
	return x

def replace_dict(x,dic):
	if (type(x)==str) | (type(x)==unicode):
		if x in list(dic.keys()):
			return dic[x]
	elif (type(x)==list):
		x=[replace_dict(z,dic) for z in x]
	elif ((type(x)==dict) | (type(x).__name__=="OrderedDict")):
		x=dict((k,replace_dict(v,dic)) for (k,v) in x.items())
	return x

def sha1(row):
	return hashlib.sha1(str(row)).hexdigest()

def ngrams(x,n=2):
	if (type(x)==list):
		return flatten([ngrams(z,n) for z in x])
	elif ((type(x)==unicode)|(type(x)==str)):
		return [x[i:i+N] for i in xrange(len(x)-N+1)]

def flatten(x):
    if (type(x)==list):
        return [a for i in x for a in flatten(i)]
    else:
        return x

def tokenize (x=None):
	if (type(x)==list):
		return flatten([tokenize(z) for z in x])
	elif ((type(x)==unicode) | (type(x)==str)):
		return re.split('\s\s*',x)
	else:
		return tokenize(str(x))


def normalize(x=None):
	if (type(x)==unicode):
		x=unicodedata.normalize('NFKD', x).encode('ascii', 'ignore')
	if (type(x)==str):
		x=re.sub('[^A-Za-z0-9]+', ' ', x.lower())
		x=re.sub('\s+', ' ', x)
		x=re.sub('^\s+$', '', x)
	elif (type(x)==list):
		x=filter(None,[normalize(z) for z in x])
		# if (len(x)==1):
		# 	x=x[0]
		# elif(len(x)==0):
		# 	x=""
	return x


def jw(s1,s2):
	maxi=0
	if (type(s1)==list):
		for s in s1:
			maxi=max(maxi,jw(s,s2))
		return maxi
	if (type(s2)==list):
		for s in s2:
			maxi=max(maxi,jw(s1,s))
		return maxi
	if (type(s1) == str):
		s1 = unicode(s1)
	if (type(s2) == str):
		s2 = unicode(s2)
	return round(100*jellyfish.jaro_winkler(s1,s2))/100

def levenshtein(s1, s2):
	if (not s1):
		s1=""
	if (not s2):
		s2=""
	if len(s1) < len(s2):
		return levenshtein(s2, s1)

	# len(s1) >= len(s2)
	if len(s2) == 0:
		return len(s1)
	previous_row = range(len(s2) + 1)
	for i, c1 in enumerate(s1):
		current_row = [i + 1]
		for j, c2 in enumerate(s2):
			insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
			deletions = current_row[j] + 1       # than s2
			substitutions = previous_row[j] + (c1 != c2)
			current_row.append(min(insertions, deletions, substitutions))
		previous_row = current_row

	return previous_row[-1]

def levenshtein_norm(s1,s2):
	if True:
		if (type(s1)==list):
			maxi=0
			for s in s1:
				maxi=max(maxi,levenshtein_norm(s,s2))
			return maxi

		if (type(s2)==list):
			maxi=0
			for s in s2:
				maxi=max(maxi,levenshtein_norm(s1,s))
			return maxi
		maxi=0
		return max(maxi,round(100-100*float(levenshtein(s1,s2))/(1+min(len(s1),len(s2))))/100)

	else:
		return 0


def safeeval(string=None,row=None):
	cell = None
	locals().update(row)
	if ("\n" in string) & ("cell" in string):
		try:
			exec string
			return cell
		except:
			return "Ooops in exec('{}'): {}".format(string,err())
	else :
		try:
			return eval(string)
		except:
			return "Ooops in eval('{}'): {}".format(string,err())


def match_lv1(x, list_strings):
	best_match = None
	best_score = 3
	for current_string in list_strings:
		current_score = compare(x, current_string)
		if (current_score==0):
			return current_string
		elif (current_score>0 & current_score < best_score):
			best_match=current_string
			best_score=current_score

	if best_score >= 2:
		return None
	return best_match

def match_jw(x, list_strings):
	best_match = None
	best_score = 0

	for current_string in list_strings:
		current_score = jellyfish.jaro_winkler(unicode(x), unicode(current_string))
		if(current_score > best_score):
			best_score = current_score
			best_match = current_string

	if (best_score>=0.95):
		return best_match
	else:
		return None

class Log(object):
	def __init__(self,name=None,test=False):
		self.name=name
		self.test=test
		self.writer=sys.stdout
		if (self.test==True):
			self.writer=StringIO()
			self.level=2
		else:
			if ("log" in conf["global"].keys()):
				try:
					self.dir=conf["global"]["log"]["dir"]
				except:
					self.dir=""
				self.file="{}./{}-{}.log".format(self.dir,datetime.datetime.now().isoformat(),self.name)

				try :
					self.writer=open(self.file,"w+")
				except:
					self.writer=sys.stdout

			try:
				self.level=conf["global"]["log"]["level"]
			except:
				self.level=1

		try:
			self.verbose=conf["global"]["log"]["verbose"]

		except:
			self.verbose=False

	def write(self,msg="",exit=False,level=1):
		try:
			self.writer
		except:
			return
		if (level<=self.level):
			fmsg="{} : {} - {}".format(datetime.datetime.now(),WHERE(1),msg)
			try:
				if (self.verbose==True):
					print(fmsg)
			except:
				pass

			self.writer.write(fmsg+"\n")
			self.writer.flush()
			if (exit):
				#os._exit(1)
				sys.exit(fmsg)
			return fmsg

class Configured(object):
	def __init__(self,family=None,name=None):
		self.name=name
		self.family=family
		try:
			self.conf=conf[family][name]
		except:
			sys.exit("Ooops: {} not found in {} conf".format(self.name,self.family))

class Connector(Configured):
	def __init__(self,name=None):
		Configured.__init__(self,"connectors",name)

		try:
			self.type=self.conf["type"]
		except:
			sys.exit("Ooops: type of connector {} has to be defined".format(self.name))

		if (self.type == "filesystem") | (self.type == "mongodb"):
			try:
				self.database=self.conf["database"]
			except:
				sys.exit("Ooops: database of connector {} has to be defined as type is {}".format(self.name,self.type))


		if (self.type == "elasticsearch") | (self.type == "mongodb"):
			try:
				self.host=self.conf["host"]
			except:
				sys.exit("Ooops: host of connector {} has to be defined as type is {}".format(connector_name,self.type))

		if (self.type == "elasticsearch"):
			try:
				self.port=self.conf["port"]
			except:
				self.port=9200
			self.es=Elasticsearch(self.host+":"+str(self.port))
			try:
				self.chunk_search=self.conf["chunk_search"]
			except:
				self.chunk_search=1000


		try:
			self.chunk=self.conf["chunk"]
		except:
			self.chunk=1000

		try:
			self.sample=self.conf["sample"]
		except:
			self.sample=500

		try:
			self.timeout=self.conf["timeout"]
		except:
			self.timeout=20

		try:
			self.thread_count=self.conf["thread_count"]
		except:
			self.thread_count=1

class Dataset(Configured):
	#a dataset is mainly a table linked to a pandas dataframe

	def __init__(self,name=None,parent=None):

		try:
			Configured.__init__(self,"datasets",name)
		except:
			if (self.name == "inmemory"):
				self.connector={"type": "inmemory", "chunk": 10000}
				return
			else:
				log.write("Ooops: no conf for dataset {}".format(self.name))

		try:
			self.parent=parent
		except:
			pass

		try:
			self.connector=Connector(self.conf["connector"])
		except:
			log.write("Ooops: failed to initiate connector for dataset {}".format(self.name))

		try:
			if type(self.conf["table"]) == str:
				self.table=self.conf["table"]
			else:
				self.table=self.conf["table"]["name"]
		except:
			log.write("Ooops: table of dataset {} has to be defined".format(self.name))

		if (self.connector.type == "elasticsearch"):
			try:
				self.select=self.conf["select"]
			except:
				self.select={"query" : {"match_all" : {}}}
			# self.select={"query":{"function_score":{"query":{"match_all":{}},"functions":[{"random_score":{}}]}}}
			try:
				self.doc_type=self.conf["doc_type"]
			except:
				self.doc_type=self.name

			try:
				self.doc_type=self.conf["doc_type"]
			except:
				self.doc_type=self.name

			try:
				self.body=json.loads(json.dumps(self.conf["body"]))
			except:
				self.body={}

		if (self.connector.type == "filesystem"):
			self.select=None
			try:
				self.files=[os.path.join(self.connector.database,f) 
							for f in os.listdir(self.connector.database)
							if re.match(r'^'+self.table+'$',f)]
				self.file=self.files[0]
			except:
				self.file=os.path.join(self.connector.database,self.table)
				#log.write("Ooops: couldn't set filename for dataset {}, connector {}".format(self.name,self.connector.name),exit=True)

			try:
				self.skiprows=self.conf["skiprows"]
			except:
				self.skiprows=0

			try:
				self.type=self.conf["type"]
			except:
				self.type="csv"

			try:
				self.prefix=self.conf["prefix"]
			except:
				self.prefix=None


			try:
				self.header=self.conf["header"]	
			except:
				if (self.type == "csv"):
					self.header="infer"
				else:
					self.header=False

			try:
				self.names=self.conf["names"]
			except:
				self.names=None

			try:
				self.sep=self.conf["sep"]
			except:
				if (self.type == "csv"):
					self.sep = ";"
				else:
					self.sep = None

			try:
				self.compression=self.conf["compression"]
			except:
				self.compression='infer'

			try:
				self.skiprows=self.conf["skiprows"]
			except:
				self.skiprows=0

			try:
				self.widths=self.conf["widths"]
			except:
				self.widths=[1000]

			try:
				self.encoding=self.conf["encoding"]
			except:
				self.encoding="utf8"



	def init_reader(self,df=None):
		try:
			self.log=self.parent.log
		except:
			pass

		if True:
			if (self.name == "inmemory"):
				if (df is not None):
					self.reader=[df]
				else:
					read_log.write("Ooops: can't initiate inmemory dataset with no dataframe",exit=True)
			elif (self.connector.type == "filesystem"):
				if (self.type == "csv"):
					self.reader=itertools.chain.from_iterable(pd.read_csv(file,sep=self.sep,usecols=self.select,chunksize=self.connector.chunk,
						compression=self.compression,encoding=self.encoding,dtype=object,header=self.header,names=self.names,skiprows=self.skiprows,
						prefix=self.prefix,iterator=True,index_col=False,keep_default_na=False) for file in self.files)
				elif (self.type == "fwf"):
					# with gzip.open(self.file, mode="r") as fh:
					# self.reader=pd.read_fwf(gzip.open(self.file,mode='rt'),chunksize=self.connector.chunk,skiprows=self.skiprows,
					self.reader=itertools.chain.from_iterable(pd.read_fwf(file,chunksize=self.connector.chunk,skiprows=self.skiprows,
						encoding=self.encoding,delimiter=self.sep,compression=self.compression,dtype=object,names=self.names,widths=self.widths,
						iterator=True,keep_default_na=False) for file in self.files)
			elif (self.connector.type == "elasticsearch"):
				self.reader= self.scanner()
		else:
			read_log.write("Ooops: couldn't initiate dataset {} : {}".format(self.name,err()),exit=True)

	def scanner(self,**kwargs):
		self.select=json.loads(json.dumps(self.select))
		scan=helpers.scan(client=self.connector.es, query=self.select, index=self.table, doc_type=self.doc_type,preserve_order=True,size=self.connector.chunk)

		hits=[]
		ids=[]
		for j,item in enumerate(scan):
			hits.append(item)
			ids.append(item['_id'])

			if (((j+1)%self.connector.chunk)==0):
				df=pd.concat(map(pd.DataFrame.from_dict, hits), axis=1)['_source'].T.reset_index(drop=True)
				df['_id']=ids
				hits=[]
				ids=[]
				yield df
		if (len(hits)>0):
			df=pd.concat(map(pd.DataFrame.from_dict, hits), axis=1)['_source'].T.reset_index(drop=True)
			df['_id']=ids
			yield df

	def init_writer(self):
		try:
			self.log=self.parent.log
		except:
			self.log=log

		#currently only manage elasticsearch injection
		if (self.name == "inmemory"):
			return
		elif (self.connector.type == "elasticsearch"):
			# ignore 400 cause by IndexAlreadyExistsException when creating an index
			try:
				if (self.mode == 'create'):
					self.connector.es.indices.delete(index=self.table, ignore=[400, 404])
					self.log.write("detete {}:{}/{}".format(self.connector.host,self.connector.port,self.table))
					self.connector.es.indices.create(index=self.table,body=self.body)
					self.log.write("create {}:{}/{}".format(self.connector.host,self.connector.port,self.table))
			except:
				self.log.write("Ooops: problem while initiating elasticsearch index {} for dataset {} : {}".format(self.table,self.name,err()),exit=True)
		elif (self.connector.type == "filesystem"):
			if (self.mode == 'create'):
				try:
					os.remove(self.file)
				except:
					# further better except should make difference btw no existing file and unwritable
					pass
		return None

	def write(self,chunk=0,df=None):
		size=df.shape[0]
		if (self.name == "inmemory"):
			return size
		processed=0
		if (size <= self.connector.chunk):
			df_list=[df]
		else:
			df_list=np.array_split(df,list(range(self.connector.chunk,size,self.connector.chunk)))
		for df in df_list:
			size=df.shape[0]
			if (self.connector.type == "elasticsearch"):
					#df.fillna("",inplace=True)
					if ('_id' in df.columns) & (self.mode == 'update'):
						records=df.drop(['_id'],axis=1).fillna("").T.to_dict()
						ids=df['_id'].T.to_dict()
						actions=[{'_op_type': 'index', '_id': ids[it], '_index': self.table,'_type': self.name, "_source": records[it]} for it in records]
					else:
						records=df.fillna("").T.to_dict()
						actions=[{'_op_type': 'index','_index': self.table,'_type': self.name, "_source": records[it]} for it in records]

					try:
						if (self.connector.thread_count>1):
							deque(helpers.parallel_bulk(self.connector.es,actions,thread_count=self.connector.thread_count))
						else:
							helpers.bulk(self.connector.es,actions)
						self.log.write("inserted {} lines to {}:{}/{}".format(size,self.connector.host,self.connector.port,self.table))
						processed+=size
					except:
						self.log.write("elasticsearch bulk failed {}:{}/{} \n {}".format(self.connector.host,self.connector.port,self.table,err()))
			elif (self.connector.type == "filesystem"):
				self.log.write("filesystem write {}".format(self.name))
				if (self.type == "csv"):
					try:
						if self.compression == 'infer':
							self.compression = None
						if (chunk == 0):
							header = self.header
						else:
							header = None
						df.to_csv(self.file,mode='a',index=False,sep=self.sep,
							compression=self.compression,encoding=self.encoding,header=header)
					except:
						self.log.write("write to csv failed writing {} : {}".format(self.file,err()))
				elif (self.type == "fwf"):
					if (chunk == 0):
						header = self.header
					else:
						header = False					
					try:
						to_fwf(df,self.file,names=self.names,header=header,sep=self.sep,widths=self.widths,append=True,encoding=self.encoding,log=self.log)
					except:
						self.log.write("write to fwf failed writing {} : {}".format(self.file,err()))
					pass
				pass
		return processed

class Recipe(Configured):
	def __init__(self,name=None,args={}):
		try:
			Configured.__init__(self,"recipes",name)
			self.type="configured"
			self.args=args
			self.log=None
		except:
			if (hasattr(self.__class__,"internal_"+name) and callable(getattr(self.__class__,"internal_"+name))):
				self.input=Dataset("inmemory",parent=self)
				self.input.select=None
				self.output=Dataset("inmemory",parent=self)
				self.type="internal"
				self.args=args
				return
			else:
				sys.exit("Ooops: can't couldn't find recipe {} in conf and no internal_{} function".format(self.name,self.name))

		#initiate input connection : creater a reader or use inmemory dataset
		try:
			if ("input" in self.args.keys()):
				self.input=Dataset(self.args.input,parent=self)
			elif ((type(self.conf["input"]) == str) | (type(self.conf["input"]) == unicode)):
				self.input=Dataset(self.conf["input"],parent=self)
			else:
				self.input=Dataset(self.conf["input"]["dataset"],parent=self)

			try:
				if (isinstance(self.conf["input"]["select"],list)):
					self.input.select=[unicode(x) for x in self.conf["input"]["select"]]
				else:
					self.input.select=self.conf["input"]["select"]
			except:
				self.input.select=None

			try:
				self.input.chunked=self.conf["input"]["chunked"]
			except:
				self.input.chunked=True

			try:
				self.threads=self.conf["threads"]
			except:
				try:
					self.threads=conf["global"]["threads_by_job"]
				except:
					self.threads=1

		except:
			self.input=Dataset("inmemory",parent=self)
			self.input.select=None


		#initiate output connection : create a writer or use current dataframe
		try:
			if ("output" in self.args.keys()):
				self.output=Dataset(self.args.output,parent=self)
			elif ((type(self.conf["output"]) == str) | (type(self.conf["output"]) == unicode)):
				self.output=Dataset(self.conf["output"],parent=self)
			else:
				self.output=Dataset(self.conf["output"]["dataset"],parent=self)
			try:
				# mode can be 'create, append, update'
				self.output.mode=self.conf["output"]["mode"]
			except:
				self.output.mode='create'
		except:
			self.output=Dataset("inmemory",parent=self)

		try:
			self.steps=[]
			for s in self.conf["steps"]:
				function=s.keys()[0]
				try:
					self.steps.append(Recipe(name=function,args=s[function]))
				except:
					self.log.write("Ooops: recipe {} calls an unknown function {}".format(self.name,function))
		except:
			pass

	def init(self,df=None,parent=None,test=False):
		try:
			self.test=test
			if (parent != None):
				self.parent=parent
				self.log=self.parent.log
			else:
				self.parent=None
				self.log=Log(self.name,test=test)

		except:
			self.log.write("Ooops: couldn't init log for recipe {}".format(self.name),exit=True)
		try:
			self.input.init_reader(df=df)
		except:
			self.log.write("Ooops: couldn't init input {} of recipe {}: {}".format(self.input.name,self.name,err()))
		if (self.test==False):
			try:
				self.output.init_writer()
			except:
				self.log.write("Ooops: couldn't init output {} of recipe {}".format(self.output.name,self.name))

	def set_job(self,job=None):
		self.job=job
		self.job.date = datetime.datetime.now().isoformat()
		return

	def start_job(self):
		self.job.start()
		return

	def join_job(self):
		self.job.join()

	def stop_job(self):
		self.job.terminate()
		return

	def job_status(self):
		try:
			if self.job.is_alive():
				return "up"
			else:
				try:
					self.job.join()
					self.job=None
					return "done"
				except:
					return "down"
		except:
			return "down"

	def run_chunk(self,i,df):
		# if ((self.name == "join") & (i<=conf["global"]["threads_by_job"]) & (i>1)):
		# 	#stupid but working hack to leave time for inmemory preload of first thread first chunk
		# 	#the limit is if the treatment of a chunk takes more than 30s... better workaround has to be found
		# 	time.sleep(30)
		if (self.input.name != "inmemory"):
			self.log.write("proceed chunk {} : {} rows from {} with recipe {}".format((i+1),df.shape[0],self.input.name,self.name))
		if (self.type == "internal"):
			df=getattr(self.__class__,"internal_"+self.name)(self,df=df)
		elif(len(self.steps)>0):
			for recipe in self.steps:
				try:
					self.log.write("{} > {}".format(self.name,recipe.name),level=2)
					recipe.init(df=df,parent=self,test=self.test)
					# recipe.run()
					df=recipe.run_chunk(i,df)
					if (recipe.name=="pause"):
						return df
				except:
					self.log.write("Ooops: error while calling {} in {} - {}".format(recipe.name,self.name,err()))
		if ((self.output.name != "inmemory") & (self.test==False)):
			#df.fillna('',inplace=True)
			#print self.name,self.input.name,i,self.input.processed,self.output.name
			self.input.processed+=self.output.write(i,df)
			self.log.write("wrote {} to {} after recipe {}".format(df.shape[0],self.output.name,self.name))
		return df

	def run(self,head=None):
		if (head is None):
			try:
				head=self.conf["test_chunk_size"]
			except:
				head=conf["global"]["test_chunk_size"]
		#log("initiating recipe {}".format(self.name))
		self.df=[]
		self.input.processed=0
		try:
			# for i, df in enumerate(self.input.reader):
			# 	self.run_chunk(i,df,test)
			#first lauch the first chunk for initialization of "inmemory" datasets, then iterate with // threads
			if (self.input.chunked==True):
				self.df=next(self.input.reader,"")
				if(self.test==True):
					self.df=self.df.head(n=head)
			else:
				self.df=pd.concat([df for df in self.input.reader])

			# removes trailing space in columns
			self.df.rename(columns=lambda x: x.strip(), inplace=True)

			# runs the recipe
			self.df=self.run_chunk(0,self.df)

			if (self.test==True):
				# end of work if in test mode
				return self.df
			else:
				# proceed to the whole dataset with // threads
				# # concurrent futures version - only scales to +30% !
				# with concurrent.futures.ThreadPoolExecutor(max_workers=conf["global"]["threads_by_job"]) as executor:
				# 	future_to_df={executor.submit(self.run_chunk,i,df): df for i, df in enumerate(self.input.reader)}
				# 	for future in concurrent.futures.as_completed(future_to_df):
				# 		pass
				# # process to parallelization with multiprocessing lib
				queue={}
				for i, df in enumerate(self.input.reader):
					# removes trailing space in columns
					df.rename(columns=lambda x: x.strip(), inplace=True)
					nt= i%self.threads
					if (nt in queue.keys()):
						queue[nt].join()
					queue[nt]=Process(target=self.run_chunk,args=[i+1,df])
					queue[nt].start()

		except SystemExit:
			try:
				for thread in queue.keys():
					queue[thread].terminate()
			except:
				pass
			self.log.write("Ooops: SIGTERM {}".format(self.name),exit=True)
		except:
			if (self.test==True):
				error=err()
				try:
					self.df=df
				except:
					self.df=None
				self.log.write("error in main loop of {} {}: {}".format(self.name,str(self.input.select),error))
				return self.df
			else:
				self.log.write("Ooops: error while running {} - {}".format(self.name,err()))
			try:
				for thread in queue.keys():
					queue[thread].terminate()
			except:
				pass

	def select_columns(self,df=None):
		try:
			if ("select" in self.args.keys()):
				if (type(self.args["select"])==str) | (type(self.args["select"])==unicode):
					self.cols=[x for x in list(df) if re.match(self.args["select"],x)]
				else:
					self.cols=self.args["select"]
			else:
			#apply to all columns if none selected
				self.cols=list(df)
		except:
			self.cols=[]


	def prepare_categorical(self,df=None):
		df = df[self.categorical].reset_index(drop=True).T.to_dict().values()
		prep = DictVectorizer()
		df = prep.fit_transform(df).toarray()
		return df

	def prepare_numerical(self,df=None):
		df=df[self.numerical].fillna("")
		df=df.applymap(lambda x: 0 if ((str(x) == "") | (x == None)) else float(x))
		#imp = Imputer(missing_values=np.nan, strategy='mean', axis=0)
		#df=imp.fit_transform(df)
		return df

	def internal_eval(self,df=None):
		try:
			cols=[]
			for step in self.args:
				for col in step.keys():
					cols.append(col)
					if True:
						if type(step[col])==str:
							df[col]=df.apply(lambda row:safeeval(step[col],row),axis=1)
						elif type(step[col])==unicode:
							df[col]=df.apply(lambda row:safeeval(step[col],row),axis=1)
						elif (type(step[col])==list):
							multicol=[unicode(x) for x in step[col]]
							#print col,multicol, list(df)
							df[col]=df.apply(lambda row: [safeeval(x,row) for x in multicol], axis=1)
					else:
						pass
			if ("Ooops" in str(df[cols])):
				# report detailed error analysis
				global_col_err=[]
				partial_col_err=[]
				nerr_total=0
				for col in cols:
					col_err=df[df[col].apply(lambda x: "Ooops" in str(x))]
					nerr=col_err.shape[0]
					if (nerr == df.shape[0]):
						global_col_err.append(col)
					elif (nerr> 0):
						partial_col_err.append(col)
						nerr_total+=nerr
				if (len(global_col_err)>0):
					self.log.write("Ooops: warning in {} : global error in {}".format(self.name,global_col_err),exit=False)
				if (len(partial_col_err)>0):
					self.log.write("Ooops: warning in {} : {}/{} errors in {}".format(self.name,nerr_total,df.shape[0],partial_col_err),exit=False)
			return df
		except:
			self.log.write("Ooops: problem in {} - {}: {} - {}".format(self.name,col,step[col],err()),exit=False)
			return df


	def internal_rename(self,df=None):
		dic={v: k for k, v in self.args.iteritems()}
		df.rename(columns=dic,inplace=True)
		return df

	def internal_map(self,df=None):
		for col in list(self.args.keys()):
			if True:
				if type(self.args[col])==str:
					df[col]=df[self.args[col]]
				elif type(self.args[col])==unicode:
					df[col]=df[self.args[col]]
				elif (type(self.args[col])==list):
					multicol=[unicode(x) for x in self.args[col]]
					df[col]=df.apply(lambda row: [row[x] for x in multicol], axis=1)
			else:
				pass
		return df


	def internal_shuffle(self,df=None):
		# fully shuffles columnes and lines
		try:
			return df.apply(np.random.permutation)
		except:
			self.log.write("Ooops: problem in {} - {}".format(self.name,err()),exit=False)


	def internal_build_model(self,df=None):
		# callable recipe for building method
		# tested only with regression tree
		try:

			if ("numerical" in self.args.keys()):
				if (type(self.args["numerical"])==str) | (type(self.args["numerical"])==unicode):
					self.numerical=[x for x in list(df) if re.match(self.args["numerical"],x)]
				else:
					self.numerical=self.args["numerical"]
			else:
				self.numerical=[]

			if ("categorical" in self.args.keys()):
				if (type(self.args["categorical"])==str) | (type(self.args["categorical"])==unicode):
					self.categorical=[x for x in list(df) if re.match(self.args["categorical"],x)]
				else:
					self.categorical=self.args["categorical"]
			else:
				self.categorical=[]

			if ("target" in self.args.keys()):
				if (type(self.args["target"])==str) | (type(self.args["target"])==unicode):
					self.target=[x for x in list(df) if re.match(self.args["target"],x)]
				else:
					self.target=self.args["target"]
			else:
				self.log.write("Ooops: no target specified for model")
				return df


			self.model={"library": sklearn.ensemble, "method": "RandomForestRegressor", "parameters": {}, "tries": 5}

			for arg in list(["method","parameters","library","tries","test_size","name"]):
				try:
					self.model[arg]=json.loads(json.dumps(self.args["model"][arg]))
				except:
					try:
						self.model[arg]=json.loads(json.dumps(conf["machine_learning"]["model"][arg]))
					except:
						pass

			# prepare data
			Xn=self.prepare_numerical(df)
			Xc=self.prepare_categorical(df)
			# (Xn,prep_num)=self.prepare_numerical(df[self.numerical])
			# (Xc,prep_cat)=self.prepare_categorical(df[self.categorical])

			X=np.hstack((Xn,Xc))
			#for debug: self.log.write("{} {} {} {} {}".format(X.shape,len(self.numerical),Xn.shape,len(self.categorical),Xc.shape))

			Y=df[self.target].applymap(lambda x: 1 if x else 0)
			# prep = DictVectorizer()
			# X=X.to_dict().values()
			# X = prep.fit_transform(X).toarray()
			err_min=1
			for i in range(0,self.model["tries"]) :
				X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=self.model["test_size"])
				clf = getattr(self.model["library"],self.model["method"])(**self.model["parameters"])
				# clf = gettattr('',self.type)(n_estimators=100,max_depth=5,min_samples_leaf=10)
				clf = clf.fit(X_train, Y_train)
				# erra = mean_squared_error( clf.predict(X_train), Y_train)**0.5
				# errb = mean_squared_error( clf.predict(X_test), Y_test)**0.5
				erra = roc_auc_score( Y_train, clf.predict(X_train) )
				errb = roc_auc_score(Y_test, clf.predict(X_test) )
				if (errb<err_min):
					best_clf=clf
					err_min=errb
				self.log.write("estimator {}: auc_train {}, auc_score {}".format(i,erra,errb))
			df["matchid_hit_score_ml"]=best_clf.predict(X)
			df["matchid_hit_score_ml"]=df["matchid_hit_score_ml"].apply(lambda x: round(100*x))

			self.log.write("{}\n{}".format(self.numerical,self.categorical))
			if (self.test==False):
				try:
					filename=os.path.join(conf["global"]["paths"]["models"],secure_filename(self.model["name"]+".model"))
					joblib.dump(best_clf, filename)
					# filename=os.path.join(conf["global"]["paths"]["models"],secure_filename(self.model["name"]+".cat"))
					# joblib.dump(prep_cat,filename)
					# filename=os.path.join(conf["global"]["paths"]["models"],secure_filename(self.model["name"]+".num"))
					# joblib.dump(prep_num,filename)
					self.log.write("Saved model {}".format(self.model["name"]))
				except:
					self.log.write("Ooops: couldn't save model in {} - {}".format(self.name,err()),exit=False)

		except:
			self.log.write("Ooops: problem while building model from numerical: {} and categorical: {} to {} in {} - {}".format(self.numerical,self.categorical,self.target,self.name,err()),exit=False)
			return df

		return df

	def internal_apply_model(self,df=None):
		# callable recipe for building method
		# tested only with regression tree
		try:
			if ("numerical" in self.args.keys()):
				if (type(self.args["numerical"])==str) | (type(self.args["numerical"])==unicode):
					self.numerical=[x for x in list(df) if re.match(self.args["numerical"],x)]
				else:
					self.numerical=self.args["numerical"]
			else:
				self.numerical=[]

			if ("numerical" in self.args.keys()):
				if (type(self.args["numerical"])==str) | (type(self.args["numerical"])==unicode):
					self.numerical=[x for x in list(df) if re.match(self.args["numerical"],x)]
				else:
					self.numerical=self.args["numerical"]
			else:
				self.numerical=[]


			if ("categorical" in self.args.keys()):
				if (type(self.args["categorical"])==str) | (type(self.args["categorical"])==unicode):
					self.categorical=[x for x in list(df) if re.match(self.args["categorical"],x)]
				else:
					self.categorical=self.args["categorical"]
			else:
				self.categorical=[]


			if ("target" in self.args.keys()):
				self.target=self.args["target"]
			else:
				self.log.write("Ooops: no target specified for model prediction")
				return df


			#load model
			filename=os.path.join(conf["global"]["paths"]["models"],secure_filename(self.args["name"]+".model"))
			clf=joblib.load(filename)
			# filename=os.path.join(conf["global"]["paths"]["models"],secure_filename(self.args["name"]+".cat"))
			# prep_cat=joblib.load(filename)
			# filename=os.path.join(conf["global"]["paths"]["models"],secure_filename(self.args["name"]+".num"))
			# prep_num=joblib.load(filename)


			# prepare data
			Xn=self.prepare_numerical(df)
			Xc=self.prepare_categorical(df)
			# (Xn,prep_num)=self.prepare_numerical(df[self.numerical],prep_num)
			# (Xc,prep_cat)=self.prepare_categorical(df[self.categorical],prep_cat)
			X=np.hstack((Xn,Xc))
			#for debug : self.log.write("{} {} {} {} {}".format(X.shape,len(self.numerical),Xn.shape,len(self.categorical),Xc.shape))

			df[self.target]=clf.predict(X)
			df[self.target]=df[self.target].apply(lambda x: round(100*x))

		except:
			self.log.write("Ooops: problem while applying model from numerical: {} and categorical: {} to {} in {} - {}".format(self.numerical,self.categorical,self.target,self.name,err()),exit=False)

		return df




	def internal_keep(self,df=None):
		#keep only selected columns
		self.select_columns(df=df)
		try:
			if ("where" in self.args.keys()):
				df["matchid_selection_xykfsd"]=df.apply(lambda row:safeeval(self.args["where"],row),axis=1)
				df=df[df.matchid_selection_xykfsd == True]
				del df["matchid_selection_xykfsd"]
			return df[self.cols]
		except:
			self.log.write("Ooops: problem with columns selection in {} - {} - {}".format(self.name,self.cols,err()),exit=False)
			return df


	def internal_to_integer(self,df=None):
		#keep only selected columns
		self.select_columns(df=df)
		try:
			df[self.cols]=df[self.cols].applymap(lambda x: np.nan if (str(x) == "") else int(x))
			return df
		except:
			self.log.write("Ooops: in {}, problem converting to int: {} - {}".format(self.name,self.cols,err()),exit=False)
			return df

	def internal_to_float(self,df=None):
		#keep only selected columns
		self.select_columns(df=df)
		try:
			na_value=self.args["na_value"]
		except:
			na_value=np.nan
		try:
			df[self.cols]=df[self.cols].applymap(lambda x: na_value if (str(x) == "") else float(x))
			return df
		except:
			self.log.write("Ooops: in {}, problem converting to float: {} - {}".format(self.name,self.cols,err()),exit=False)
			return df


	def internal_ngram(self,df=None):
		#keep only selected columns
		self.select_columns(df=df)
		try:
			if ("where" in self.args.keys()):
				df["matchid_selection_xykfsd"]=df.apply(lambda row:safeeval(self.args["where"],row),axis=1)
				df=df[df.matchid_selection_xykfsd == True]
				del df["matchid_selection_xykfsd"]
			return df[self.cols]
		except:
			self.log.write("Ooops: problem with columns selection in {} - {} - {}".format(self.name,self.cols,err()),exit=False)
			return df

	# def internal_sql(self,df=None):
	# 	if True:
	# 		if ("query" in self.args.keys()):
	# 			print self.args["query"]
	# 			print sqldf(self.args["query"], locals())
	# 		return df
	# 	else:
	# 		return df


	def internal_delete(self,df=None):
		#keep only selected columns
		self.select_columns(df=df)
		#log("selecting {}".format(self.cols),level=3)
		try:
			df.drop(self.cols,axis=1,inplace=True)
			# for col in self.cols:
			# 	del df[col]
			return df
		except:
			self.log.write("Ooops: problem with columns selection in {} - {} - {}".format(self.name,self.cols,err()),exit=False)
			return df



	def internal_join(self,df=None):
		try:
			join_type="in_memory"
			if (self.args==None):
				sys.exit("Ooops: no args in join")
			if ("type" in self.args.keys()):
				if (self.args["type"] == "elasticsearch"):
					join_type="elasticsearch"
			if (join_type == "in_memory"): # join in memory
				ds = self.args["dataset"]

				# cache inmemory reading
				# a flush method should be created
				try:
					# inmemory cache
					inmemory[ds].df
				except:
					self.log.write("Creating cache for join with dataset {} in {}".format(ds,self.name))
					inmemory[ds]=Dataset(self.args["dataset"])
					inmemory[ds].init_reader()
					inmemory[ds].df=pd.concat([dx for dx in inmemory[ds].reader]).reset_index(drop=True)


				# collects useful columns
				if ("select" in list(self.args.keys())):
					#select columns to retrieve in join
					cols=[self.args["select"][col] for col in list(self.args["select"].keys())]
					if ("strict" in list(self.args.keys())):
						#keep joining cols
						cols=list(set().union(cols,	[self.args["strict"][x] for x in list(self.args["strict"].keys())]))
					if ("fuzzy" in list(self.args.keys())):
						#keep fuzzy joining cols
						cols=list(set().union(cols,	[self.args["fuzzy"][x] for x in list(self.args["fuzzy"].keys())]))
						#initiate levenstein matcher (beta : not optimized)
						#this method remains in memory
						try:
							inmemory[ds].matcher
						except:
							inmemory[ds].matcher={}
						for col in list(self.args["fuzzy"].keys()):
							try:
								inmemory[ds].matcher[self.args["fuzzy"][col]]
							except:
								self.log.write("Creating automata cache for fuzzy join on column {} of dataset {} in {}".format(col,ds,self.name))
								words=sorted(set(inmemory[ds].df[self.args["fuzzy"][col]].tolist()))
								inmemory[ds].matcher[self.args["fuzzy"][col]]=automata.Matcher(words)

				# caches filtered version of the dataset				
				try:
					join_df = inmemory[ds].filtered[sha1(cols)]
				except:
					try:
						self.log.write("Creating filtered cache for join with dataset {} in {}".format(ds,self.name))
						inmemory[ds].filtered
					except:
						inmemory[ds].filtered = {}
					inmemory[ds].filtered[sha1(cols)] = inmemory[ds].df[cols]
					join_df = inmemory[ds].filtered[sha1(cols)]

				if ("fuzzy" in list(self.args.keys())):
					for col in list(self.args["fuzzy"].keys()):
						#get fuzzy matches for the fuzzy columns
						if ("fuzzy_method" in list(self.args.keys())):
							fuzzy_method = self.args["fuzzy_method"]
						else:
							fuzzy_method="automata"
						if (fuzzy_method=="automata"):
							#using levenshtein automata (tested 10x faster as tested against fastcomp and jaro winkler)
							#a full openfst precompile automata would be still faster but not coded for now
							df[col+"_match"]=df[col].map(lambda x: 
								next(itertools.chain.from_iterable(
									automata.find_all_matches(x, dist,inmemory[ds].matcher[self.args["fuzzy"][col]])
									for dist in range(2)),""))
						elif (fuzzy_method=="jellyfish"):
							#using jellyfish jaro winkler
							df[col+"_match"]=df[col].map(lambda x:match_jw(x,join_df[self.args["fuzzy"][col]]))
						elif (fuzzy_method=="fastcomp"):
							#using fastcomp
							df[col+"_match"]=df[col].map(lambda x:match_lv1(x,join_df[self.args["fuzzy"][col]]))
					#now prematched fuzzy terms in cols _match are ok for a strict join
					#list joining columns
					left_on=[col+"_match" for col in self.args["fuzzy"].keys()]
					right_on=[self.args["fuzzy"][x] for x in self.args["fuzzy"].keys()]
					if ("strict" in list(self.args.keys())):
						#complete joining columns list if asked
						left_on=list(set().union(left_on,list(self.args["strict"].keys())))
						right_on=list(set().union(right_on,[self.args["strict"][x] for x in list(self.args["strict"].keys())]))

					#joining, the right dataset being keepd in memory
					df=pd.merge(df,join_df,
						how='left',left_on=left_on,
						right_on=right_on,
						left_index=False,right_index=False)
					# self.log.write("{}x{} - {}\n{}".format(left_on,right_on,self.name,df[list(set().union(left_on,right_on))].head(n=5)))

					#map new names of retrieved colums
					if ("select" in self.args):
						reverse={v: k for k, v in self.args["select"].iteritems()}
						# python 3 reverse={v: k for k, v in self.args["select"].items()}
						df.rename(columns=reverse,inplace=True)
					#remove unnecessary columns of the right_on
					for key in right_on:
						try:
							del df[key]
						except:
							pass
				elif ("strict" in self.args.keys()):
					# simple strict join
					df=pd.merge(df,join_df,
						how='left',left_on=list(self.args["strict"].keys()),
						right_on=[self.args["strict"][x] for x in list(self.args["strict"].keys())],
						left_index=False,right_index=False)

					#map new names of retrieved colums
					if ("select" in self.args.keys()):
						reverse={v: k for k, v in self.args["select"].iteritems()}
						# python3 reverse={v: k for k, v in self.args["select"].items()}
						df.rename(columns=reverse,inplace=True)
					#remove unnecessary columns of the right_on
					for key in [self.args["strict"][x] for x in self.args["strict"].keys()]:
						try:
							del df[key]
						except:
							pass
			else: # join with elasticsearch
				if True:
					es=Dataset(self.args["dataset"])
					query=self.args["query"]
					index=0
					if True:
						m_res=[]

						rest=df.applymap(lambda x: "" if x is None else x)
						rest.fillna("",inplace=True)

						#elasticsearch bulk search
						while rest.shape[0]>0:
							part=rest[:es.connector.chunk_search]
							rest=rest[es.connector.chunk_search:]
							index+=es.connector.chunk_search
							bulk="\n".join(part.apply(lambda row: jsonDumps({"index": es.table})+"\n"+jsonDumps(replace_dict(query,row)),axis=1))
							#self.log.write("\n{}".format(bulk))
							tries=0
							success=False
							max_tries=3
							while(tries<max_tries):
								try:
									res=es.connector.es.msearch(bulk)
									df_res = pd.concat(map(pd.DataFrame.from_dict, res['responses']), axis=1)['hits'].T.reset_index(drop=True)
									max_tries=tries
									success=True
								except elasticsearch.TransportError:
									error=err()+bulk.encode("utf-8", 'ignore')
									tries=max_tries
									df_res=part['matchid_id'].apply(lambda x: {"_source" : {}})
								except:
									tries+=1
									df_res=part['matchid_id'].apply(lambda x: {"_source" : {}})
							if (success==False):
								self.log.write("join {} x {} failure in sub-chunk {} to {} : {}".format(self.name,self.args["dataset"],index-es.connector.chunk_search,index,error))
							m_res.append(df_res)
						df_res=pd.concat(m_res).reset_index(drop=True)

						# #self.log.write("debug: {}".format(df_res))
						df_res['matchid_hit_matches_unfiltered']=df_res['total']
						#df_res.drop(['total','failed','successful','max_score'],axis=1,inplace=True)
						df_res.drop(['failed','successful'],axis=1,inplace=True)
						df=pd.concat([df.reset_index(drop=True),df_res],axis=1)
						#self.log.write("after ES request:{}".format(df.shape))

						try:
							unfold=self.args["unfold"]
						except:
							unfold=True
						#unfold
						if (unfold == True):
							unfold=Recipe('unfold',args={"select": ['hits'], "fill_na": ""})
							unfold.init(df=df,parent=self,test=self.test)
							df=unfold.run_chunk(0,df)
							#self.log.write("after unfold:{}".format(df.shape))
							try:
								keep_unmatched=self.args["keep_unmatched"]
							except:
								keep_unmatched=False
							if (keep_unmatched == False):
								df=df[df.hits != ""]
							del df_res

							# #unnest columns of each match : a <> {key1:val11, key2:val21} gives : a <> val11, val21
							try:
								unnest=self.args["unnest"]
							except:
								unnest=True
							if (unnest == True):
								try:
									prefix=self.args["prefix"]
								except:
									prefix="hit_"
								df['hits']=df['hits'].apply(lambda x: {} if (x == "") else x['_source'] )

								unnest=Recipe('unnest',args={"select": ['hits'],"prefix": prefix})
								unnest.init(df=df,parent=self,test=self.test)
								df=unnest.run_chunk(0,df)
								#self.log.write("after unnest :{}".format(df.shape))

					else:
						pass
				else:
					return df
		except:
			self.log.write("join {} x {} failed : {}".format(self.name,self.args["dataset"],err()))
		return df.fillna('')


	def internal_unnest(self,df=None):
		self.select_columns(df=df)
		try:
			prefix=self.args["prefix"]
		except:
			prefix=''

		try:
			df_list=[df]
			for col in self.cols:
				df_list.append(df[col].apply(pd.Series).add_prefix(prefix))
			return pd.concat(df_list,axis=1).drop(self.cols,axis=1)
		except:
			self.log.write("Ooops: error in unnest: {}".format(err()))
			return df


	def internal_unfold(self,df=None):
		self.select_columns(df=df)
		try:
			fill_na=self.args["fill_na"]
		except:
			fill_na=""
		# make sure `self.cols` is a list
		try:
			if self.cols and not isinstance(self.cols, list):
				self.cols = [self.cols]
			# all columns except `self.cols`
			idx_cols = df.columns.difference(self.cols)

			# calculate lengths of lists
			lens = df[self.cols[0]].str.len()

			if (lens > 0).all():
				# ALL lists in cells aren't empty
				return pd.DataFrame({
				col:np.repeat(df[col].values, df[self.cols[0]].str.len())
					for col in idx_cols
				}).assign(**{col:np.concatenate(df[col].values) for col in self.cols}) \
				.loc[:, df.columns]
			else:
				# at least one list in cells is empty
				return pd.DataFrame({
					col:np.repeat(df[col].values, df[self.cols[0]].str.len())
					for col in idx_cols
				}).assign(**{col:np.concatenate(df[col].values) for col in self.cols}) \
				.append(df.loc[lens==0, idx_cols]).fillna(fill_na) \
				.loc[:, df.columns]
		except:
			self.log.write("Ooops: failure in unfold : {}".format(err()))
			return df

	def internal_parsedate(self,df=None):
		self.select_columns(df=df)
		if ("format" in self.args.keys()):
			#parse string do datetime i.e. 20001020 + %Y%m%d => 2000-10-20T00:00:00Z
			df[self.cols]=df[self.cols].applymap(lambda x:
				parsedate(x,self.args["format"]))

		return df


	def internal_replace(self,df=None):
		if True:
			self.select_columns(df=df)
			if ("regex" in self.args.keys()):
				regex=[]
				# warning: replace use a dict which is not ordered
				for r in self.args["regex"]:
					regex.append([re.compile(r.keys()[0]),r[r.keys()[0]]])
				pd.options.mode.chained_assignment = None
				df[self.cols]=df[self.cols].applymap(lambda x: replace_regex(x,regex))
			return df
		else:
			return df

	def internal_normalize(self,df=None):
		if True:
			self.select_columns(df=df)
			df[self.cols]=df[self.cols].applymap(normalize)
			return df
		else:
			return df

	def internal_pause(self,df=None):
		try:
			try:
				head=self.args["head"]
			except :
				head=conf["global"]["test_chunk_size"]

			self.select_columns(df=df)
			return df[self.cols].head(n=head)
		except:
			return df

def thread_job(recipe=None, result={}):
	try:
		result["df"] = recipe.run()
		result["log"] = str(recipe.log.writer.getvalue())
	except:
		pass

def jsonize(j=None):
	# return Response(json.dumps(js),status=200, mimetype='application/json')
	return jsonify(j)


def allowed_upload_file(filename=None):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in conf["global"]["data_extensions"]


def allowed_conf_file(filename=None):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in conf["global"]["recipe_extensions"]



read_conf()

app = Flask(__name__)
api=Api(app,version="0.1",title="matchID API",description="API for data matching developpement")
app.config['APPLICATION_ROOT']=conf["global"]["api"]["prefix"]

@api.route('/conf/', endpoint='conf' )
class Conf(Resource):
	def get(self):
		'''get all configured elements
		Lists all configured elements of the backend, as described in the yaml files :
		- global configuration
		- projects :
		  - datasets
		  - recipes'''
		try:
			read_conf()
			return conf["global"]
		except:
			return {"error": "problem while reading conf"}

@api.route('/upload/', endpoint='upload')
class Upload(Resource):
	def get(self):
		'''list uploaded resources'''
		return list([filenames for root, dirnames, filenames in os.walk(conf["global"]["paths"]["upload"])])[0]

	@api.expect(parsers.upload_parser)
	def post(self):
		'''upload multiple tabular data files, .gz or .txt or .csv'''
		response={"upload_status":{}}
		args = parsers.upload_parser.parse_args()
		for file in args['file']:
			if (allowed_upload_file(file.filename)):
				try:
					file.save(os.path.join(conf["global"]["paths"]["upload"], secure_filename(file.filename)))
					response["upload_status"][file.filename]="ok"
				except:
					response["upload_status"][file.filename]=err()
			else:
				response["upload_status"][file.filename]="extension not allowed"
		return response

@api.route('/upload/<file>', endpoint='upload/<file>')
@api.doc(parmas={'file': 'file name of a previously uploaded file'})
class actionFile(Resource):
	def get(self,file):
		'''get back uploaded file'''
		filetype="unknown"
		pfile=os.path.join(conf["global"]["paths"]["upload"],file)
		try:
			df=pd.read_csv(pfile,nrows=100)
			filetype="csv"
		except:
			pass
		return {"file": file, "type_guessed": filetype}

	def delete(self,file):
		'''deleted uploaded file'''
		try:
			pfile=os.path.join(conf["global"]["paths"]["upload"],file)
			os.remove(pfile)
			return {"file": file, "status": "deleted"}
		except:
			api.abort(404,{"file": file, "status": err()})


@api.route('/conf/<project>/', endpoint='conf/<project>')
@api.doc(parms={'project': 'name of a project'})
class DirectoryConf(Resource):
	def get(self,project):
		'''get configuration files of a project'''
		read_conf()
		if project in list(conf["global"]["projects"].keys()):
			return conf["global"]["projects"][project]
		else:
			api.abort(404)

	@api.expect(parsers.conf_parser)
	def post(self,project):
		'''(KO) import a zipped project'''
		if (directory != "conf"):
			response={"upload_status":{}}
			args = parsers.conf_parser.parse_args()
			for file in args['file']:
				if (allowed_conf_file(file.filename)):
					try:
						file.save(os.path.join(conf["global"]["paths"]["conf"][project], secure_filename(file.filename)))
						response["upload_status"][file.filename]="ok"
					except:
						response["upload_status"][file.filename]=err()
				else:
					response["upload_status"][file.filename]="extension not allowed"
				read_conf()
				response["yaml_validator"]=conf["global"]["projects"][project]
			return response
		else:
			api.abort(403)

	def put(self,project):
		'''create a project'''
		if (project == "conf"):
			api.abort(403)
		elif project in conf["global"]["projects"].keys():
			api.abort(400, 'project "{}" already exists'.format(project))
		else:
			try:
				dirname=os.path.join(conf["global"]["paths"]["projects"],project)
				os.mkdir(dirname)
				os.mkdir(os.path.join(dirname,'recipes'))
				os.mkdir(os.path.join(dirname,'datasets'))
				read_conf()
				return {"message": "{} successfully created".format(project)}
			except:
				api.abort(400,err())

	def delete(self,project):
		'''delete a project'''
		if (project == "conf"):
			api.abort(403)
		elif project in conf["global"]["projects"].keys():
			response={project: "not deleted"}
			try:
				dirname=os.path.join(conf["global"]["paths"]["projects"],project)
				shutil.rmtree(dirname)
				response[project]="deleted"
			except:
				response[project]="deletion failed - "+err()
			read_conf()
			#response["yaml_validator"]=conf["global"]["projects"][project]
			return response
		else:
			api.abort(404)

@api.route('/conf/<project>/<path:file>', endpoint='conf/<project>/<path:file>')
class FileConf(Resource):
	def get(self,project,file):
		'''get a text/yaml configuration file from project'''
		try:
			read_conf()
			if (file in conf["global"]["projects"][project]["files"]):
				try:
					pfile=os.path.join(conf["global"]["projects"][project]["path"],file)
					with open(pfile) as f:
						return Response(f.read(),mimetype="text/plain")
				except:
					api.abort(404)
			else:
				api.abort(404)
		except:
			api.abort(404)

	def delete(self,project,file):
		'''delete a text/yaml configuration file from project'''
		if (project != "conf"):
			if (file in conf["global"]["projects"][project]["files"]):
				try:
					pfile=os.path.join(conf["global"]["projects"][project]["path"],file)
					os.remove(pfile)
					read_conf()
					return jsonize({"conf": project, "file":file, "status": "removed"})
				except:
					api.abort(403)

	@api.expect(parsers.yaml_parser)
	def post(self,project,file):
		'''upload a text/yaml configuration file from project'''
		if (project != "project"):
			args = parsers.yaml_parser.parse_args()
			filecontent=args['yaml']
			if (allowed_conf_file(file)):
				try:
					pfile=os.path.join(conf["global"]["projects"][project]["path"],file)
					with open(pfile,'w') as f:
						f.write(filecontent.encode("utf-8", 'ignore'))
					response={file: {"saved": "ok"}}
					read_conf()
					response[file]["yaml_validator"]=conf["global"]["projects"][project]["files"][file]
					return response
				except:
					api.abort(400,{file: {"saved" : "ko - "+err()}})
			else:
				api.abort(403)
		else:
			api.abort(403)



@api.route('/datasets/', endpoint='datasets')
class ListDatasets(Resource):
	def get(self):
		'''get json of all configured datasets'''
		return conf["datasets"]

@api.route('/datasets/<dataset>/', endpoint='datasets/<dataset>')
class DatasetApi(Resource):
	def get(self,dataset):
		'''get json of a configured dataset'''
		try:
			return conf["datasets"][dataset]
		except:
			api.abort(404)

	def post(self,dataset):
		'''get sample of a configured dataset, number of rows being configured in connector.samples'''
		ds=Dataset(dataset)
		if (ds.connector.type == "elasticsearch"):
	 		ds.select={"query":{"function_score": {"query":ds.select["query"],"random_score":{}}}}
		ds.init_reader()
		try:
			df=next(ds.reader,"").head(n=ds.connector.sample).reset_index(drop=True)
			#df.fillna('',inplace=True)
			return {"data": list(df.fillna("").T.to_dict().values())}
		except:
			return {"data":[]}

	def delete(self,dataset):
		'''delete the content of a dataset (currently only working on elasticsearch datasets)'''
		ds=Dataset(dataset)
		if (ds.connector.type == "elasticsearch"):
			try:
				ds.connector.es.indices.delete(index=ds.table, ignore=[400, 404])
				log.write("detete {}:{}/{}".format(ds.connector.host,ds.connector.port,ds.table))
				ds.connector.es.indices.create(index=ds.table)
				log.write("create {}:{}/{}".format(ds.connector.host,ds.connector.port,ds.table))
				return {"status": "ok"}
			except:
				return {"status": "ko - " + err()}
		else:
			return api.abort(403)


@api.route('/datasets/<dataset>/<action>', endpoint='datasets/<dataset>/<action>')
class pushToValidation(Resource):
	def put(self,dataset,action):
		'''action = validation : configure the frontend to point to this dataset'''
		if (action=="validation"):
			if (not(dataset in conf["datasets"].keys())):
				api.abort(404)
			if not("validation" in conf["datasets"][dataset].keys()):
				api.abort(403)
			if (conf["datasets"][dataset]["validation"]==True):
				try:
					props = {}
					for config in conf["global"]["validation"].keys():
						configfile=os.path.join(conf["global"]["paths"]["validation"],secure_filename(config+".json"))
						dic={
							"prefix": conf["global"]["api"]["prefix"],
							"domain": conf["global"]["api"]["domain"],
							"dataset": dataset
						}
						props[config] = replace_dict(conf["global"]["validation"][config],dic)
						# with open(configfile, 'w') as outfile:
						# 	json.dump(props[config],outfile,indent=2)
					return {"dataset": dataset, "status": "to validation", "props": props}
				except :
						return api.abort(500,{"dataset": dataset, "status": "error: "+err()})
			else:
				return api.abort(403,{"dataset": dataset, "status": "no validation allowed"})

		else:
			api.abort(404)

	def post(self,dataset,action):
		'''(KO) search into the dataset'''
		if (action=="_search"):
			return {"status": "in dev"}
		else:
			api.abort(403)

	def get(self,dataset,action):
		'''(KO) does nothing yet'''
		if (action=="yaml"):
			return


@api.route('/recipes/', endpoint='recipes')
class ListRecipes(Resource):
	def get(self):
		'''get json of all configured recipes'''
		return conf["recipes"]

@api.route('/recipes/<recipe>/', endpoint='recipes/<recipe>')
class RecipeApi(Resource):
	def get(self,recipe):
		'''get json of a configured recipe'''
		try:
			return conf["recipes"][recipe]
		except:
			api.abort(404)


@api.route('/recipes/<recipe>/<action>', endpoint='recipes/<recipe>/<action>')
class RecipeRun(Resource):
	def get(self,recipe,action):
		'''retrieve information on a recipe
		** action ** possible values are :
		- ** status ** : get status (running or not) of a recipe
		- ** log ** : get log of a running recipe'''
		if (action=="status"):
			#get status of job
			try:
				return {"recipe":recipe, "status": jobs[str(recipe)].job_status()}
			except:
				return {"recipe":recipe, "status": "down"}
		elif (action=="log"):
			#get logs
			try:
				# try if there is a current log
				with open(jobs[recipe].log.file, 'r') as f:
					response = f.read()
					return Response(response,mimetype="text/plain")
			except:
				try:
					# search for a previous log
					a = conf["recipes"][recipe] # check if recipe is declared
					logfiles = [os.path.join(conf["global"]["log"]["dir"],f)
								for f in os.listdir(conf["global"]["log"]["dir"])
								if re.match(r'^.*-' + recipe + '.log$',f)]
					logfiles.sort()
					file = logfiles[-1]
					with open(file, 'r') as f:
						response = f.read()
						return Response(response,mimetype="text/plain")
				except:
					api.abort(404)
		api.abort(403)

	@api.expect(parsers.live_parser)
	def post(self,recipe,action):
		'''apply recipe on posted data
		** action ** possible values are :
		- ** apply ** : apply recipe on posted data
		'''
		if (action=="apply"):
			args = parsers.live_parser.parse_args()
			file=args['file']
			if not (allowed_upload_file(file.filename)):
				api.abort(403)
			r=Recipe(recipe)
			r.input.chunked=False
			r.input.file=file.stream
			r.init(test=True)
			r.run()
			if isinstance(r.df, pd.DataFrame):
				df=r.df.fillna("")
				try:
					return jsonize({"data": df.T.to_dict().values(), "log": str(r.log.writer.getvalue())})
				except:
					df=df.applymap(lambda x: str(x))
					return jsonize({"data": df.T.to_dict().values(), "log": str(r.log.writer.getvalue())})
			else:
				return {"log": r.log.writer.getvalue()}


	def put(self,recipe,action):
		'''test, run or stop recipe
		** action ** possible values are :
		- ** test ** : test recipe on sample data
		- ** run ** : run the recipe
		- ** stop ** : stop a running recipe (soft kill : it may take some time to really stop)
		'''
		if (action=="test"):
			try: 
				read_conf()
				result = manager.dict()
				r=Recipe(recipe)
				r.init(test=True)
				r.set_job(Process(target=thread_job,args=[r, result]))
				r.start_job()
				r.join_job()
				r.df = result["df"]
				r.log = result["log"]
			except:
				return {"data": [{"result": "empty"}], "log": "Ooops: {}".format(err())}
			if isinstance(r.df, pd.DataFrame):
				df=r.df.fillna("")
				try:
					return jsonize({"data": df.T.to_dict().values(), "log": result["log"]})
				except:
					df=df.applymap(lambda x: str(x))
					return jsonize({"data": df.T.to_dict().values(), "log": result["log"]})
			else:
				return {"data": [{"result": "empty"}], "log": result["log"]}
		elif (action=="run"):
			#run recipe (gives a job)
			try:
				if (recipe in list(jobs.keys())):
					status=jobs[recipe].job_status()
					if (status=="up"):
						return {"recipe": recipe, "status": status}
			except:
				api.abort(403)

			jobs[recipe]=Recipe(recipe)
			jobs[recipe].init()

			# jobs[recipe].set_job(threading.Thread(target=thread_job,args=[jobs[recipe]]))
			jobs[recipe].set_job(Process(target=thread_job,args=[jobs[recipe]]))
			jobs[recipe].start_job()
			return {"recipe":recipe, "status": "new job"}
		elif (action=="stop"):
			try:
				if (recipe in list(jobs.keys())):
					jobs[recipe].stop_job()
					return {"recipe": recipe, "status": "stopped"}
			except:
				api.abort(404)


@api.route('/jobs/', endpoint='jobs')
class jobsList(Resource):
	def get(self):
		'''retrieve jobs list
		'''
		# response = jobs.keys()
		response = {"running": {}, "done": {}}
		for recipe, job in jobs.iteritems():
			status = job.job_status()
			if (status != "down"):
				response["running"][recipe] = { "status": status,
												"file": re.sub(r".*/","", job.log.file),
												"date": re.search("(\d{4}.?\d{2}.?\d{2}T?.*?)-.*.log",job.log.file,re.IGNORECASE).group(1)
											  }

		logfiles = [f
							for f in os.listdir(conf["global"]["log"]["dir"])
							if re.match(r'^.*.log$',f)]
		for file in logfiles:
			recipe = re.search(".*-(.*?).log", file, re.IGNORECASE).group(1)
			date = re.search("(\d{4}.?\d{2}.?\d{2}T?.*?)-.*.log", file, re.IGNORECASE).group(1)
			if (recipe in conf["recipes"].keys()):
				try:
					if (response["running"][recipe]["date"] != date):
						try:
							response["done"][recipe].append({"date": date, "file": file})
						except:
							response["done"][recipe]=[{"date": date, "file": file}]
				except:
					try:
						response["done"][recipe].append({"date": date, "file": file})
					except:
						response["done"][recipe]=[{"date": date, "file": file}]

		return response

if __name__ == '__main__':
	read_conf()
	app.config['DEBUG'] = conf["global"]["api"]["debug"]

	log=Log("main")

	# recipe="dataprep_snpc"
	# r=Recipe(recipe)
	# r.init()
	# r.run()

    # Load a dummy app at the root URL to give 404 errors.
    # Serve app at APPLICATION_ROOT for localhost development.
	application = DispatcherMiddleware(Flask('dummy_app'), {
		app.config['APPLICATION_ROOT']: app,
	})
	run_simple(conf["global"]["api"]["host"], conf["global"]["api"]["port"], application, use_reloader=conf["global"]["api"]["use_reloader"])
	#app.run(host='localhost', port=8080)
