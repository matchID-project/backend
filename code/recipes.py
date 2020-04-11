#!/usr/bin/env python3
# -*- coding: utf-8 -*-


### import basics
import sys
import os
import io
import re
import datetime
import hashlib
import unicodedata
import shutil
import csv
import boto3
from smart_open import open

from werkzeug.utils import secure_filename
from io import StringIO

import traceback
import yaml as y
import json
import itertools
import time
import operator
import simplejson
from collections import Iterable
from collections import OrderedDict
from pandas.io.json import json_normalize
from collections import deque


# interact with datasets
import gzip

#from pandasql import sqldf
import elasticsearch
from elasticsearch import Elasticsearch, helpers
from sqlalchemy import create_engine, Column, Integer, Sequence, String, Date, Float, BIGINT
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import redisearch as rs
import pandas as pd

# parallelize
# import concurrent.futures
#import threading
from multiprocessing import Process, Queue, current_process

import uuid

# datascience dependecies
# dataprep with pandas
import random
import numpy as np
# ml dependencies
from sklearn.utils import shuffle
import sklearn.ensemble
import networkx as nx
# from graph_tool.all import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
import joblib
from numpy import array

# matchID imports
import config
from config import ordered_load, deepupdate, Configured
from log import Log, err
import automata
from tools import *

def fwf_format(row, widths, sep=""):
    return sep.join([row[col].ljust(widths[i] - len(sep)) for i, col in enumerate(row.keys())])


def to_fwf(df, fname, widths=None, sep="", header=False, names=None, append=False, encoding="utf8", log=None):
    if (log == None):
        wlog = log
    mode = "w"
    try:
        wdf = df[names]
        if (sep == None):
            sep = ""
        wdf = wdf.apply(lambda row: fwf_format(row, widths, sep), axis=1)
        if header:
            header = sep.join([str(col).ljust(widths[i] - len(sep))
                               for i, col in enumerate(names)])
        else:
            header = None
        if append == True:
            mode = "a"
        with open(fname, mode) as f:
            if (header == None):
                np.savetxt(f, wdf.values, fmt="%s")
            else:
                np.savetxt(f, wdf.values, header=header.encode(
                    encoding), fmt="%s", comments="")
        return
    except:
        if (log != None):
            log.write(error=err())
        else:
            raise


class Connector(Configured):

    def __init__(self, name=None):
        Configured.__init__(self, "connectors", name)

        try:
            self.thread_count = self.conf["thread_count"]
        except:
            self.thread_count = 1

        try:
            self.type = self.conf["type"]
        except:
            sys.exit(
                "Ooops: type of connector {} has to be defined".format(self.name))

        if (self.type == "mongodb"):
            try:
                self.database = self.conf["database"]
            except:
                sys.exit("Ooops: database of {} connector {} has to be defined".format(
                    self.type, self.name))
        if (self.type == "filesystem"):
            try:
                self.directory = self.conf["directory"]
            except:
                sys.exit("Ooops: directory of {} connector {} has to be defined".format(
                    self.type, self.name))

        if (self.type == "s3"):
            try:
                self.bucket = self.conf["bucket"]
            except:
                sys.exit("Ooops: bucket of {} connector {} has to be defined".format(
                    self.type, self.name))
            try:
                self.aws_access_key_id = self.conf["aws_access_key_id"]
            except:
                self.aws_access_key_id = os.getenv('aws_access_key_id')
            try:
                self.aws_secret_access_key = self.conf["aws_secret_access_key"]
            except:
                self.aws_secret_access_key = os.getenv('aws_secret_access_key')
            try:
                self.endpoint_url = self.conf["endpoint_url"]
            except:
                self.endpoint_url = None
            try:
                self.signature_version = self.conf["signature_version"]
            except:
                self.signature_version = None
            try:
                self.region_name = self.conf["region_name"]
            except:
                self.region_name = None
            self.s3_session = boto3.Session(
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.region_name
                )
            #self.client = session.client('s3', endpoint_url=self.endpoint_url)
            self.s3_resource =  self.s3_session.resource('s3', endpoint_url =  self.endpoint_url)
            self.transport_params={
                "session": self.s3_session,
                "resource_kwargs": {"endpoint_url": self.endpoint_url}
            }

        try:
            self.timeout = self.conf["timeout"]
        except:
            self.timeout = 10

        if (self.type == "elasticsearch") | (self.type == "mongodb"):
            try:
                self.host = self.conf["host"]
            except:
                sys.exit("Ooops: host of connector {} has to be defined as type is {}".format(
                    connector_name, self.type))

        if (self.type == "elasticsearch"):
            try:
                self.port = self.conf["port"]
            except:
                self.port = 9200
            try:
                self.scheme = self.conf["scheme"]
            except:
                self.scheme = "http"
            self.es = Elasticsearch(
                self.host, port=self.port, scheme=self.scheme, timeout=self.timeout)
            try:
                self.chunk_search = self.conf["chunk_search"]
            except:
                self.chunk_search = 1000

            try:
                self.max_tries = self.conf["max_tries"]
            except:
                self.max_tries = 2

            try:
                self.safe = self.conf["safe"]
            except:
                self.safe = True

        try:
            self.chunk = self.conf["chunk"]
        except:
            self.chunk = 1000

        try:
            self.sample = self.conf["sample"]
        except:
            self.sample = 500

        if (self.type == "sql"):
            try:
                self.encoding = self.conf["encoding"]
            except:
                self.encoding = "UTF8"
            self.uri = self.conf["uri"]
            self.sql = create_engine(self.uri, encoding=self.encoding)

        if (self.type == "redisearch"):
            try:
                self.host = self.conf["host"]
            except:
                self.host = "redis"
            try:
                self.port = self.conf["port"]
            except:
                self.port = 6379
            try:
                self.batchIndexer = self.conf["batchIndexer"]
            except:
                self.batchIndexer = 100000

class Dataset(Configured):
    # a dataset is mainly a table linked to a pandas dataframe

    def __init__(self, name=None, parent=None):

        try:
            Configured.__init__(self, "datasets", name)
        except:
            if (self.name == "inmemory"):
                self.connector = {"type": "inmemory", "chunk": 10000}
                self.filter = None
                return
            else:
                config.log.write(error="no conf for dataset {}".format(self.name))

        try:
            self.parent = parent
        except:
            pass

        self.filter = None

        try:
            self.connector = Connector(self.conf["connector"])
        except:
            config.log.write(msg="failed to initiate connector for dataset {}".format(
                self.name), error=err())

        try:
            self.table = self.conf["table"]
        except:
            config.log.write(
                error="table of dataset {} has to be defined".format(self.name))

        try:
            self.encoding = self.conf["encoding"]
        except:
            self.encoding = "utf8"

        try:
            self.thread_count = self.conf["thread_count"]
        except:
            self.thread_count = self.connector.thread_count

        try:
            self.chunk = self.conf["chunk"]
        except:
            self.chunk = self.connector.chunk

        if (self.connector.type == "redisearch"):
            try:
                self.index = self.conf["index"]
            except:
                self.index = []

        if (self.connector.type == "sql"):
            try:
                self.select = self.conf["select"].replace('\n', ' ')
            except:
                self.select = "select * from {}".format(self.table)
            try:
                self.mode = self.conf["mode"]
            except:
                self.mode = 'basic'

        if (self.connector.type == "elasticsearch"):
            try:
                self.select = self.conf["select"]
            except:
                self.select = {"query": {"match_all": {}}}
            # self.select={"query":{"function_score":{"query":{"match_all":{}},"functions":[{"random_score":{}}]}}}
            try:
                self.doc_type = self.conf["doc_type"]
            except:
                self.doc_type = self.name

            try:
                self.random_view = self.conf["random_view"]
            except:
                self.random_view = True

            try:
                self.body = json.loads(json.dumps(self.conf["body"]))
            except:
                self.body = {}

            # inherited properties from connector, than can be overrided
            try:
                self.max_tries = self.conf["max_tries"]
            except:
                self.max_tries = self.connector.max_tries

            try:
                self.safe = self.conf["safe"]
            except:
                self.safe = self.connector.safe

            try:
                self.timeout = self.conf["timeout"]
            except:
                self.timeout = self.connector.timeout

            try:
                self.chunk_search = self.conf["chunk_search"]
            except:
                self.chunk_search = self.connector.chunk_search

        if (self.connector.type == "filesystem"):
            self.select = None
            try:
                self.files = [os.path.join(self.connector.directory, f)
                              for f in os.listdir(self.connector.directory)
                              if re.match(r'^' + self.table['regex'] + '$', f)]
                self.file = self.files[0]
            except:
                self.file = os.path.join(self.connector.directory, self.table)
                self.files = [self.file]
                #log.write("Ooops: couldn't set filename for dataset {}, connector {}".format(self.name,self.connector.name),exit=True)

        if (self.connector.type == "s3"):
            self.select = None
            if (type(self.table) == str):
                self.file = "s3://" + os.path.join(self.connector.bucket,self.table)
                self.files = [self.file]
            else:
                try:
                    self.files = ["s3://" + os.path.join(f.bucket_name,f.key)
                              for f in self.connector.s3_resource.Bucket(self.connector.bucket).objects.all()
                              if re.match(r'^' + self.table['regex'] + '$', f.key)]
                    self.file = self.files[0]
                except:
                    self.files= [str(err())]
                    # config.log.write("Ooops: couldn't match files for {} in connector {}".format(self.name,self.connector.name),exit=True)


        if (self.connector.type == "filesystem") | (self.connector.type == "s3"):
            try:
                self.skiprows = self.conf["skiprows"]
            except:
                self.skiprows = 0

            try:
                self.type = self.conf["type"]
            except:
                self.type = "csv"

            try:
                self.prefix = self.conf["prefix"]
            except:
                self.prefix = None

            try:
                self.header = self.conf["header"]
            except:
                if (self.type == "csv"):
                    self.header = "infer"
                else:
                    self.header = False

            try:
                self.names = self.conf["names"]
            except:
                self.names = None

            try:
                self.sep = self.conf["sep"]
            except:
                if (self.type == "csv"):
                    self.sep = ";"
                else:
                    self.sep = None

            try:
                self.compression = self.conf["compression"]
            except:
                self.compression = 'infer'

            try:
                self.skiprows = self.conf["skiprows"]
            except:
                self.skiprows = 0

            try:
                self.widths = self.conf["widths"]
            except:
                self.widths = [1000]

            try:
                quoting = self.conf["quoting"]
                if (quoting == 'QUOTE_MINIMAL'):
                    self.quoting = csv.QUOTE_MINIMAL
                elif (quoting == 'QUOTE_NONE'):
                    self.quoting = csv.QUOTE_NONE
                elif (quoting == 'QUOTE_ALL'):
                    self.quoting = csv.QUOTE_ALL
                elif (quoting == 'QUOTE_NONNUMERIC'):
                    self.quoting = csv.QUOTE_NONNUMERIC
            except:
                self.quoting = csv.QUOTE_MINIMAL

    def init_reader(self, df = None, test = False, test_chunk_size=None):
        try:
            self.log = self.parent.log
        except:
            self.log = config.log
        try:
            test_chunk_size = self.parent.conf["test_chunk_size"]
        except:
            test_chunk_size = config.conf["global"]["test_chunk_size"]

        if True:
            if (self.name == "inmemory"):
                if (df is not None):
                    self.reader = [
                        dict({
                            "desc": {
                                "source": {
                                    "type": "inmemory",
                                    "name": "inmemory"
                                },
                                "chunk": {"id": 0}
                            },
                            "df": df
                            })
                        ]
                else:
                    if((len(self.before) + len(self.after)) == 0):
                        self.log.write(
                            error="can't initiate inmemory dataset with no dataframe", exit=True)
                    else:
                        self.reader = []
            elif (self.connector.type == "filesystem") | (self.connector.type == "s3"):
                self.reader = self.iterator_from_files()
            elif (self.connector.type == "elasticsearch"):
                self.reader = self.scanner()
            elif (self.connector.type == "sql"):
                if (self.select == None):
                    self.select = "select * from {}".format(self.table)
                if self.mode == 'expert':
                    fbuf = StringIO()
                    if test:
                        table = r"(^|\s+|\()" + re.escape(self.table) + r"(\s+|\.|\)|$)"
                        query = 'WITH MATCHID_INPUT_TABLE AS (SELECT * from {} limit {})\n'.format(self.table, test_chunk_size)
                        query = query + re.sub(table, r'\1MATCHID_INPUT_TABLE\2', self.select, flags=re.IGNORECASE)
                        query = "COPY (select * from (\n{}) query limit {}) TO STDOUT WITH (FORMAT CSV, HEADER TRUE, DELIMITER '\x01')".format(query, test_chunk_size)
                    else:
                        query = "COPY (\n{}) TO STDOUT WITH (FORMAT CSV, HEADER TRUE, DELIMITER '\x01')".format(self.select)
                    self.log.write("execute statement using expert mode (WARNING: all typed columns will be converted to string):\n{}".format(query))
                    self.connector.sql.raw_connection().cursor().copy_expert(query, fbuf)
                    fbuf.seek(0)
                    self.reader = pd.read_csv(fbuf, sep='\x01', dtype=object, encoding="utf8", iterator=True, keep_default_na=False, chunksize=self.chunk)
                else:
                    if test:
                        table = r"(^|\s+|\()" + re.escape(self.table) + r"(\s+|\.|\)|$)"
                        query = 'WITH MATCHID_INPUT_TABLE AS (SELECT * from {} limit {})\n'.format(self.table, test_chunk_size)
                        query = query + re.sub(table, r'\1MATCHID_INPUT_TABLE\2', self.select, flags=re.IGNORECASE)
                        query = "select * from (\n{}) query limit {}".format(query, test_chunk_size)
                    else:
                        query = self.select
                    self.log.write("execute statement:\n{}".format(query))
                    self.reader = pd.read_sql(query, self.connector.sql, chunksize=self.chunk)


            try:
                if (self.filter != None):
                    self.filter.init(df=pd.DataFrame(), parent=self)
                    self.reader = map(
                        lambda data: dict({
                            "desc": data['desc'],
                            "df": self.filter.run_chunk(data['chunk']['id'], data['df'], data['desc'])}),
                        self.reader)
                    self.log.write(msg="filtered dataset")
            except:
                self.log.write(
                    msg="failed to initiate the filter", error=err())

        else:
            self.log.write(msg="couldn't initiate dataset {}".format(
                self.name), error=err(), exit=True)

    def iterator_from_files(self):
        if (self.connector.type == "filesystem") | (self.connector.type == "s3"):
            n_chunk_total = 0
            n_rows_total = 0
            for n_file, file in enumerate(self.files):
                n_rows_file = 0
                if (self.type == "csv"):
                    reader = pd.read_csv(
                        self.open(file),
                        sep=self.sep, usecols=self.select, chunksize=self.chunk,
                        encoding=self.encoding,dtype=object, header=self.header, names=self.names, skiprows=self.skiprows,
                        prefix=self.prefix, iterator=True, index_col=False, keep_default_na=False
                    )
                elif (self.type == "fwf"):
                    reader = pd.read_fwf(
                        self.open(file),
                        chunksize=self.connector.chunk, skiprows=self.skiprows,
                        encoding=self.encoding,
                        delimiter=self.sep, compression='infer', dtype=object, names=self.names, widths=self.widths,
                        iterator=True, keep_default_na=False
                    )
                elif (self.type == "hdf"):
                    reader = pd.read_hdf(
                        self.open(file),
                        chunksize=self.chunk
                    )
                elif (self.type == "msgpack"):
                    reader = self.iterator_to_chunked_iterator(
                        pd.read_msgpack(
                            self.open(file),
                            iterator=True, encoding=self.encoding
                        )
                    )
                for n_chunk_file, df in enumerate(reader):
                    size = df.shape[0]
                    n_rows_total = n_rows_total + size
                    n_rows_file = n_rows_file + size
                    # print(df.head())
                    # sys.stdout.flush()
                    yield dict({ "desc": {
                            "source": {
                                "dataset": self.name,
                                "type": "file",
                                "name": file,
                                "file_id": n_file + 1,
                                "chunk_size": self.chunk
                            },
                            "chunk": {
                                "id": n_chunk_total + 1,
                                "local_id": n_chunk_file + 1,
                                "rows": size
                            },
                            "sum": {
                                "rows": n_rows_total,
                                "file_rows": n_rows_file
                            },
                            "total": {
                                "files": len(self.files)
                            }
                        },
                        "df": df
                    })
                    n_chunk_total = n_chunk_total + 1

    def open(self, file, mode='rb'):
        if (self.connector.type == "s3"):
            return open(file, mode, transport_params=self.connector.transport_params)
        return open(file, mode)

    def iterator_to_chunked_iterator(self, iterator):
        # force to regroup in chunks iterators that are only line by line
        df_list=[]
        current_size = 0
        for j, df in enumerate(iterator):
            while (current_size + df.shape[0] > self.chunk):
                df_list.append(df.head(n = self.chunk - current_size))
                yield pd.concat(df_list)
                df_list = []
                current_size = 0
                df = df.tail(n = df.shape[0] - (self.chunk - current_size))
                if (current_size > 0):
                    df_list.append(df.tail(n = current_size))
            if (df.shape[0] > 0):
                df_list.append(df)
                current_size = current_size + df.shape[0]
        if (current_size > 0):
            yield pd.concat(df_list)

    def scanner(self, **kwargs):
        self.select = json.loads(json.dumps(self.select))
        scan = helpers.scan(client=self.connector.es, scroll='1000m', clear_scroll=False, query=self.select,
                            index=self.table, preserve_order=True, size=self.chunk)

        hits = []
        ids = []
        labels = ['_id', '_source']
        c = 0
        s = 0
        for j, item in enumerate(scan):
            item = (item['_id'], item['_source'])
            hits.append(item)

            if (((j + 1) % self.chunk) == 0):
                df = pd.DataFrame.from_records(hits, columns=labels)
                df = pd.concat(
                    [df['_id'], df['_source'].apply(pd.Series)], axis=1)
                hits = []
                ids = []
                size = df.shape[0]
                s = s + size
                yield dict({
                    "desc": {
                        "source": {
                            "type": "elasticsearch",
                            "name": self.table,
                            "chunk_size": self.chunk
                        },
                        "chunk": {
                            "id": c + 1,
                            "rows": size
                        },
                        "sum": {
                            "rows": s
                        }
                    },
                    "df": df
                })
                c = c + 1
        if (len(hits) > 0):
            df = pd.DataFrame.from_records(hits, columns=labels)
            df = pd.concat([df['_id'], df['_source'].apply(pd.Series)], axis=1)
            yield dict({
                "source": {
                    "type": "elasticsearch",
                    "name": self.table
                },
                "chunk": {
                    "total": c,
                    "rows": df.shape[0]
                },
                "df": df
            })

    def init_writer(self):
        try:
            self.log = self.parent.log
        except:
            self.log = config.log

        # currently only manage elasticsearch injection
        if (self.name == "inmemory"):
            return
        elif (self.connector.type == "elasticsearch"):
            # ignore 400 cause by IndexAlreadyExistsException when creating an
            # index
            try:
                if (self.mode == 'create'):
                    self.connector.es.indices.delete(
                        index=self.table, ignore=[400, 404])
                    self.log.write(
                        "detete {}:{}/{}".format(self.connector.host, self.connector.port, self.table))
                    self.connector.es.indices.create(
                        index=self.table, body=self.body)
                    self.log.write(
                        "create {}:{}/{}".format(self.connector.host, self.connector.port, self.table))
            except:
                self.log.write(msg="problem while initiating elasticsearch index {} for dataset {}".format(
                    self.table, self.name), error=err(), exit=True)
        elif (self.connector.type == "filesystem"):
            if (self.mode == 'create'):
                try:
                    os.remove(self.file)
                except:
                    # further better except should make difference btw no
                    # existing file and unwritable
                    pass
                self.fs = open(self.file, 'wb')
            else:
                self.fs = open(self.file, 'ab')
            self.log.write(msg="initiated stream output {}".format(self.name))
        elif (self.connector.type == "s3"):
            try:
                self.log.write(msg="tries to remove {}".format(self.file))
                self.connector.s3_resource.Object(self.connector.bucket, self.file.replace(self.connector.bucket + '/', '')).delete()
                self.log.write(msg="removed successfully {}".format(self.file))
            except:
                self.log.write(msg="couldn't remove {} {}".format(self.file,err()))
                # further better except should make difference btw no
                # existing file and unwritable
                pass
            self.fs = open(self.file,
                'wb', transport_params=self.connector.transport_params)
            self.log.write(msg="initiated stream output {}".format(self.name))

        elif (self.connector.type == "sql"):
            if (self.mode == 'create'):
                self.connector.sql.execute(
                    'DROP TABLE IF EXISTS {};'.format(self.table))

        elif (self.connector.type == "redisearch"):
            self.client = rs.Client(self.table, host=self.connector.host, port=self.connector.port)
            if (self.mode == 'create'):
                try:
                    self.client.drop_index()
                    self.log.write(msg="DROP previously created index {}".format(self.table))
                except:
                    pass

        return None

    def write(self, chunk=0, df=None):
        size = df.shape[0]
        if (self.name == "inmemory"):
            return size
        processed = 0
        i = 0
        if (size <= self.chunk):
            df_list = [df]
        else:
            df_list = np.array_split(
                df, list(range(self.chunk, size, self.chunk)))
        for df in df_list:
            i += 1
            size = df.shape[0]

            if (self.connector.type == "elasticsearch"):
                df = df.fillna("")
                if (self.connector.safe == False) & ('_id' not in df.columns) & (self.mode == 'create'):
                        # unsafe insert speed enable to speed up
                    actions = [{'_op_type': mode, '_index': self.table, '_source': dict(
                        (k, v) for k, v in records[it].items() if (v != ""))} for it in records]
                else:
                    if ('_id' not in df.columns):
                        df['_id'] = df.apply(lambda row: sha1(row), axis=1)
                    records = df.drop(['_id'], axis=1).T.to_dict()
                    ids = df['_id'].T.to_dict()
                    if (self.mode == "update"):
                        actions = [{'_op_type': 'update', '_id': ids[it], '_index': self.table, 'doc_as_upsert': True, 'doc': dict(
                            (k, v) for k, v in records[it].items() if (v != ""))} for it in records]
                    else:
                        actions = [{'_op_type': 'index', '_id': ids[it], '_index': self.table, '_source': dict(
                            (k, v) for k, v in records[it].items() if (v != ""))} for it in records]
                try:
                    tries = 0
                    success = False
                    failure = None
                    max_tries = self.max_tries
                    #self.log.write("try to instert subchunk {} of {} lines to {}/{}".format(i,size,self.connector.name,self.table))
                    while(tries < max_tries):
                        try:
                            # if (self.connector.thread_count>1):
                            # 	deque(helpers.parallel_bulk(self.connector.es,actions,thread_count=self.connector.thread_count))
                            # else:
                            helpers.bulk(self.connector.es, actions)
                            max_tries = tries
                            success = True
                        except elasticsearch.SerializationError:
                            error = err()
                            if ('"errors":false' in error):
                                # processed+=size
                                tries += 1
                                # prevents combo deny of service of
                                # elasticsearch
                                time.sleep(random.random() * (4 ** tries))
                            elif (('JSONDecodeError' in error) & (not (re.match('"failed":[1-9]', error)))):
                                max_tries = tries
                                self.log.write(
                                    msg="elasticsearch JSONDecodeError but found no error")
                                # processed+=size
                            else:
                                max_tries = tries
                        except elasticsearch.ConnectionTimeout:
                            error = err()
                            tries += 1
                            # prevents combo deny of service of elasticsearch
                            time.sleep(random.random() * (4 ** tries))
                            #self.log.write("elasticsearch bulk ConnectionTimeout warning {}/{}".format(self.connector.name,self.table))
                        except elasticsearch.helpers.BulkIndexError:
                            error = err()
                            if ('es_rejected_execution_exception' in error):
                                tries += 1
                                # prevents combo deny of service of
                                # elasticsearch
                                time.sleep(random.random() * (4 ** tries))
                        except:
                            tries = max_tries
                            error = err()
                    if (success == False):
                        self.log.write(msg="elasticsearch bulk of subchunk {} failed after {} tries {}/{}".format(
                            i, tries, self.connector.name, self.table), error=error)
                    else:
                        processed += size
                        if (tries > 0):
                            self.log.write("inserted subchunk {}, {} lines to {}/{} on {}th try".format(
                                i, size, self.connector.name, self.table, tries + 1))
                        else:
                            self.log.write(
                                "inserted subchunk {}, {} lines to {}/{}".format(i, size, self.connector.name, self.table))
                except:
                    self.log.write(msg="elasticsearch bulk of subchunk {} failed {}/{}".format(
                        i, self.connector.name, self.table), error=err())

            elif (self.connector.type in ["filesystem", "s3"]):
                # self.log.write("filesystem write {}".format(self.name))
                if (self.type == "csv"):
                    try:
                        if self.compression == 'infer':
                            self.compression = None
                        if (chunk == 0):
                            header = self.header
                        else:
                            header = False
                        if (self.names != None):
                            df=df[self.names]
                        else:
                            df.sort_index(axis=1, inplace=True)

                        df.to_csv(
                            self.fs,
                            mode='a', index=False, sep=self.sep, quoting=self.quoting,
                            encoding=self.encoding, header=header)
                    except:
                        self.log.write(
                            "write to csv failed writing {}".format(self.file), err())
                elif (self.type == "fwf"):
                    if (chunk == 0):
                        header = self.header
                    else:
                        header = False
                    try:
                        to_fwf(df,
                            self.fs,
                            names=self.names, header=header, sep=self.sep,
                            widths=self.widths, append=True, encoding=self.encoding, log=self.log)
                    except:
                        self.log.write(
                            "write to fwf failed writing {}".format(self.file), err())
                    pass
                elif (self.type == "hdf"):
                    try:
                        df.to_hdf(
                            self.fs,
                            key=self.name,
                            mode='a', format='table')
                    except:
                        self.log.write(
                            "write to hdf failed writing {}".format(self.file), err())
                elif (self.type == "msgpack"):
                    try:
                        df.to_msgpack(
                            self.fs,
                            append=True,
                            encoding=self.encoding)
                    except:
                        self.log.write(
                            "write to msgpack failed writing {}".format(self.file), err())
                else:
                    self.log.write("no method for writing to {} with type {}".format(
                        self.file, self.type))
                if self.connector.type == "filesystem":
                    self.fs.flush()
            elif (self.connector.type == "sql"):
                try:
                    self.log.write(
                        msg="try to write {} rows".format(df.shape[0]))
                    df.sort_index(axis=1, inplace=True)
                    if not self.connector.sql.has_table(self.table):
                       df[:0].to_sql(name=self.table, con=self.connector.sql, if_exists='fail', index=False)
                       self.log.write(msg="created table {} with columns {}".format(self.table, list(df)))
                    try:
                        sql_cnxn = self.connector.sql.raw_connection()
                        cursor = sql_cnxn.cursor()
                        fbuf = StringIO()
                        df.to_csv(fbuf, index=False, header=False, sep='\x01', encoding="utf8")
                        fbuf.seek(0)
                        cursor.copy_expert("COPY " + self.table + " FROM STDIN WITH DELIMITER '\x01' NULL ''", fbuf)
                        sql_cnxn.commit()
                        cursor.close()
                    except:
                        self.log.write(msg="WARNING: direct COPY to {} method failed, fall back to to_sql\n{}".format(self.table, err()))
                        self.log.reject(fbuf.getvalue())
                        df.to_sql(name=self.table, con=self.connector.sql,
                            if_exists='append', index=False, chunksize=self.chunk)

                except:
                    self.log.write(msg="couldn't write to {}".format(self.table),
                        error=err())

            elif (self.connector.type == "redisearch"):
                self.log.write(
                    msg="try to write {} rows".format(df.shape[0]))
                df.sort_index(axis=1, inplace=True)
                try:
                    self.client.create_index([rs.TagField(col, separator = ' ', no_index=(True if col in self.index else False)) for col in list(set(list(df)))])
                    self.client.batch_indexer(chunk_size=self.batchIndexer)
                    self.log.write(
                        msg="created table {} with index for {}".format(self.table,", ".join([col for col in list.df and col in self.inx])))
                except:
                    self.log.write(msg="couldn't create {}".format(self.table),
                        error=err())

                try:
                    for index, row in df.iterrows():
                        doc_id = size * chunk + self.chunk * i + index
                        row = row.to_dict()
                        self.client.add_document(doc_id, **row)
                except:
                    self.log.write(msg="couldn't write to {}".format(self.table),
                        error=err())

        return processed

    def close(self):
        if self.connector.type in ["filesystem", "s3"]:
            self.fs.close()

class Recipe(Configured):

    def __init__(self, name=None, args={}):
        try:
            Configured.__init__(self, "recipes", name)
            self.type = "configured"
            self.args = args
            self.errors = 0
            self.log = None
        except:
            if (hasattr(self.__class__, "internal_" + name) and callable(getattr(self.__class__, "internal_" + name))):
                self.input = Dataset("inmemory", parent=self)
                self.input.select = None
                self.steps = []
                self.output = Dataset("inmemory", parent=self)
                self.type = "internal"
                self.before = []
                self.after = []
                self.head = None
                self.log = None
                self.args = args
                return
            else:
                self.log = config.log
                self.log.write(error="can't couldn't find recipe {} in conf and no internal_{} function".format(
                    self.name, self.name), exit=True)

        # initiate input connection : creater a reader or use inmemory dataset
        try:
            if ("input" in list(self.args.keys())):
                self.input = Dataset(self.args.input, parent=self)
            elif (type(self.conf["input"]) == str):
                self.input = Dataset(self.conf["input"], parent=self)
            else:
                self.input = Dataset(self.conf["input"][
                                     "dataset"], parent=self)

            try:
                if (isinstance(self.conf["input"]["select"], list)):
                    self.input.select = [str(x) for x in self.conf[
                        "input"]["select"]]
                else:
                    self.input.select = self.conf["input"]["select"]
            except:
                self.input.select = None

            try:
                r = list(self.conf["input"]["filter"].keys())[0]
                self.input.filter = Recipe(name=r, args=self.conf[
                                           "input"]["filter"][r])
            except:
                self.input.filter = None

            try:
                self.input.chunked = self.conf["input"]["chunked"]
            except:
                self.input.chunked = True

            try:
                self.input.chunk = self.conf["input"]["chunk"]
            except:
                pass

            try:
                self.input.thread_count = self.conf["input"]["thread_count"]
            except:
                pass

            try:
                self.input.max_tries = self.conf["input"]["max_tries"]
            except:
                pass

        except:
            self.input = Dataset("inmemory", parent=self)
            self.input.select = None
            self.input.chunked = True

        try:
            self.threads = self.conf["threads"]
        except:
            try:
                self.threads = config.conf["global"]["threads_by_job"]
            except:
                self.threads = 1

        try:
            if ("before" in list(self.conf.keys())):
                self.before = self.conf["before"]
            elif ("run" in list(self.conf.keys())):
                self.before = self.conf["run"]
            else:
                self.before = []
        except:
            self.before = []

        try:
            self.after = self.conf["after"]
        except:
            self.after = []

        # initiate output connection : create a writer or use current dataframe
        try:
            if ("output" in list(self.args.keys())):
                self.output = Dataset(self.args.output, parent=self)
            elif (type(self.conf["output"]) == str):
                self.output = Dataset(self.conf["output"], parent=self)
            else:
                self.output = Dataset(self.conf["output"][
                                      "dataset"], parent=self)
            try:
                # mode can be 'create, append, update'
                self.output.mode = self.conf["output"]["mode"]
            except:
                self.output.mode = 'create'

            try:
                self.output.chunk = self.conf["output"]["chunk"]
            except:
                pass

            try:
                self.output.thread_count = self.conf["output"]["thread_count"]
            except:
                pass

            try:
                self.output.max_tries = self.conf["output"]["max_tries"]
            except:
                pass

        except:
            self.output = Dataset("inmemory", parent=self)

        try:
            self.write_queue_length = self.conf["write_queue_length"]
        except:
            try:
                self.write_queue_length = config.conf[
                    "global"]["write_queue_length"]
            except:
                self.write_queue_length = 20

        try:
            self.steps = []
            for s in self.conf["steps"]:
                function = list(s.keys())[0]
                try:
                    self.steps.append(Recipe(name=function, args=s[function]))
                except:
                    self.log.write(
                        error="recipe {} calls an unknown function {}".format(self.name, function))
        except:
            pass

    def init(self, df=None, parent=None, test=False, callback=None):
        try:
            self.callback = callback
        except:
            pass
        try:
            self.test = test
            if (parent != None):
                self.parent = parent
                self.log = self.parent.log
            else:
                self.parent = None
                self.log = Log(self.name, test=test)

        except:
            if (self.log == None):
                self.log = config.log
            self.log.write(msg="couldn't init log for recipe {}".format(
                self.name), error=err(), exit=True)
        try:
            if ((self.test == False) and (len(self.steps) == 0) and (self.input.connector.type == "sql") and (self.output.connector.type == "sql") and (self.input.select != None) and (self.input.connector.name == self.output.connector.name)):
                return
        except:
            pass

        try:
            self.input.init_reader(df=df, test=test)
        except:
            if ((len(self.before) + len(self.after)) == 0):
                self.log.write(msg="couldn't init input {} of recipe {}".format(
                    self.input.name, self.name), error=err())

    def set_job(self, job=None):
        self.job = job
        self.job.date = datetime.datetime.now().isoformat()
        return

    def start_job(self):
        self.job.start()
        return

    def join_job(self):
        self.job.join()

    def stop_job(self):
        self.job.terminate()
        time.sleep(5)
        self.job.terminate()
        return

    def job_status(self):
        try:
            if self.job.is_alive():
                return "up"
            else:
                try:
                    self.job.join()
                    self.job = None
                    return "done"
                except:
                    return "down"
        except:
            return "down"

    def write(self, i, df, supervisor=None):
        self.log.chunk = i
        if (supervisor != None):
            supervisor[i] = "writing"
        self.input.processed += self.output.write(i, df)
        if (supervisor != None):
            supervisor[i] = "done"
        self.log.write("wrote {} to {} after recipe {}".format(
            df.shape[0], self.output.name, self.name))

    def write_queue(self, queue, supervisor=None):
        exit = False
        if (self.test == False):
            try:
                self.output.init_writer()
            except:
                self.log.write(msg="couldn't init output {} of recipe {}".format(
                    self.output.name, self.name), error=err())
        w_queue = []
        try:
            max_threads = self.output.thread_count
            self.log.write("initiating queue with {} threads".format(max_threads))
        except:
            max_threads = 1
        while (exit == False):
            try:
                res = queue.get()
                if (type(res) == bool):
                    exit = True
                else:
                    #self.log.write("current queue has {} threads".format(len(w_queue)))
                    if (len(w_queue) == max_threads):
                        supervisor[res[0]] = "write_queue"
                        while (len(w_queue) == max_threads):
                            w_queue = [t for t in w_queue if (
                                t[1].is_alive() & (supervisor[t[0]] == "writing"))]
                            time.sleep(0.05)
                    supervisor[res[0]] = "writing"
                    if ((max_threads == 1) | (self.output.connector.type in ["filesystem", "s3"])):
                        # filestream can't be parallelized
                        self.write(res[0], res[1], supervisor)
                    else:
                        thread = Process(target=self.write, args=[
                                        res[0], res[1], supervisor])
                        thread.start()
                        w_queue.append([res[0], thread])
                        time.sleep(0.05)
            except:
                # self.log.write("waiting to write - {}, {}".format(self.name, w_queue))
                time.sleep(1)
                pass
        try:
            while (len(w_queue) > 0):
                w_queue = [t for t in w_queue if (
                    t[1].is_alive() & (supervisor[t[0]] == "writing"))]
        except:
            pass
        self.output.close()

    def run_chunk(self, i, df, desc=None, queue=None, supervisor=None):
        if (supervisor != None):
            supervisor[i] = "run_chunk"
        df.rename(columns=lambda x: x.strip(), inplace=True)
        # if ((self.name == "join") & (i<=config.conf["global"]["threads_by_job"]) & (i>1)):
        # 	#stupid but working hack to leave time for inmemory preload of first thread first chunk
        # 	#the limit is if the treatment of a chunk takes more than 30s... better workaround has to be found
        # 	time.sleep(30)
        self.log.chunk = desc if (desc != None) else i
        if (self.input.name != "inmemory"):
                self.log.write("proceed {} rows from {} with recipe {}".format(
                    df.shape[0], self.input.name, self.name))
        if (self.type == "internal"):
            df = getattr(self.__class__, "internal_" + self.name)(self, df=df, desc=desc)
        elif((len(self.steps) > 0) | ("steps" in list(self.conf.keys()))):
            for recipe in self.steps:
                try:
                    self.log.write("{} > {}".format(
                        self.name, recipe.name), level=2)
                    recipe.init(df=df, parent=self, test=self.test)
                    # recipe.run()
                    df = recipe.run_chunk(i, df, desc)
                    if (recipe.name == "pause"):
                        return df
                except:
                    self.log.write(msg="error while calling {} in {}".format(
                        recipe.name, self.name), error=err())
            if ((self.output.name != "inmemory") & (self.test == False)):
                if (queue != None):
                    queue.put([i, df])
                    if (supervisor != None):
                        supervisor[i] = "run_done"
                else:
                    # threads the writing, to optimize cpu usage, as write
                    # generate idle time
                    write_job = Process(target=self.write, args=[i, df])
                    write_job.start()
        return df

    def run_deps(self, recipes):
        if (len(recipes) == 0):
            return
        if(self.test == True):
            self.log.write(
                msg="no call of full run (before/run/after) in test mode - recipes to run: {}".format(recipes))
            return

        queue = []
        for recipe in recipes:
            thread = False
            if (re.sub(r'\s*\&\s*$', '', recipe) != recipe):
                recipe = re.sub(r'\s*\&\s*$', '', recipe)
                thread = True

            config.jobs[recipe] = Recipe(recipe)
            config.jobs[recipe].init(callback=config.manager.dict())
            config.jobs[recipe].set_job(
                Process(target=thread_job, args=[config.jobs[recipe]]))
            config.jobs[recipe].start_job()
            self.log.write(msg="run {}".format(recipe))
            if (thread == True):
                queue.append(config.jobs[recipe])
            else:
                config.jobs[recipe].join_job()
                config.jobs[recipe].errors = config.jobs[
                    recipe].callback["errors"]
                if (config.jobs[recipe].errors > 0):
                    self.log.write(error="Finished {} with {} errors".format(
                        recipe, jobs[recipe].errors))
                else:
                    self.log.write(
                        msg="Finished {} with no error".format(recipe))

        for r in queue:
            r.join_job()
            r.errors = config.jobs[r.name].callback["errors"]
            if (r.errors > 0):
                self.log.write(
                    error="Finished {} with {} errors".format(r.name, r.errors))
            else:
                self.log.write(msg="Finished {} with no error".format(r.name))

    def supervise(self, queue, supervisor):
        self.log.chunk = "supervisor"
        self.log.write("initiating supervisor")
        supervisor["end"] = False
        if ("supervisor_interval" in list(self.conf.keys())):
            wait = self.conf["supervisor_interval"]
            while (supervisor["end"] == False):
                try:
                    writing = len([x for x in list(supervisor.keys())
                                   if (supervisor[x] == "writing")])
                    running = len([x for x in list(supervisor.keys())
                                   if (supervisor[x] == "run_chunk")])
                    self.log.write("threads - running: {}/{} - writing: {}/{}/{} ".format(
                        running, self.threads, writing, queue.qsize(), self.output.thread_count))
                    time.sleep(wait)
                except:
                    time.sleep(0.05)
                    pass

    def run(self, head=None, write_queue=None, supervisor=None):
        desc = None
        # recipes to run before
        self.run_deps(self.before)

        if (head is None):
            try:
                head = self.conf["test_chunk_size"]
            except:
                head = config.conf["global"]["test_chunk_size"]
        #log("initiating recipe {}".format(self.name))
        self.df = pd.DataFrame()
        self.input.processed = 0
        try:
            sql_direct = False

            if ((self.test == False) and (len(self.steps) == 0) and (self.input.connector.type == "sql") and (self.output.connector.type == "sql") and (self.input.select != None) and (self.input.connector.name == self.output.connector.name)):
                sql_direct = True

            elif ((self.input.chunked == True) | (self.test == True)):
                try:
                    data = next(self.input.reader, "")
                    self.df = data['df']
                    desc = data['desc']
                    if(self.test == True):
                        self.df = self.df.head(n=head)
                except:
                    if ((len(self.before) + len(self.after)) == 0):
                        raise err()

            else:
                self.log.write("reading whole input before processing recipe")
                self.df = []
                size = 0
                for i, data in enumerate(self.input.reader):
                    df = data['df']
                    self.log.chunk = data['desc']
                    self.df.append(df)
                    size = data['desc']['rows']['chunk']
                    self.log.write(msg="loaded {} rows from {}".format(size, data['desc']['source']['name']))
                self.df = pd.concat(self.df)
                self.log.write(msg="concatenated {} rows".format(size))

            # runs the recipe
            if (self.test == True):  # test mode
                # end of work if in test mode
                print(('run : desc',desc))
                self.df = self.run_chunk(0, self.df, desc)
                return self.df
            elif (not sql_direct):
                if (supervisor == None):
                    supervisor = config.manager.dict()
                if (write_queue == None):
                    try:
                        write_queue = Queue(self.write_queue_length)
                    except:
                        self
                        write_queue = Queue()
                # create the writer queue
                try:
                    supervisor_thread = Process(target=self.supervise, args=[
                                                write_queue, supervisor])
                    supervisor_thread.start()
                except:
                    self.log.write(msg="couldn't thread supervise", error=err())
                self.log.write(msg="supervise thread initiated")

                try:
                    write_thread = Process(target=self.write_queue, args=[
                                       write_queue, supervisor])
                    write_thread.start()
                except:
                    self.log.write(msg="couldn't thread write_queue", error=err())
                self.log.write(msg="write_queue thread initiated")
                run_queue = []
                print("je suis dans run après write_queue threading")

                self.df = self.run_chunk(0, self.df, desc, write_queue, supervisor)

                for i, data in enumerate(self.input.reader):
                    df = data['df']
                    desc = data['desc']
                    supervisor[i + 1] = "started"
                    self.log.chunk = "main_thread"
                    # wait if running queue is full
                    if (len(run_queue) == self.threads):
                        supervisor[i + 1] = "run_queued"
                        while (len(run_queue) == self.threads):
                            try:
                                run_queue = [t for t in run_queue if (
                                    t[1].is_alive() & (supervisor[t[0]] == "run_chunk"))]
                            except:
                                pass
                            time.sleep(0.05)
                    supervisor[i + 1] = "run_chunk"
                    run_thread = Process(target=self.run_chunk, args=[
                                         i + 1, df, desc, write_queue, supervisor])
                    run_thread.start()
                    run_queue.append([i + 1, run_thread])

                # joining all threaded jobs
                while (len(run_queue) > 0):
                    try:
                        run_queue = [t for t in run_queue if (
                            t[1].is_alive() & (supervisor[t[0]] == "run_chunk"))]
                    except:
                        pass
                    time.sleep(0.05)
                self.log.chunk = "end"
                self.log.write("end of compute, flushing results")
                while(write_queue.qsize() > 0):
                    time.sleep(1)
                write_queue.put(True)
                write_thread.join()
                supervisor["end"] = True
                supervisor_thread.terminate()

            else:
                query = 'DROP TABLE IF EXISTS {};\nCREATE TABLE {} as select * from ({}) query;'.format(self.output.table, self.output.table, self.input.select)
                if (self.input.table not in query):
                    self.log.write(msg="input table {} not in query:{".format(self.input.table,self.input.query), err="", exit=True)
                self.log.write(msg="recipe is a direct SQL statement:\n{}".format(query))
                sql_response = self.input.connector.sql.execute(query)

        except SystemExit:
            try:
                self.log.write("terminating ...")
                if (not sql_direct):
                    write_queue.put(True)
                    write_thread.terminate()
                    supervisor["end"] = True
                    supervisor_thread.terminate()

                    for t in run_queue:
                        try:
                            self.log.write("terminating chunk {}".format(t[0]))
                            t[1].terminate()
                        except:
                            pass

            except:
                pass

            self.log.chunk = "end"
            time.sleep(1)
            self.log.write(error="Recipe {} aborted via SIGTERM".format(
                self.name), exit=True)
            return
        except:
            if (self.test == True):
                error = err()
                try:
                    self.df = df
                except:
                    self.df = None
                if ((len(self.steps) > 0) | ("steps" in list(self.conf.keys()))):
                    self.log.write(msg="in main loop of {} {}".format(
                        self.name, str(self.input.select)), error=error)
                else:
                    if((len(self.before) + len(self.after)) == 0):
                        self.log.write(
                            error="a recipe should contain a least a steps, before, run or after section")
                return self.df
            else:
                error = err()
                # if (len(self.steps)>0 | ("steps" in self.conf.keys())):
                self.log.write(msg="while running {}".format(
                    self.name), error=error)
            try:
                self.log.write("terminating ...")
                if (not sql_direct):
                    write_queue.put(True)
                    write_thread.terminate()
                    supervisor["end"] = True
                    supervisor_thread.terminate()

                    for t in run_queue:
                        try:
                            self.log.write("terminating chunk {}".format(t[0]))
                            t[1].terminate()
                        except:
                            pass
            except:
                pass

        # recipes to run after
        self.run_deps(self.after)

        chunk_number = self.log.chunk

        self.log.chunk = "end"
        if (not sql_direct):
            try:
                with open(self.log.file, 'r') as f:
                    logtext = f.read().split("\n")
                self.errors = len(set([re.search('chunk (\d+)', line).group(1)
                                    for line in logtext if (("Ooops" in line) & ("chunk " in line))]))
                self.processed = sum([int(re.search(
                    'proceed (\d+) rows', line).group(1)) for line in logtext if "proceed" in line])
                self.written = sum([int(re.search('wrote (\d+)', line).group(1))
                                    for line in logtext if "wrote" in line])

            except:
                self.errors = err()
                self.processed = 0
                self.written = 0
            if (self.errors > 0):
                self.log.write(msg="Recipe {} finished with errors on {} chunks (i.e. max {} errors out of {}) - {} lines written".format(
                    self.name, self.errors, self.errors * self.input.connector.chunk, self.processed, self.written))
            else:
                self.log.write(msg="Recipe {} successfully fininshed with no error, {} lines processed, {} lines written".format(
                    self.name, self.processed, self.written))

        else:
            try:
                self.log.write(msg="successfully executed SQL statement: {} rows inserted".format(sql_response.rowcount))
            except:
                self.log.write(msg="SQL statement ended with error, please check error message above")

        self.log.write("end of all")

    def select_columns(self, df=None, arg="select"):
        try:
            if ("select" in list(self.args.keys())):
                if (type(self.args[arg]) == str):
                    self.cols = [x for x in list(
                        df) if re.match(self.args[arg] + "$", x)]
                else:
                    self.cols = self.args[arg]
            else:
                # apply to all columns if none selected
                self.cols = [x for x in list(df)]
        except:
            self.cols = [x for x in list(df)]

    def prepare_categorical(self, df=None):
        df = list(df[self.categorical].reset_index(drop=True).T.to_dict().values())
        prep = DictVectorizer()
        df = prep.fit_transform(df).toarray()
        return df

    def prepare_numerical(self, df=None):
        df = df[self.numerical].fillna("")
        df = df.applymap(lambda x: 0 if (
            (str(x) == "") | (x == None)) else float(x))
        return df

    def internal_fillna(self, df=None, desc=None):
        # try:
        for step in self.args:
            for col in list(step.keys()):
                # self.log.write("{}".format(col))
                if (col not in list(df)):
                    df[col] = step[col]
                else:
                    df[col] = df[col].fillna(step[col])
        return df
        # except:
        # 	self.log.write("Ooops: problem in {} - {}: {} - {}".format(self.name,col,step[col],err()),exit=False)
        # 	return df

    def internal_exec(self,df=None, desc=None):
        if (type(self.args) == str):
            exec(self.args)
        elif (type(self.args) == list):
            for expression in self.args:
                exec(expression)
        return df

    def internal_eval(self, df=None, desc=None):
        try:
            cols = []
            for step in self.args:
                for col in list(step.keys()):
                    cols.append(col)
                    if True:
                        if (type(step[col]) == str):
                            # self.log.write("Ooops: output shape {} to {}".format(dh.shape, df.shape),exit=False)
                            try:
                                df[col] = df.apply(
                                    lambda row: safeeval(step[col], row), axis=1)
                            except SystemExit:
                                return df
                            except:
                                a = df.apply(
                                    lambda row: [safeeval(step[col], row)], axis=1)
                                df[col] = a
                        elif (type(step[col]) == list):
                            multicol = [str(x) for x in step[col]]
                            # print col,multicol, list(df)
                            df[col] = df.apply(lambda row: [safeeval(
                                x, row) for x in multicol], reduce=False, axis=1)
                            #df[col]=pd.concat([df.apply(lambda row: safeeval(x,row), axis=1) for x in multicol], axis = 1)
                    else:
                        pass
            if ("Ooops" in str(df[cols])):
                # report detailed error analysis
                global_col_err = []
                partial_col_err = []
                nerr_total = 0
                for col in cols:
                    col_err = df[df[col].apply(lambda x: "Ooops" in str(x))]
                    nerr = col_err.shape[0]
                    if (nerr == df.shape[0]):
                        global_col_err.append(col)
                    elif (nerr > 0):
                        partial_col_err.append(col)
                        nerr_total += nerr
                if (len(global_col_err) > 0):
                    self.log.write(error="warning in {} : global error in {}".format(
                        self.name, global_col_err), exit=False)
                if (len(partial_col_err) > 0):
                    self.log.write(error="warning in {} : {}/{} errors in {}".format(
                        self.name, nerr_total, df.shape[0], partial_col_err), exit=False)
            return df
        except SystemExit:
            return df
        except:
            self.log.write(msg="problem in {} - {} - {}".format(self.name,
                                                                col, step[col]), error=err(), exit=False)
            return df

    def internal_rename(self, df=None, desc=None):
        dic = {v: k for k, v in self.args.items()}
        df.rename(columns=dic, inplace=True)
        return df

    def internal_map(self, df=None, desc=None):
        for col in list(self.args.keys()):
            if True:
                if type(self.args[col]) == str:
                    df[col] = df[self.args[col]]
                elif type(self.args[col]) == str:
                    df[col] = df[self.args[col]]
                elif (type(self.args[col]) == list):
                    multicol = [str(x) for x in self.args[col]]
                    df[col] = df.apply(lambda row: [row[x]
                                                    for x in multicol], axis=1)
            else:
                pass
        return df

    def internal_shuffle(self, df=None, desc=None):
        # fully shuffles columnes and lines
        try:
            return df.apply(np.random.permutation)
        except SystemExit:
            return df
        except:
            self.log.write(
                msg="problem in {} - {}".format(self.name), error=err(), exit=False)
            return df

    def internal_build_model(self, df=None, desc=None):
        # callable recipe for building method
        # tested only with regression tree
        try:

            if ("numerical" in list(self.args.keys())):
                if (type(self.args["numerical"]) == str):
                    self.numerical = [x for x in list(
                        df) if re.match(self.args["numerical"], x)]
                else:
                    self.numerical = self.args["numerical"]
            else:
                self.numerical = []

            if ("categorical" in list(self.args.keys())):
                if (type(self.args["categorical"]) == str):
                    self.categorical = [x for x in list(
                        df) if re.match(self.args["categorical"], x)]
                else:
                    self.categorical = self.args["categorical"]
            else:
                self.categorical = []

            if ("target" in list(self.args.keys())):
                if (type(self.args["target"]) == str):
                    self.target = [x for x in list(
                        df) if re.match(self.args["target"], x)]
                else:
                    self.target = self.args["target"]
            else:
                self.log.write(error="no target specified for model")
                return df

            self.model = {"library": sklearn.ensemble,
                          "method": "RandomForestRegressor", "parameters": {}, "tries": 5}

            for arg in list(["method", "parameters", "library", "tries", "test_size", "name"]):
                try:
                    self.model[arg] = json.loads(
                        json.dumps(self.args["model"][arg]))
                except:
                    try:
                        self.model[arg] = json.loads(json.dumps(
                            config.conf["machine_learning"]["model"][arg]))
                    except:
                        pass

            # prepare data
            Xn = self.prepare_numerical(df)
            Xc = self.prepare_categorical(df)
            # (Xn,prep_num)=self.prepare_numerical(df[self.numerical])
            # (Xc,prep_cat)=self.prepare_categorical(df[self.categorical])

            X = np.hstack((Xn, Xc))
            # for debug: self.log.write("{} {} {} {}
            # {}".format(X.shape,len(self.numerical),Xn.shape,len(self.categorical),Xc.shape))

            Y = df[self.target].applymap(lambda x: 1 if x else 0)
            # prep = DictVectorizer()
            # X=X.to_dict().values()
            # X = prep.fit_transform(X).toarray()
            err_min = 1
            for i in range(0, self.model["tries"]):
                X_train, X_test, Y_train, Y_test = train_test_split(
                    X, Y, test_size=self.model["test_size"])
                clf = getattr(self.model["library"], self.model[
                              "method"])(**self.model["parameters"])
                # clf = gettattr('',self.type)(n_estimators=100,max_depth=5,min_samples_leaf=10)
                clf = clf.fit(X_train, Y_train)
                # erra = mean_squared_error( clf.predict(X_train), Y_train)**0.5
                # errb = mean_squared_error( clf.predict(X_test), Y_test)**0.5
                erra = roc_auc_score(Y_train, clf.predict(X_train))
                errb = roc_auc_score(Y_test, clf.predict(X_test))
                if (errb < err_min):
                    best_clf = clf
                    err_min = errb
                self.log.write(
                    "estimator {}: auc_train {}, auc_score {}".format(i, erra, errb))
            df["matchid_hit_score_ml"] = best_clf.predict(X)
            df["matchid_hit_score_ml"] = df[
                "matchid_hit_score_ml"].apply(lambda x: round(100 * x))

            self.log.write("{}\n{}".format(self.numerical, self.categorical))
            if (self.test == False):
                try:
                    filename = os.path.join(config.conf["global"]["paths"][
                                            "models"], secure_filename(self.model["name"] + ".model"))
                    joblib.dump(best_clf, filename)
                    # filename=os.path.join(config.conf["global"]["paths"]["models"],secure_filename(self.model["name"]+".cat"))
                    # joblib.dump(prep_cat,filename)
                    # filename=os.path.join(config.conf["global"]["paths"]["models"],secure_filename(self.model["name"]+".num"))
                    # joblib.dump(prep_num,filename)
                    self.log.write("Saved model {}".format(self.model["name"]))
                except:
                    self.log.write(msg="couldn't save model in {}".format(
                        self.name), error=err(), exit=False)

        except SystemExit:
            return df
        except:
            self.log.write(msg="problem while building model from numerical: {} and categorical: {} to {} in {}".format(
                self.numerical, self.categorical, self.target, self.name), error=err(), exit=False)
            return df
        return df

    def internal_apply_model(self, df=None, desc=None):
        # callable recipe for building method
        # tested only with regression tree
        try:
            if ("numerical" in list(self.args.keys())):
                if (type(self.args["numerical"]) == str):
                    self.numerical = [x for x in list(
                        df) if re.match(self.args["numerical"], x)]
                else:
                    self.numerical = self.args["numerical"]
            else:
                self.numerical = []

            if ("numerical" in list(self.args.keys())):
                if (type(self.args["numerical"]) == str):
                    self.numerical = [x for x in list(
                        df) if re.match(self.args["numerical"], x)]
                else:
                    self.numerical = self.args["numerical"]
            else:
                self.numerical = []

            if ("categorical" in list(self.args.keys())):
                if (type(self.args["categorical"]) == str):
                    self.categorical = [x for x in list(
                        df) if re.match(self.args["categorical"], x)]
                else:
                    self.categorical = self.args["categorical"]
            else:
                self.categorical = []

            if ("target" in list(self.args.keys())):
                self.target = self.args["target"]
            else:
                self.log.write(
                    error="no target specified for model prediction")
                return df

            # load model
            filename = os.path.join(config.conf["global"]["paths"][
                                    "models"], secure_filename(self.args["name"] + ".model"))
            clf = joblib.load(filename)
            # filename=os.path.join(config.conf["global"]["paths"]["models"],secure_filename(self.args["name"]+".cat"))
            # prep_cat=joblib.load(filename)
            # filename=os.path.join(config.conf["global"]["paths"]["models"],secure_filename(self.args["name"]+".num"))
            # prep_num=joblib.load(filename)

            # prepare data
            Xn = self.prepare_numerical(df)
            Xc = self.prepare_categorical(df)
            # (Xn,prep_num)=self.prepare_numerical(df[self.numerical],prep_num)
            # (Xc,prep_cat)=self.prepare_categorical(df[self.categorical],prep_cat)
            X = np.hstack((Xn, Xc))
            # for debug : self.log.write("{} {} {} {}
            # {}".format(X.shape,len(self.numerical),Xn.shape,len(self.categorical),Xc.shape))

            df[self.target] = clf.predict(X)
            df[self.target] = df[self.target].apply(lambda x: round(100 * x))
        except SystemExit:
            return df

        except:
            self.log.write(msg="problem while applying model from numerical: {} and categorical: {} to {} in {}".format(
                self.numerical, self.categorical, self.target, self.name), error=err(), exit=False)

        return df

    def internal_keep(self, df=None, desc=None):
        # keep only selected columns
        self.select_columns(df=df)
        try:
            if ("where" in list(self.args.keys())):
                df["matchid_selection_xykfsd"] = df.apply(
                    lambda row: safeeval(self.args["where"], row), axis=1)
                df = df[df.matchid_selection_xykfsd == True]
                del df["matchid_selection_xykfsd"]
            return df[self.cols]
        except SystemExit:
            return df
        except:
            self.log.write(msg="{}".format(self.cols), error=err(), exit=False)
            return df

    def internal_to_integer(self, df=None, desc=None):
        # keep only selected columns
        self.select_columns(df=df)
        try:
            df[self.cols] = df[self.cols].applymap(
                lambda x: np.nan if (str(x) == "") else int(x))
            return df
        except SystemExit:
            return df
        except:
            self.log.write(msg="{}".format(self.cols), error=err(), exit=False)
            return df

    def internal_list_to_tuple(self, df=None, desc=None):
        # keep only selected columns
        self.select_columns(df=df)
        try:
            df[self.cols] = df[self.cols].applymap(
                lambda x: tuple(x) if (type(x) == list) else x)
            return df
        except SystemExit:
            return df
        except:
            self.log.write(msg="{}".format(self.cols), error=err(), exit=False)
            return df

    def internal_tuple_to_list(self, df=None, desc=None):
        # keep only selected columns
        self.select_columns(df=df)
        try:
            df[self.cols] = df[self.cols].applymap(
                lambda x: list(x) if (type(x) == tuple) else x)
            return df
        except SystemExit:
            return df
        except:
            self.log.write(msg="{}".format(self.cols), error=err(), exit=False)
            return df

    def internal_to_float(self, df=None, desc=None):
        # keep only selected columns
        self.select_columns(df=df)
        try:
            na_value = self.args["na_value"]
        except:
            na_value = np.nan
        try:
            df[self.cols] = df[self.cols].applymap(
                lambda x: na_value if (str(x) == "") else float(x))
            return df
        except SystemExit:
            return df
        except:
            self.log.write(msg="{}".format(self.cols), error=err(), exit=False)
            return df

    def internal_ngram(self, df=None, desc=None):
        # keep only selected columns
        self.select_columns(df=df)
        if ("n" in list(self.args.keys())):
            n = self.args['n']
        else:
            n = list([2, 3])
        try:
            df[self.cols] = df[self.cols].applymap(
                lambda x: ngrams(tokenize(normalize(x)), n))
            return df
        except SystemExit:
            return df
        except:
            self.log.write(msg="{}".format(self.cols), error=err(), exit=False)
            return df

    def internal_clique(self, df=None, desc=None):
        try:
            self.select_columns(df=df)
            nodes = self.cols
            if (len(nodes) > 2):
                self.log.write(msg="nodes columns {}".format(
                    nodes), error="2 columns exactly are required")

            try:
                prefix = self.args["prefix"]
            except:
                prefix = "graph_"

            try:
                if (self.args["compute"] == None):
                    to_compute = []
                elif (self.args["compute"] == "all"):
                    to_compute = ["clique_list", "degree", "clustering", "triangles", "closeness_centrality",
                                  "pagerank", "square_clustering", "eigenvector_centrality_numpy"]
                else:
                    to_compute = list(
                        set(flatten([["clique_list"], self.args["compute"]])))
            except:
                to_compute = []

            # create graph from links
            graph = nx.Graph()
            graph.add_edges_from(
                list(zip(df[nodes[0]].values.tolist(), df[nodes[1]].values.tolist())))

            # compute every factor to compute
            computed = []
            for method in to_compute:
                try:
                    if (method == "degree"):
                        deg = pd.DataFrame(
                            pd.Series(nx.degree(graph)).apply(pd.Series))
                        deg = deg.set_index(list(deg)[0]).rename(
                            index=str, columns={list(deg)[1]: prefix + method})
                        computed.append(deg)
                    elif (method != "clique_list"):
                        computed.append(
                            pd.Series(getattr(nx, method)(graph), name=prefix + method))
                except:
                    self.log.write(
                        msg="computing {}".format(method), error=err())

            # generate cluster/clique
            id = {}
            if ("clique_list" in to_compute):
                cluster_nodes = {}
            for cluster in nx.connected_components(graph):
                cluster_id = sha1(uuid.uuid4())
                cluster = sorted(cluster)
                for node in cluster:
                    id[node] = cluster_id
                    if ("clique_list" in to_compute):
                        cluster_nodes[node] = cluster

            computed.append(pd.Series(id, name=prefix + "clique_id"))
            if ("clique_list" in to_compute):
                computed.append(
                    pd.Series(cluster_nodes, name=prefix + "clique"))

            df_graph = pd.concat(computed, axis=1)
            df_graph[prefix + "clique_size"] = df_graph.groupby(
                [prefix + "clique_id"])[prefix + "clique_id"].transform('count')

            df_graph.reset_index(inplace=True)
            df_graph.rename(columns={"index": nodes[0]}, inplace=True)
            df = df.merge(df_graph, on=nodes[0], how='left')
        except:
            self.log.write(msg="", error=err())
        return df

    def internal_sql(self, df=None, desc=None):
        if (type(self.args) == str):
            self.input.connector.sql.execute(self.args)
        elif (type(self.args) == list):
            for expression in self.args:
                self.input.connector.sql.execute(expression)
        return df

    def internal_delete(self, df=None, desc=None):
        # keep only selected columns
        self.select_columns(df=df)
        #log("selecting {}".format(self.cols),level=3)
        try:
            df.drop(self.cols, axis=1, inplace=True)
            # for col in self.cols:
            # 	del df[col]
            return df
        except SystemExit:
            return df
        except:
            self.log.write(msg="{}".format(self.cols), error=err(), exit=False)
            return df

    def internal_groupby(self, df=None, desc=None):
        self.select_columns(df=df)
        try:
            if ("agg" in list(self.args.keys())):
                self.cols = [
                    x for x in self.cols if x not in list(self.args["agg"].keys())]
                dic = {'list': union}
                aggs = replace_dict(self.args["agg"], dic)
                df = df.groupby(self.cols).agg(aggs).reset_index()
            if ("transform" in list(self.args.keys())):
                for step in self.args["transform"]:
                    for col in list(step.keys()):
                        if (step[col] != "rank"):
                            df[col + '_' + step[col]
                               ] = df.groupby(self.cols)[col].transform(step[col])
                        else:
                            df[col + '_' + step[col]] = df.groupby(
                                self.cols)[col].transform(step[col], method='dense')
            if ("rank" in list(self.args.keys())):
                for col in self.args["rank"]:
                    df[col + '_rank'] = df.groupby(self.cols)[col].rank(
                        method='dense', ascending=False)

            # if ("apply" in self.args.keys()):
            # 	self.cols = [x for x in self.cols if x not in self.args["apply"].keys()]
            # 	dfg=df.groupby(self.cols)
            # 	df=pd.concat([dfg.apply(lambda x: safeeval((self.args["apply"][col]),{"x": x})) for col in self.args["apply"].keys()])
        except SystemExit:
            return df
        except:
            self.log.write(msg="{}".format(self.cols), error=err())
        return df

    def internal_join(self, df=None, desc=None):
        try:
            join_type = "in_memory"
            if (self.args == None):
                self.log.write(error="no args in join", exit=True)
            if ("type" in list(self.args.keys())):
                if (self.args["type"] == "elasticsearch"):
                    join_type = "elasticsearch"
            if (join_type == "in_memory"):  # join in memory
                ds = self.args["dataset"]

                # cache inmemory reading
                # a flush method should be created
                try:
                    # inmemory cache
                    config.inmemory[ds].df
                except:
                    self.log.write(
                        "Creating cache for join with dataset {} in {}".format(ds, self.name))
                    config.inmemory[ds] = Dataset(self.args["dataset"])
                    config.inmemory[ds].init_reader()
                    config.inmemory[ds].df = pd.concat(
                        [dx['df'] for dx in config.inmemory[ds].reader]).reset_index(drop=True)

                # collects useful columns
                if ("select" in list(self.args.keys())):
                    # select columns to retrieve in join
                    cols = [self.args["select"][col]
                            for col in list(self.args["select"].keys())]
                    if ("strict" in list(self.args.keys())):
                        # keep joining cols
                        cols = list(set().union(cols,	[self.args["strict"][
                                    x] for x in list(self.args["strict"].keys())]))
                    if ("fuzzy" in list(self.args.keys())):
                        # keep fuzzy joining cols
                        cols = list(set().union(cols,	[self.args["fuzzy"][
                                    x] for x in list(self.args["fuzzy"].keys())]))
                        # initiate levenstein matcher (beta : not optimized)
                        # this method remains in memory
                        try:
                            config.inmemory[ds].matcher
                        except:
                            config.inmemory[ds].matcher = {}
                        for col in list(self.args["fuzzy"].keys()):
                            try:
                                config.inmemory[ds].matcher[
                                    self.args["fuzzy"][col]]
                            except:
                                self.log.write("Creating automata cache for fuzzy join on column {} of dataset {} in {}".format(
                                    col, ds, self.name))
                                words = sorted(set(config.inmemory[ds].df[
                                               self.args["fuzzy"][col]].tolist()))
                                config.inmemory[ds].matcher[
                                    self.args["fuzzy"][col]] = automata.Matcher(words)

                # caches filtered version of the dataset
                try:
                    join_df = config.inmemory[ds].filtered[sha1(cols)]
                except:
                    try:
                        self.log.write(
                            "Creating filtered cache for join with dataset {} in {}".format(ds, self.name))
                        config.inmemory[ds].filtered
                    except:
                        config.inmemory[ds].filtered = {}
                    config.inmemory[ds].filtered[sha1(cols)] = config.inmemory[
                        ds].df[cols]
                    join_df = config.inmemory[ds].filtered[sha1(cols)]

                if ("fuzzy" in list(self.args.keys())):
                    for col in list(self.args["fuzzy"].keys()):
                        # get fuzzy matches for the fuzzy columns
                        if ("fuzzy_method" in list(self.args.keys())):
                            fuzzy_method = self.args["fuzzy_method"]
                        else:
                            fuzzy_method = "automata"
                        if (fuzzy_method == "automata"):
                            # using levenshtein automata (tested 10x faster as tested against fastcomp and jaro winkler)
                            # a full openfst precompile automata would be still
                            # faster but not coded for now
                            df[col + "_match"] = df[col].map(lambda x:
                                                             next(itertools.chain.from_iterable(
                                                                 automata.find_all_matches(x, dist, config.inmemory[
                                                                     ds].matcher[self.args["fuzzy"][col]])
                                                                 for dist in range(2)), ""))
                        elif (fuzzy_method == "jellyfish"):
                            # using jellyfish jaro winkler
                            df[col + "_match"] = df[col].map(
                                lambda x: match_jw(x, join_df[self.args["fuzzy"][col]]))
                        elif (fuzzy_method == "fastcomp"):
                            # using fastcomp
                            df[col + "_match"] = df[col].map(
                                lambda x: match_lv1(x, join_df[self.args["fuzzy"][col]]))
                    # now prematched fuzzy terms in cols _match are ok for a strict join
                    # list joining columns
                    left_on = [
                        col + "_match" for col in list(self.args["fuzzy"].keys())]
                    right_on = [self.args["fuzzy"][x]
                                for x in list(self.args["fuzzy"].keys())]
                    if ("strict" in list(self.args.keys())):
                        # complete joining columns list if asked
                        left_on = list(set().union(
                            left_on, list(self.args["strict"].keys())))
                        right_on = list(set().union(right_on, [self.args["strict"][
                                        x] for x in list(self.args["strict"].keys())]))

                    # joining, the right dataset being keepd in memory
                    df = pd.merge(df, join_df,
                                  how='left', left_on=left_on,
                                  right_on=right_on,
                                  left_index=False, right_index=False)
                    # self.log.write("{}x{} - {}\n{}".format(left_on,right_on,self.name,df[list(set().union(left_on,right_on))].head(n=5)))

                    # map new names of retrieved colums
                    if ("select" in self.args):
                        reverse = {v: k for k, v in self.args[
                            "select"].items()}
                        # python 3 reverse={v: k for k, v in
                        # self.args["select"].items()}
                        df.rename(columns=reverse, inplace=True)
                    # remove unnecessary columns of the right_on
                    right_on = [x for x in right_on if x not in left_on]
                    df.drop(right_on, axis=1, inplace=True)

                elif ("strict" in list(self.args.keys())):
                    # simple strict join
                    df = pd.merge(df, join_df,
                                  how='left', left_on=list(self.args["strict"].keys()),
                                  right_on=[self.args["strict"][x]
                                            for x in list(self.args["strict"].keys())],
                                  left_index=False, right_index=False)

                    # map new names of retrieved colums
                    if ("select" in list(self.args.keys())):
                        reverse = {v: k for k, v in self.args[
                            "select"].items()}
                        # python3 reverse={v: k for k, v in
                        # self.args["select"].items()}
                        df.rename(columns=reverse, inplace=True)
                    # remove unnecessary columns of the right_on
                    for key in [self.args["strict"][x] for x in list(self.args["strict"].keys())]:
                        try:
                            del df[key]
                        except:
                            pass
            else:  # join with elasticsearch
                if True:
                    es = Dataset(self.args["dataset"])
                    query = self.args["query"]
                    index = 0
                    try:
                        prefix = self.args["prefix"]
                    except:
                        prefix = "hit_"
                    if True:
                        m_res = []

                        rest = df.applymap(lambda x: "" if x is None else x)
                        rest.fillna("", inplace=True)

                        # elasticsearch bulk search
                        while rest.shape[0] > 0:
                            part = rest[:es.connector.chunk_search]
                            rest = rest[es.connector.chunk_search:]
                            index += es.connector.chunk_search
                            bulk = "\n".join(part.apply(lambda row: jsonDumps(
                                {"index": es.table}) + "\n" + jsonDumps(replace_dict(query, row)), axis=1))
                            # self.log.write("\n{}".format(bulk))
                            tries = 0
                            success = False
                            failure = None
                            max_tries = es.max_tries
                            while(tries < max_tries):
                                try:
                                    res = es.connector.es.msearch(
                                        bulk, request_timeout=10 + 10 * tries)
                                    df_res = pd.concat(list(map(pd.DataFrame.from_dict, res['responses'])), axis=1)[
                                        'hits'].T.reset_index(drop=True)
                                    max_tries = tries
                                    success = True
                                except elasticsearch.TransportError:
                                    error = err() + bulk.encode("utf-8", 'ignore')
                                    tries = max_tries
                                    df_res = part['matchid_id'].apply(
                                        lambda x: {"_source": {}})
                                except:
                                    tries += 1
                                    # prevents combo deny of service of
                                    # elasticsearch
                                    time.sleep(tries * 5)
                                    self.log.write(msg="warning - join {} x {} retry sub-chunk {} to {}".format(
                                        self.name, self.args["dataset"], index - es.connector.chunk_search, index))
                                    failure = "Timeout"
                                    df_res = part['matchid_id'].apply(
                                        lambda x: {"_source": {}})
                            if (success == False):
                                self.log.write(msg="join {} x {} failure in sub-chunk {} to {}".format(
                                    self.name, self.args["dataset"], index - es.connector.chunk_search, index), error=error)
                            m_res.append(df_res)
                        df_res = pd.concat(m_res).reset_index(drop=True)

                        # #self.log.write("debug: {}".format(df_res))
                        df_res['matchid_hit_matches_unfiltered'] = df_res[
                            'total']
                        # df_res.drop(['total','failed','successful','max_score'],axis=1,inplace=True)
                        df_res.drop(['failed', 'successful',
                                     'skipped'], axis=1, inplace=True)
                        df_res.rename(columns={
                                      'total': prefix + 'total', 'max_score': prefix + 'max_score'}, inplace=True)
                        df = pd.concat(
                            [df.reset_index(drop=True), df_res], axis=1)
                        #self.log.write("after ES request:{}".format(df.shape))

                        try:
                            unfold = self.args["unfold"]
                        except:
                            unfold = True
                        # unfold
                        if (unfold == True):
                            unfold = Recipe(
                                'unfold', args={"select": ['hits'], "fill_na": ""})
                            unfold.init(df=df, parent=self, test=self.test)
                            df = unfold.run_chunk(0, df)
                            #self.log.write("after unfold:{}".format(df.shape))
                            try:
                                keep_unmatched = self.args["keep_unmatched"]
                            except:
                                keep_unmatched = False
                            if (keep_unmatched == False):
                                df = df[df.hits != ""]
                            del df_res

                            # #unnest columns of each match : a <> {key1:val11, key2:val21} gives : a <> val11, val21
                            try:
                                unnest = self.args["unnest"]
                            except:
                                unnest = True
                            if (unnest == True):
                                df['hits'] = df['hits'].apply(lambda x: {} if (
                                    x == "") else deepupdate(x['_source'], {'score': x['_score']}))

                                unnest = Recipe('unnest', args={"select": [
                                                'hits'], "prefix": prefix})
                                unnest.init(df=df, parent=self, test=self.test)
                                df = unnest.run_chunk(0, df)
                                #self.log.write("after unnest :{}".format(df.shape))

                    else:
                        pass
                else:
                    return df
        except SystemExit:
            return df
        except:
            self.log.write("join {} x {} failed : {}".format(
                self.name, self.args["dataset"], err()))
        return df.fillna('')

    def internal_unnest(self, df=None, desc=None):
        self.select_columns(df=df)
        try:
            prefix = self.args["prefix"]
        except:
            prefix = ''

        try:
            df_list = [df]
            for col in self.cols:
                df_list.append(df[col].apply(pd.Series).add_prefix(prefix))
            return pd.concat(df_list, axis=1).drop(self.cols, axis=1)
        except SystemExit:
            return df
        except:
            self.log.write(error=err())
            return df

    def internal_nest(self, df=None, desc=None):
        self.select_columns(df=df)
        try:
            target = self.args["target"]
        except:
            target = "nested"
        try:
            df[target] = df[self.cols].apply(lambda x: x.to_json(), axis=1)
            df = self.internal_delete(df)
        except SystemExit:
            return df
        except:
            self.log.write(error=err())
        return df

    def internal_unfold(self, df=None, desc=None):
        self.select_columns(df=df)
        try:
            fill_na = self.args["fill_na"]
        except:
            fill_na = ""
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
                    col: np.repeat(df[col].values, df[self.cols[0]].str.len())
                    for col in idx_cols
                }).assign(**{col: np.concatenate(df[col].values) for col in self.cols}) \
                    .loc[:, df.columns]
            else:
                # at least one list in cells is empty
                return pd.DataFrame({
                    col: np.repeat(df[col].values, df[self.cols[0]].str.len())
                    for col in idx_cols
                }).assign(**{col: np.concatenate(df[col].values) for col in self.cols}) \
                    .append(df.loc[lens == 0, idx_cols]).fillna(fill_na) \
                    .loc[:, df.columns]
        except SystemExit:
            return df
        except:
            self.log.write(error=err())
            return df

    def internal_parsedate(self, df=None, desc=None):
        self.select_columns(df=df)
        if ("format" in list(self.args.keys())):
            # parse string do datetime i.e. 20001020 + %Y%m%d =>
            # 2000-10-20T00:00:00Z
            for col in self.cols:
                df[col] = pd.to_datetime(
                    df[col], errors='coerce', format=self.args["format"])
            # df[self.cols]=df[self.cols].applymap(lambda x:
            #	parsedate(x,self.args["format"]))

        return df

    def internal_replace(self, df=None, desc=None):
        if True:
            self.select_columns(df=df)
            if ("regex" in list(self.args.keys())):
                regex = []
                # warning: replace use a dict which is not ordered
                for r in self.args["regex"]:
                    regex.append([re.compile(list(r.keys())[0]), r[list(r.keys())[0]]])
                pd.options.mode.chained_assignment = None
                df[self.cols] = df[self.cols].applymap(
                    lambda x: replace_regex(x, regex))
            return df
        else:
            return df

    def internal_normalize(self, df=None, desc=None):
        if True:
            self.select_columns(df=df)
            for col in list(self.cols):
                df[col] = df[col].str.normalize('NFKD').str.encode('ASCII',errors='ignore').str.decode('ASCII')
            return df
        else:
            return df

    def internal_pause(self, df=None, desc=None):
        return df


def thread_job(recipe=None):
    try:
        try:
            try:
                config.jobs_list[recipe.name] = {
                    "status": "up", "log": recipe.log.file, "pid": os.getpid(), "callback": recipe.callback}
            except:
                config.jobs_list[recipe.name] = {
                    "status": "up", "log": None, "pid": os.getpid(), "callback": recipe.callback}

            recipe.callback["df"] = recipe.run()
        except SystemExit:
            recipe.log.write("terminating ...")
            for p in multiprocessing.active_children():
                p.terminate()
        del config.jobs_list[recipe.name]

        recipe.callback["errors"] = recipe.errors
        try:
            recipe.callback["log"] = str(recipe.log.writer.getvalue())
        except:
            with open(recipe.log.file, 'r') as f:
                recipe.callback["log"] = f.read()
    except:
        pass
