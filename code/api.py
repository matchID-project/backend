#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import basics
import sys
import os
import io
import fnmatch
import re
import time
import datetime
import hashlib
import unicodedata
import shutil

import traceback
import json
import yaml
import itertools
import time
import operator
import simplejson
from collections.abc import Iterable
from collections import OrderedDict
from pandas.io.json import json_normalize
from collections import deque


# interact with datasets
# from pandasql import sqldf
import elasticsearch
from elasticsearch import Elasticsearch, helpers
import pandas as pd

# parallelize
# import concurrent.futures
# import threading
from multiprocessing import Process
import uuid

# recipes

# api
from flask import Flask, current_app, jsonify, Response, abort, request, g, stream_with_context
from flask.sessions import SecureCookieSessionInterface
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from flask_restplus import Resource, Api, reqparse
from werkzeug.utils import secure_filename
from werkzeug.serving import run_simple
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.middleware.proxy_fix import ProxyFix
from flask import make_response as original_flask_make_response
from functools import wraps

# matchID imports
import parsers
import config
from tools import replace_dict
from recipes import *
from security import *
from oauth import *
from log import Log, err


def allowed_upload_file(filename=None):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.conf[
               "global"]["data_extensions"]


def allowed_conf_file(filename=None):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.conf[
               "global"]["recipe_extensions"]


config.init()
config.read_conf()
auth = LoginManager()

app = Flask(__name__)
try:
    app.config['LOGIN_DISABLED'] = config.conf["global"]["api"]["no_auth"]
except:
    pass

app.wsgi_app = ProxyFix(app.wsgi_app)
app.secret_key = config.conf["global"]["api"]["secret_key"]
auth.session_protection = "strong"
auth.init_app(app)

api = Api(app, version="0.1", title="matchID API",
          description="API for data matching developpement")
app.config['APPLICATION_ROOT'] = config.conf["global"]["api"]["prefix"]

def authorize(override_project = None, force_dataset = None, force_recipe = None, right='read'):
    def wrapper(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            try:
                if config.conf["global"]["api"]["no_auth"] == True:
                    return f(*args, **kwargs)
            except:
                pass
            if (override_project != None):
                project = override_project
            else:
                try:
                    project = kwargs['project']
                except:
                    project = None

            try:
                dataset = kwargs['dataset']
            except:
                dataset = None
            try:
                recipe = kwargs['recipe']
            except:
                recipe = None

            config.read_conf()
            if current_user is None:
                api.abort(401)
            if project is None:
                if dataset is None:
                    if recipe is None:
                        api.abort(401)
                    else:
                        try:
                            project = config.conf["recipes"][recipe]["project"]
                        except:
                            api.abort(401)
                else:
                    try:
                        project = config.conf["datasets"][dataset]["project"]
                    except:
                        api.abort(401)
            if (check_rights(current_user, project, right) == False):
                api.abort(401)
            return f(*args, **kwargs)
        return wrapped
    return wrapper

@auth.user_loader
def load_user(name):
    try:
        return User(name)
    except:
        api.abort(401)

@api.route('/users/', endpoint='users')
class ListUsers(Resource):

    @login_required
    def get(self):
        '''get list of all configured users'''
        config.read_conf()
        if (check_rights(current_user, "$admin", "read")):
            return config.conf["users"]
        else:
            return {
                "me": str(current_user.name),
                "others": list(config.conf["users"].keys())
            }

@api.route('/groups/', endpoint='groups')
class ListGroups(Resource):

    @login_required
    @authorize(override_project = "$admin")
    def get(self):
        '''get all groups'''
        config.read_conf()
        return config.conf["groups"]


@api.route('/roles/', endpoint='roles')
class ListRoles(Resource):

    @login_required
    @authorize(override_project = "$admin")
    def get(self):
        '''get all roles'''
        config.read_conf()
        return config.conf["roles"]


@api.route('/login/', endpoint='login')
class login(Resource):

    @login_required
    def get(self):
        '''return current user if logged'''
        try:
            return {"user": str(current_user.name)}
        except:
            try:
                if config.conf["global"]["api"]["no_auth"] == True:
                    login_user(User("admin"), remember=True)
            except:
                api.abort(401)

    def post(self):
        '''login api with user and hash'''
        config.read_conf()
        try:
            if (config.conf["global"]["api"]["no_auth"] == True):
                login_user(User("admin"))
                return {"user": str(current_user.name)}
        except:
            pass
        try:
            args = request.get_json(force=True)
            user = args['user']
            password = args['password']
        except:
            api.abort(403, {"error": err()})
        # Login and validate the user.
        # user should be an instance of your `User` class
        try:
            u = User(user)
            if (u.check_password(password)):
                login_user(u, remember=True)
                return {"user": str(current_user.name)}
            else:
                api.abort(403)

        except:
            api.abort(403)

@api.route('/authorize/', endpoint='authorize')
class OAuthList(Resource):
    def get(self):
        return {
            'providers': list([x for x in list(config.conf['global']['api']['oauth'].keys()) if config.conf['global']['api']['oauth'][x]['id'] != None])
            }

@api.route('/authorize/<provider>', endpoint='authorize/<provider>')
class OAuthAuthorizeAPI(Resource):
    def get(self, provider):
        '''authorize api for OAuth protocol'''
        try:
            if (current_user.name != None):
                return redirect(config.conf['global']['frontend']['url'])
        except:
            pass
        oauth = OAuthSignIn.get_provider(provider)
        return oauth.authorize()

@api.route('/callback/<provider>', endpoint='callback/<provider>')
class OAuthCallbackAPI(Resource):
    def get(self, provider):
        '''callback api for OAuth protocol'''
        try:
            if (current_user.name != None):
                return redirect(config.conf['global']['frontend']['url'])
        except:
            pass
        oauth = OAuthSignIn.get_provider(provider)
        social_id, username, email = oauth.callback()
        if social_id is None:
            api.abort(401)
        login_user(User(name=str(username), social_id=social_id, email=email, provider=provider))
        return redirect(config.conf['global']['frontend']['url'])

@api.route("/logout/", endpoint='logout')
class Logout(Resource):

    @login_required
    def post(self):
        '''logout current user'''
        logout_user()
        return {"status": "logged out"}

@api.route('/shutdown/', endpoint='shutdown')
class Shutdown(Resource):

    @login_required
    @authorize(right="admin")
    def put(self):
        '''stop matchID backend service'''
        func = request.environ.get('werkzeug.server.shutdown')
        if func is None:
            raise RuntimeError('Not running with the Werkzeug Server')
        func()
        return {"message": "Server restarting   ..."}

@api.route('/conf/', endpoint='conf')
class Conf(Resource):

    @login_required
    def get(self):
        '''get all configured elements
        Lists all configured elements of the backend, as described in the yaml files :
        - global configuration
        - projects :
          - datasets
          - recipes'''
        try:
            config.read_conf()
            if (check_rights(current_user, "$admin", "read")):
                response = config.conf["global"]
            else:
                response = {
                    "projects": {project: config.conf["global"]["projects"][project]
                                 for project in config.conf["global"]["projects"]
                                 if (check_rights(current_user, project, "read"))
                                 }
                }
            return response
        except:
            return {"error": err()}


@api.route('/upload/', endpoint='upload')
class Upload(Resource):

    @login_required
    def get(self):
        '''list uploaded resources'''
        return list([filenames for root, dirnames, filenames in os.walk(config.conf["global"]["paths"]["upload"])])[0]

    @login_required
    @api.expect(parsers.upload_parser)
    def post(self):
        '''upload multiple tabular data files, .gz or .txt or .csv'''
        response = {"upload_status": {}}
        args = parsers.upload_parser.parse_args()
        for file in args['file']:
            if (allowed_upload_file(file.filename)):
                try:
                    file.save(os.path.join(config.conf["global"]["paths"][
                              "upload"], secure_filename(file.filename)))
                    response["upload_status"][file.filename] = "ok"
                except:
                    response["upload_status"][file.filename] = err()
            else:
                response["upload_status"][
                    file.filename] = "extension not allowed"
        return response


@api.route('/upload/<file>', endpoint='upload/<file>')
@api.doc(params={'file': 'file name of a previously uploaded file'})
class actionFile(Resource):

    @login_required
    def get(self, file):
        '''get back uploaded file'''
        filetype = "unknown"
        pfile = os.path.join(config.conf["global"]["paths"]["upload"], file)
        try:
            df = pd.read_csv(pfile, nrows=100)
            filetype = "csv"
        except:
            pass
        return {"file": file, "type_guessed": filetype}

    @login_required
    def delete(self, file):
        '''deleted uploaded file'''
        try:
            pfile = os.path.join(config.conf["global"][
                "paths"]["upload"], file)
            os.remove(pfile)
            return {"file": file, "status": "deleted"}
        except:
            api.abort(404, {"file": file, "status": err()})


@api.route('/conf/<project>/', endpoint='conf/<project>')
@api.doc(parms={'project': 'name of a project'})
class DirectoryConf(Resource):

    @login_required
    @authorize(right="read")
    def get(self, project):
        '''get configuration files of a project'''
        config.read_conf()
        if project in list(config.conf["global"]["projects"].keys()):
            return config.conf["global"]["projects"][project]
        else:
            api.abort(404)

    @login_required
    @api.expect(parsers.conf_parser)
    def post(self, project):
        '''(KO) import a zipped project'''
        if (directory != "conf"):
            response = {"upload_status": {}}
            args = parsers.conf_parser.parse_args()
            for file in args['file']:
                if (allowed_conf_file(file.filename)):
                    try:
                        file.save(os.path.join(config.conf["global"]["paths"][
                                  "conf"][project], secure_filename(file.filename)))
                        response["upload_status"][file.filename] = "ok"
                    except:
                        response["upload_status"][file.filename] = err()
                else:
                    response["upload_status"][
                        file.filename] = "extension not allowed"
                config.read_conf()
                response["yaml_validator"] = config.conf[
                    "global"]["projects"][project]
            return response
        else:
            api.abort(403)

    @login_required
    @authorize(override_project = '$create_projects', right = 'create')
    def put(self, project):
        '''create a project'''
        if (project == "conf"):
            api.abort(403)
        elif project in list(config.conf["global"]["projects"].keys()):
            api.abort(400, 'project "{}" already exists'.format(project))
        else:
            try:
                dirname = os.path.join(config.conf["global"][
                    "paths"]["projects"], project)
                creds_file = os.path.join(dirname, 'creds.yml')
                os.mkdir(dirname)
                os.mkdir(os.path.join(dirname, 'recipes'))
                os.mkdir(os.path.join(dirname, 'datasets'))
                groups = {
                            'groups': {
                                str(project): {
                                    'projects': {
                                        str(project): {
                                            'admin': str(current_user.name)
                                        }
                                    }
                                }
                            }
                        }
                with open(creds_file, 'w') as f:
                    yaml.dump(groups, f)
                config.read_conf()
                return {"message": "{} successfully created".format(project)}
            except:
                api.abort(400, err())

    @login_required
    @authorize(right = 'delete')
    def delete(self, project):
        '''delete a project'''
        if (project == "conf"):
            api.abort(403)
        elif project in list(config.conf["global"]["projects"].keys()):
            response = {project: "not deleted"}
            try:
                dirname = os.path.join(config.conf["global"][
                    "paths"]["projects"], project)
                shutil.rmtree(dirname)
                response[project] = "deleted"
            except:
                response[project] = "deletion failed - " + err()
            config.read_conf()
            # response["yaml_validator"]=config.conf["global"]["projects"][project]
            return response
        else:
            api.abort(404)


@api.route('/conf/<project>/<path:file>', endpoint='conf/<project>/<path:file>')
class FileConf(Resource):

    @login_required
    def get(self, project, file):
        '''get a text/yaml configuration file from project'''
        try:
            config.read_conf()
            if (file in config.conf["global"]["projects"][project]["files"]):
                try:
                    pfile = os.path.join(config.conf["global"]["projects"][
                        project]["path"], file)
                    with open(pfile) as f:
                        return Response(f.read(), mimetype="text/plain")
                except:
                    api.abort(404)
            else:
                api.abort(404)
        except:
            api.abort(404)

    @login_required
    def delete(self, project, file):
        '''delete a text/yaml configuration file from project'''
        if (project != "conf"):
            if (file in config.conf["global"]["projects"][project]["files"]):
                try:
                    pfile = os.path.join(config.conf["global"]["projects"][
                        project]["path"], file)
                    os.remove(pfile)
                    config.read_conf()
                    return jsonify({"conf": project, "file": file, "status": "removed"})
                except:
                    api.abort(403)

    @login_required
    @api.expect(parsers.yaml_parser)
    def post(self, project, file):
        '''upload a text/yaml configuration file to a project'''
        if (project != "project"):
            args = parsers.yaml_parser.parse_args()
            filecontent = args['yaml']
            if (allowed_conf_file(file)):
                try:
                    test = config.ordered_load(filecontent)
                except:
                    api.abort(400, {file: {"saved": "ko - " + err()}})

                try:
                    pfile = os.path.join(config.conf["global"]["projects"][
                        project]["path"], file)
                    with open(pfile, 'w') as f:
                        f.write(filecontent)
                    response = {file: {"saved": "ok"}}
                    config.read_conf()
                    response[file]["yaml_validator"] = config.conf[
                        "global"]["projects"][project]["files"][file]
                    return response
                except:
                    api.abort(400, {file: {"saved": "ko - " + err()}})
            else:
                api.abort(403)
        else:
            api.abort(403)


@api.route('/connectors/', endpoint='connectors')
class ListConnectors(Resource):

    @login_required
    def get(self):
        '''get json of all configured connectors'''
        config.read_conf()
        return config.conf["connectors"]


@api.route('/datasets/', endpoint='datasets')
class ListDatasets(Resource):

    @login_required
    def get(self):
        '''get json of all configured datasets'''
        config.read_conf()
        response = {dataset: config.conf["datasets"][dataset]
                         for dataset in config.conf["datasets"]
                         if (check_rights(current_user, config.conf["datasets"][dataset]["project"], "read"))
                         }
        return response


@api.route('/datasets/<dataset>/', endpoint='datasets/<dataset>')
class DatasetApi(Resource):

    @login_required
    @authorize()
    def get(self, dataset):
        '''get json of a configured dataset'''
        config.read_conf()
        if (dataset in list(config.conf["datasets"].keys())):
            try:
                response = dict(config.conf["datasets"][dataset])
                try:
                    ds = Dataset(dataset)
                    response["type"] = ds.connector.type
                except:
                    pass
                return response
            except:
                api.abort(500)
        else:
            api.abort(404)

    @login_required
    @authorize()
    @api.expect(parsers.download_parser)
    def post(self, dataset):
        '''get sample of a configured dataset, number of rows being configured in connector.samples'''
        ds = Dataset(dataset)
        try:
            args = parsers.download_parser.parse_args()
            size = args['size']
            format_type = args['type']
            if size == None:
                size = ds.connector.sample
            if format_type == None:
                format_type = 'json'
        except:
            size = ds.connector.sample
            format_type = 'json'
        print(("args {} {}".format(format_type,size)))
        if (ds.connector.type == "elasticsearch"):
            if (ds.random_view == True):
                ds.select = {"query": {"function_score": {
                    "query": ds.select["query"], "random_score": {}}}}
        elif (ds.connector.type == "sql"):
            if (ds.select == None):
                ds.select = "select * from {}".format(ds.table)
        ds.init_reader(test=True)
        try:
            df = next(ds.reader, "")['df']
            schema = df.dtypes.apply(lambda x: str(x)).to_dict()
            if (type(df) == str):
                try:
                    return {"data": [{"error": "error: no such file {}".format(ds.file)}]}
                except:
                    return {"data": [{"error": "error: no such table {}".format(ds.table)}]}
            df = df.head(n=size).reset_index(drop=True)
            df = df.applymap(lambda x: unicode_safe(x))
            if (format_type == 'json'):
                return {"data": list(df.fillna("").T.to_dict().values()), "schema": schema}
            elif (format_type == 'csv'):
                return df.to_csv(index=False, encoding = "utf8" if ds.encoding == None else ds.encoding)
        except:
            error = err()
            try:
                return {"data": [{"error": "error: {} {}".format(error, ds.file)}]}
            except:
                return {"data": [{"error": "error: {} {}".format(error, ds.table)}]}

    @login_required
    @authorize(right="delete")
    def delete(self, dataset):
        '''delete the content of a dataset (currently only working on elasticsearch datasets)'''
        ds = Dataset(dataset)
        if (ds.connector.type == "elasticsearch"):
            try:
                ds.connector.es.indices.delete(
                    index=ds.table, ignore=[400, 404])
                config.log.write(
                    "detete {}:{}/{}".format(ds.connector.host, ds.connector.port, ds.table))
                ds.connector.es.indices.create(index=ds.table)
                config.log.write(
                    "create {}:{}/{}".format(ds.connector.host, ds.connector.port, ds.table))
                return {"status": "ok"}
            except:
                return {"status": "ko - " + err()}
        else:
            return api.abort(403)


@api.route('/datasets/<dataset>/<action>', endpoint='datasets/<dataset>/<action>')

class pushToValidation(Resource):

    @login_required
    @authorize()
    def get(self, dataset, action):
        '''action = validation : get text/yaml source code including the dataset (warning: a yaml source may include other resources)'''
        if (action == "yaml"):
            try:
                project = config.conf["datasets"][dataset]["project"]
                file = config.conf["datasets"][dataset]["source"]
                pfile = os.path.join(config.conf["global"]["projects"][
                    project]["path"], file)
                with open(pfile) as f:
                    return Response(f.read(), mimetype="text/plain")
            except:
                api.abort(503, {"error": err()})

    @login_required
    @authorize()
    def delete(self, dataset, action):
        '''delete text/yaml source code including the dataset (warning: a yaml source may include other resources'''
        if (action == "yaml"):
            try:
                project = config.conf["datasets"][dataset]["project"]
                file = config.conf["datasets"][dataset]["source"]
                pfile = os.path.join(config.conf["global"]["projects"][
                    project]["path"], file)
                os.remove(pfile)
                return {"file": file, "status": "deleted"}
            except:
                api.abort(503, {"error": err()})

    @login_required
    @authorize(right="update")
    def put(self, dataset, action):
        '''action = validation : configure the frontend to point to this dataset / action = search : search within dataset'''
        import config
        config.init()
        config.read_conf()
        if (action == "validation"):
            if (not(dataset in list(config.conf["datasets"].keys()))):
                return api.abort(404, {"dataset": dataset, "status": "dataset not found"})
            if not("validation" in list(config.conf["datasets"][dataset].keys())):
                return api.abort(403, {"dataset": dataset, "status": "validation not allowed"})
            if ((config.conf["datasets"][dataset]["validation"] == True) | (isinstance(config.conf["datasets"][dataset]["validation"], OrderedDict))):
                try:
                    props = {}
                    try:
                        cfg = deepupdate(config.conf["global"]["validation"], config.conf[
                            "datasets"][dataset]["validation"])
                    except:
                        cfg = config.conf["global"]["validation"]
                    for conf in list(cfg.keys()):
                        configfile = os.path.join(config.conf["global"]["paths"][
                            "validation"], secure_filename(conf + ".json"))
                        dic = {
                            "domain": config.conf["global"]["api"]["domain"],
                            "es_proxy_path": config.conf["global"]["api"]["es_proxy_path"],
                            "dataset": dataset
                        }
                        props[conf] = replace_dict(cfg[conf], dic)
                        print(conf)
                    print({"dataset": dataset, "status": "to validation", "props": props})
                    return {"dataset": dataset, "status": "to validation", "props": props}
                except:
                    return api.abort(500, {"dataset": dataset, "status": "error: " + err()})
            else:
                return api.abort(403, {"dataset": dataset, "status": "validation not allowed"})
        elif (action == "search"):
            if (not(dataset in list(config.conf["datasets"].keys()))):
                return api.abort(404, {"dataset": dataset, "status": "dataset not found"})
            if not("search" in list(config.conf["datasets"][dataset].keys())):
                return api.abort(403, {"dataset": dataset, "status": "search not allowed"})
            if ((config.conf["datasets"][dataset]["search"] == True) | (isinstance(config.conf["datasets"][dataset]["search"], OrderedDict))):
                try:
                    props = {}
                    try:
                        cfg = deepupdate(config.conf["global"]["search"], config.conf[
                            "datasets"][dataset]["search"])
                    except:
                        cfg = config.conf["global"]["search"]
                    for config in list(cfg.keys()):
                        configfile = os.path.join(config.conf["global"]["paths"][
                            "search"], secure_filename(config + ".json"))
                        dic = {
                            "domain": config.conf["global"]["api"]["domain"],
                            "es_proxy_path": config.conf["global"]["api"]["es_proxy_path"],
                            "dataset": dataset
                        }
                        props[config] = replace_dict(cfg[config], dic)
                        # with open(configfile, 'w') as outfile:
                        #     json.dump(props[config],outfile,indent=2)
                    return {"dataset": dataset, "status": "to search", "props": props}
                except:
                    return api.abort(500, {"dataset": dataset, "status": "error: " + err()})
            else:
                return api.abort(403, {"dataset": dataset, "status": "search not allowed"})

        else:
            api.abort(404)

    @login_required
    @authorize()
    @api.expect(parsers.live_parser)
    def post(self, dataset, action):
        '''action = _search : proxy _search api for elasticsearch dataset'''
        if ((action == "_search")):
            try:
                args = parsers.es_parser.parse_args()
                ds = Dataset(dataset)
                query = request.get_json()
                if (ds.connector.type == "elasticsearch"):
                    try:
                        ds.select = {"query": {"function_score": {
                            "query": query["query"], "random_score": {}}}}
                    except:
                        ds.select = query
                    try:
                        size = args['size']
                    except:
                        size = ds.chunk
                    try:
                        # hack for speed up an minimal rendering on object
                        resp = original_flask_make_response(json.dumps(ds.connector.es.search(
                            body=ds.select, index=ds.table, size=size)))
                        resp.headers['Content-Type'] = 'application/json'
                        return resp
                    except:
                        return api.abort(403, err())
                else:
                    api.abort(403, "not an elasticsearch dataset")
            except:
                return {"status": "ko - " + err()}
        else:
            api.abort(403)


@api.route('/datasets/<dataset>/<doc_type>/<id>/<action>', endpoint='datasets/<dataset>/<doc_type>/<id>/<action>')
class pushToValidation(Resource):

    @login_required
    @authorize(right="update")
    @api.expect(parsers.es_parser)
    def post(self, dataset, doc_type, id, action):
        '''elasticsearch update api proxy'''
        if ((action == "_update")):
            try:
                args = parsers.es_parser.parse_args()
                ds = Dataset(dataset)
                data = request.get_json()
            except:
                return {"status": "ko - " + err()}
            if (ds.connector.type == "elasticsearch"):
                try:
                    # hack for speed up an minimal rendering on object
                    resp = original_flask_make_response(json.dumps(ds.connector.es.update(
                        body=data, index=ds.table, id=id, doc_type=doc_type)))
                    resp.headers['Content-Type'] = 'application/json'
                    return resp
                except:
                    return api.abort(403, err())
            else:
                api.abort(403, "not an elasticsearch dataset")
        else:
            api.abort(403)


@api.route('/recipes/', endpoint='recipes')
class ListRecipes(Resource):

    @login_required
    def get(self):
        '''get json of all configured recipes'''
        response = {recipe: config.conf["recipes"][recipe]
                         for recipe in config.conf["recipes"]
                         if (check_rights(current_user, config.conf["recipes"][recipe]["project"], "read"))
                         }
        return response


@api.route('/recipes/<recipe>/', endpoint='recipes/<recipe>')
class RecipeApi(Resource):

    @login_required
    @authorize()
    def get(self, recipe):
        '''get json of a configured recipe'''
        try:
            return config.conf["recipes"][recipe]
        except:
            api.abort(404)


@api.route('/recipes/<recipe>/<action>', endpoint='recipes/<recipe>/<action>')
class RecipeRun(Resource):

    @login_required
    @authorize()
    def get(self, recipe, action):
        '''retrieve information on a recipe
        ** action ** possible values are :
        - ** yaml ** : get text/yaml code including the recipe
        - ** status ** : get status (running or not) of a recipe
        - ** log ** : stream log of running recipe, or returns last log'''
        if (action == "yaml"):
            try:
                project = config.conf["recipes"][recipe]["project"]
                file = config.conf["recipes"][recipe]["source"]
                pfile = os.path.join(config.conf["global"]["projects"][
                    project]["path"], file)
                with open(pfile) as f:
                    return Response(f.read(), mimetype="text/plain")
            except:
                api.abort(503, {"error": err()})

        if (action == "status"):
            # get status of job
            try:
                logfiles = [f
                            for f in os.listdir(config.conf["global"]["log"]["dir"])
                            if re.match(r'^.*-' + recipe + '.log$', f)]
                logfiles.sort(reverse=True)
                if (len(logfiles) == 0):
                    return {"recipe": recipe, "status": "down"}
                if ((time.time() - os.stat(os.path.join(config.conf["global"]["log"]["dir"],logfiles[0])).st_mtime) < 5):
                    return {"recipe": recipe, "status": "up"}
                return {"recipe": recipe, "status": "down"}
                # still bogus:
                # return {"recipe":recipe, "status":
                # config.jobs_list[str(recipe)]}
            except:
                return {"recipe": recipe, "status": "down"}
        elif (action == "log"):
            # get logs
            try:
                # try if there is a current log
                file = config.jobs[recipe].log.file
                open(file, 'r')
            except:
                try:
                    # search for a previous log
                    # check if recipe is declared
                    a = config.conf["recipes"][recipe]
                    logfiles = [os.path.join(config.conf["global"]["log"]["dir"], f)
                                for f in os.listdir(config.conf["global"]["log"]["dir"])
                                if re.match(r'^.*-' + recipe + '.log$', f)]
                    logfiles.sort(reverse=True)
                    if (len(logfiles) == 0):
                        return Response("", mimetype="text/plain")
                    file = logfiles[0]
                except:
                    return Response("", mimetype="text/plain")
            try:
                if ((time.time() - os.stat(os.path.join(config.conf["global"]["log"]["dir"],file)).st_mtime) >= 5):
                    with open(file, 'r') as f:
                        response = f.read()
                        # old log : return it full
                        return Response(response, mimetype="text/plain")
                    # return {"hop": "la"}
                else:
                    def tailLog(file):
                        # method for tail -f file
                        f = open(file,'r')
                        yield 'retry: 3000\n'
                        yield 'event: message\n' + re.sub("^", "data: ", f.read()[:-1], flags = re.M) + '\n\n'
                        #Find the size of the file and move to the end
                        st_results = os.stat(file)
                        st_size = st_results[6]
                        f.seek(st_size)
                        wait = 0
                        while wait < 5:
                            where = f.tell()
                            line = f.readline()
                            if not line:
                                wait += 1
                                time.sleep(1)
                                f.seek(where)
                            else:
                                wait = 0
                                yield 'event: message\n'+'data: ' + line + '\n'

                        yield 'event: close\ndata: end\n\n'
                    response = Response(stream_with_context(tailLog(file)), mimetype = "text/event-stream")
                    response.headers['X-Accel-Buffering'] = 'no'
                    return response
            except:
                return Response(str(err()), mimetype="text/plain")
        api.abort(403)

    @login_required
    @authorize(right="update")
    @api.expect(parsers.live_parser)
    def post(self, recipe, action):
        '''apply recipe on posted data
        ** action ** possible values are :
        - ** apply ** : apply recipe on posted data
        '''
        if (action == "apply"):
            args = parsers.live_parser.parse_args()
            file = args['file']
            if not (allowed_upload_file(file.filename)):
                api.abort(403)
            r = Recipe(recipe)
            r.input.chunked = False
            r.input.file = file.stream
            r.init(test=True)
            r.run()
            if isinstance(r.df, pd.DataFrame):
                df = r.df.fillna("")
                try:
                    return jsonify({"data": list(df.T.to_dict().values()), "log": str(r.log.writer.getvalue())})
                except:
                    df = df.applymap(lambda x: str(x))
                    return jsonify({"data": list(df.T.to_dict().values()), "log": str(r.log.writer.getvalue())})
            else:
                return {"log": r.log.writer.getvalue()}

    @login_required
    @authorize(right="update")
    def put(self, recipe, action):
        '''test, run or stop recipe
        ** action ** possible values are :
        - ** test ** : test recipe on sample data
        - ** run ** : run the recipe
        - ** stop ** : stop a running recipe (soft kill : it may take some time to really stop)
        '''
        config.read_conf()
        if (action == "test"):
            try:
                callback = config.manager.dict()
                r = Recipe(recipe)
                r.init(test=True, callback=callback)
                r.set_job(Process(target=thread_job, args=[r]))
                r.start_job()
                r.join_job()
                r.df = r.callback["df"]
                r.log = r.callback["log"]
                r.errors = r.callback["errors"]

            except:
                return {"data": [{"result": "failed"}], "log": "Ooops: {}".format(err())}
            if isinstance(r.df, pd.DataFrame):
                df = r.df.fillna("")
                if (r.df.shape[0] == 0):
                    return {"data": [{"result": "empty"}], "log": r.callback["log"]}
                try:
                    return jsonify({"data": list(df.T.to_dict().values()), "log": r.callback["log"]})
                except:
                    df = df.applymap(lambda x: unicode_safe(x))
                    return jsonify({"data": list(df.T.to_dict().values()), "log": r.callback["log"]})
            else:
                return {"data": [{"result": "empty"}], "log": r.callback["log"]}
        elif (action == "run"):
            # run recipe (gives a job)
            try:
                if (recipe in list(config.jobs.keys())):
                    status = config.jobs[recipe].job_status()
                    if (status == "up"):
                        return {"recipe": recipe, "status": status}
            except:
                api.abort(403)

            config.jobs[recipe] = Recipe(recipe)
            config.jobs[recipe].init()
            config.jobs[recipe].set_job(
                Process(target=thread_job, args=[config.jobs[recipe]]))
            config.jobs[recipe].start_job()
            return {"recipe": recipe, "status": "new job"}
        elif (action == "stop"):
            try:
                if (recipe in list(config.jobs.keys())):
                    thread = Process(config.jobs[recipe].stop_job())
                    thread.start()
                    return {"recipe": recipe, "status": "stopping"}
            except:
                api.abort(404)

    @login_required
    @authorize()
    def delete(self, recipe, action):
        '''delete text/yaml source code including the recipe (warning: a yaml source may include other resources)'''
        if (action == "yaml"):
            try:
                project = config.conf["recipes"][recipe]["project"]
                file = config.conf["recipes"][recipe]["source"]
                pfile = os.path.join(config.conf["global"]["projects"][
                    project]["path"], file)
                os.remove(pfile)
                return {"file": file, "status": "deleted"}
            except:
                api.abort(503, {"error": err()})

@api.route('/jobs/', endpoint='jobs')
class jobsList(Resource):

    @login_required
    def get(self):
        '''retrieve jobs list
        '''
        response = {"running": [], "done": []}
        logfiles = [f
                    for f in os.listdir(config.conf["global"]["log"]["dir"])
                    if re.match(r'^.*.log$', f)]
        logfiles.sort(reverse=True)
        for file in logfiles:
            if ((time.time() - os.stat(os.path.join(config.conf["global"]["log"]["dir"],file)).st_mtime) < 5):
                running = True
            else:
                running = False
            recipe = re.search(".*-(.*?).log", file, re.IGNORECASE).group(1)

            date = re.sub(
                r"(\d{4}.?\d{2}.?\d{2})T(..:..).*log", r"\1-\2", file)
            if (recipe in list(config.conf["recipes"].keys())):
                if (check_rights(current_user, config.conf["recipes"][recipe]["project"], "read")):
                    if running:
                        response["running"].append(
                            {"recipe": recipe, "date": date, "file": file})
                    else:
                        response["done"].append(
                            {"recipe": recipe, "date": date, "file": file})

        return response

if __name__ == '__main__':
    config.read_conf()
    try:
        app.config['DEBUG'] = (str(config.conf["global"]["api"]["debug"]) == "True")
    except:
        pass

    config.log = Log("main")

    # recipe="dataprep_snpc"
    # r=Recipe(recipe)
    # r.init()
    # r.run()

    # Load a dummy app at the root URL to give 404 errors.
    # Serve app at APPLICATION_ROOT for localhost development.

    application = DispatcherMiddleware(Flask('dummy_app'), {
        app.config['APPLICATION_ROOT']: app,
    })

    try:
        BACKEND_PORT = int(config.conf["global"]["api"]["port"])
    except:
        BACKEND_PORT = 8081

    run_simple(config.conf["global"]["api"]["host"],
               BACKEND_PORT,
               application,
               threaded = config.conf["global"]["api"]["threaded"],
               processes = config.conf["global"]["api"]["processes"],
               use_reloader = (str(config.conf["global"]["api"]["use_reloader"])=="True"))
