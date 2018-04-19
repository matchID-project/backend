#!/usr/bin/env python2
# -*- coding: utf-8 -*-


import os
import yaml
import hashlib
import config
from config import Configured
from tools import *
from log import Log, err
from api import app, api

def check_rights(user, project, right):
    user = user.name
    test = [group for
            group in config.conf["groups"] if check_rights_groups(group, user, project, right)]
    return (len(test) > 0)


def check_rights_groups(group, user, project, right):
    group = Group(group)
    try:
        for p in ["_all", project]:
            if (p in group.projects.keys()):
                for u in ["_all", user]:
                    for r in group.projects[p].keys():
                        try:
                            if (u in group.projects[p][r].keys()):
                                r = Role(r)
                                if r.right[right] == True:
                                    return True
                        except:
                            if (u == group.projects[p][r]):
                                r = Role(r)
                                if r.right[right] == True:
                                    return True
    except:
        config.log.write(err())
    return False


class Group(Configured):

    def __init__(self, name=None):
        Configured.__init__(self, "groups", name)

        try:
            self.projects = self.conf["projects"]
        except:
            self.projects = {}


class User(Configured):

    def __init__(self, name=None, social_id=None, email=None, provider=None):
        if social_id == None:
            Configured.__init__(self, "users", name)
        else:
            try:
                Configured.__init__(self, "users", name)
            except:
                self.name = name
                creds_file = os.path.join(config.conf["global"][
                    "paths"]["conf"], 'security', provider+'.yml')
                u = {
                            'users': {
                                str(name): {
                                    'social_id': social_id,
                                    'provider': str(provider)
                                }
                            }
                        }
                if email != None:
                    u['users'][str(name)]['email'] = str(email)
                with open(creds_file, 'w') as f:
                    yaml.dump(u, f)

        try:
            self.display_name = self.conf["display_name"]
        except:
            self.display_name = self.name

        try:
            self.password = self.conf["password"]
        except:
            self.password = None

        try:
            self.email = self.conf["email"]
        except:
            self.email = email

        try:
            self.social_id = self.conf["social_id"]
        except:
            self.social_id = social_id

        try:
            self.social_provider = self.conf["social_provider"]
        except:
            self.social_provider = None

        self.auth = False
        self.active = False
        self.anonymous = False

    def check_password(self, password):
        return (self.password == password)

    def is_authenticated(self):
        return self.auth

    def is_active(self):
        return self.active

    def is_anonymous(self):
        return self.anonymous

    def get_id(self):
        return self.name


class Role(Configured):

    def __init__(self, name=None):
        Configured.__init__(self, "roles", name)

        self.right = {}

        for right in ["create", "read", "update", "delete"]:
            try:
                self.right[right] = self.conf[right]
            except:
                self.right[right] = False


class OAuthSignIn(object):
    providers = None

    def __init__(self, provider_name):
        self.provider_name = provider_name
        credentials = current_app.config['OAUTH_CREDENTIALS'][provider_name]
        self.consumer_id = credentials['id']
        self.consumer_secret = credentials['secret']

    def authorize(self):
        pass

    def callback(self):
        pass

    def get_callback_url(self):
        return url_for('oauth_callback', provider=self.provider_name,
                       _external=True)

    @classmethod
    def get_provider(self, provider_name):
        if self.providers is None:
            self.providers = {}
            for provider_class in self.__subclasses__():
                provider = provider_class()
                self.providers[provider.provider_name] = provider
        return self.providers[provider_name]
