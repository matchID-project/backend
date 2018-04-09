#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import hashlib
import config
from config import Configured
from tools import *
from log import Log, err


class Group(Configured):

    def __init__(self, name=None):
        Configured.__init__(self, "groups", name)


class User(Configured):

    def __init__(self, name=None):
        Configured.__init__(self, "users", name)

        try:
            self.display_name = self.conf["display_name"]
        except:
            self.display_name = self.name

        try:
            self.groups = self.conf["groups"]
        except:
            self.groups = []
        try:
            self.password = self.conf["password"]
        except:
            self.password = None

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
