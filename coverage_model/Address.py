__author__ = 'casey'

from ast import literal_eval

@staticmethod
def from_tuple_str(str):
    tup = literal_eval(str)
    if len(tup) > 1:
        address_type = tup[0]
        if address_type is 'Address':
            return Address.from_tuple_str(str)
        if address_type is 'FileAddress':
            return FileAddress.from_tuple_str(str)
        if address_type is 'BrickAddress':
            return BrickAddress.from_tuple_str(str)
        if address_type is 'BrickFileAddress':
            return BrickFileAddress.from_tuple_str(str)
        else:
            raise ValueError("".join(["Do not know how to build address type: ", tup[0]]))


class Address(object):

    def __init__(self, coverage_uid):
        self.coverage_uid = coverage_uid
        pass

    def get_top_level_key(self):
        raise NotImplementedError('Not implemented by base class')

    def as_tuple_str(self):
        tup = "Address", self.coverage_uid
        return str(tup)

    @staticmethod
    def from_tuple_str(str):
        tup = literal_eval(str)
        if len(tup) != 1:
            raise ValueError("".join(["Expected tuple of size 1.  Found ", str(tup)]))
        if tup[0] == "Address":
            return Address(tup[0])
        else:
            raise ValueError("".join(["Do not know how to build address type: ", tup[0]]))

import os


class FileAddress(Address):
    def __init__(self, coverage_uid, file_path, begin=0, end=-1, validate=False):
        Address.__init__(self, coverage_uid)
        if validate:
            if not os.path.exists(file_path):
                raise ValueError("".join(["File does not exist at path: ", file_path]))
        self.file_path = file_path
        self.begin = begin
        self.end = end

    def as_tuple_str(self):
        tup = "FileAddress", self.coverage_uid, self.file_path, self.begin, self.end
        return str(tup)

    @staticmethod
    def from_tuple_str(str):
        tup = literal_eval(str)
        if len(tup) != 5:
            raise ValueError("".join(["Expected tuple of size 5.  Found ", str(tup)]))
        if tup[0] == "FileAddress":
            return FileAddress(tup[1], tup[2], tup[3], tup[4])
        else:
            raise ValueError("".join(["Do not know how to build address type: ", tup[0]]))

    def get_top_level_key(self):
        return self.file_path

    def __lt__(self, other):
        return self.__key__() < other.__key__()

    def __key__(self):
        return self.file_path, self.begin, self.end

    def __hash__(self):
        return hash(self.__key__())

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self.__key__())


class BrickAddress(Address):
    def __init__(self, coverage_uid, brick_id, brick_slice):
        Address.__init__(self, coverage_uid)
        self.brick_id = brick_id
        self.brick_slice = brick_slice

    def as_tuple_str(self):
        tup = "BrickAddress", self.coverage_uid, self.brick_id, self.brick_slice
        return str(tup)

    @staticmethod
    def from_tuple_str(src_str):
        tup = literal_eval(src_str)
        if len(tup) != 4:
            raise ValueError("".join(["Expected tuple of size 5.  Found ", src_str(tup)]))
        if tup[0] == "BrickAddress":
            return BrickAddress(tup[1], tup[2], tup[3])
        else:
            raise ValueError("".join(["Do not know how to build address type: ", tup[0]]))

    def get_top_level_key(self):
        return self.coverage_uid, self.brick_id

    def __lt__(self, other):
        return self.__key__() < other.__key__()

    def __key__(self):
        return "BrickAddress", self.coverage_uid, self.brick_id, self.brick_slice

    def __hash__(self):
        return hash(self.__key__())

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self.__key__())


class BrickFileAddress(Address):
    def __init__(self, coverage_uid, brick_id):
        Address.__init__(self, coverage_uid)
        self.brick_id = brick_id

    def as_tuple_str(self):
        tup = "BrickFileAddress", self.coverage_uid, self.brick_id
        return str(tup)

    @staticmethod
    def from_tuple_str(src_str):
        tup = literal_eval(src_str)
        if len(tup) != 4:
            raise ValueError("".join(["Expected tuple of size 5.  Found ", src_str(tup)]))
        if tup[0] == "BrickFileAddress":
            return BrickFileAddress(tup[1], tup[2], tup[3])
        else:
            raise ValueError("".join(["Do not know how to build address type: ", tup[0]]))

    def get_top_level_key(self):
        return self.coverage_uid, self.brick_id

    def __lt__(self, other):
        return self.__key__() < other.__key__()

    def __key__(self):
        return "BrickFileAddress", self.coverage_uid, self.brick_id

    def __hash__(self):
        return hash(self.__key__())

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self.__key__())