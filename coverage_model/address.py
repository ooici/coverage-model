__author__ = 'casey'

import ast

class AddressFactory(object):

    @staticmethod
    def from_db_str(st):
        try:
            if len(st) > 0 and ':::' in st:
                s = st.split(":::")
                if s[0] == BrickAddress.__name__:
                    return BrickAddress.from_db_str(st)
                elif s[0] == BrickFileAddress.__name__:
                    return BrickFileAddress.from_db_str(st)
                elif s[0] == FileAddress.__name__:
                    return FileAddress.from_db_str(st)
                elif s[0] == Address.__name__:
                    return Address.from_db_str(st)
        except Exception as ex:
            pass
        raise ValueError("Do not know how to build address from string: %s", st)

    @staticmethod
    def from_str(st):
        try:
            d = ast.literal_eval(st)
            if isinstance(d, dict):
                if 'type' in d:
                    t = d['type']
                    if t == BrickAddress.__name__:
                        return BrickAddress.from_dict(d)
                    elif t == BrickFileAddress.__name__:
                        return BrickFileAddress.from_dict(d)
                    elif t == FileAddress.__name__:
                        return FileAddress.from_dict(d)
                    elif t == Address.__name__:
                        return Address.from_dict(d)
        except Exception as ex:
            pass
        raise ValueError("Do not know how to build address from string: %s", st)


    @staticmethod
    def from_tuple(tup):
        if len(tup) > 1:
            address_type = tup[0]
            if address_type == Address.__name__:
                return Address.from_tuple(tup)
            elif address_type == FileAddress.__name__:
                return FileAddress.from_tuple(tup)
            elif address_type == BrickAddress.__name__:
                return BrickAddress.from_tuple(tup)
            elif address_type == BrickFileAddress.__name__:
                return BrickFileAddress.from_tuple(tup)
            else:
                raise ValueError("".join(["Do not know how to build address type: ", tup[0]]))


class Address(object):

    def __init__(self, coverage_uid):
        self.coverage_uid = coverage_uid
        pass

    def get_top_level_key(self):
        raise NotImplementedError('Not implemented by base class')

    def as_dict(self):
        return {'type': Address.__name__,
                'coverage_uid': self.coverage_uid}

    @staticmethod
    def from_dict(dic):
        if 'type' in dic and dic['type'] == Address.__name__:
            if 'coverage_uid' in dic:
                return Address(dic['coverage_uid'])
        raise ValueError("Do not know how to build address from %s ", str(dic))

    def as_tuple(self):
        tup = "Address", self.coverage_uid
        return tup

    @staticmethod
    def from_tuple(tup):
        if len(tup) != 1:
            raise ValueError("".join(["Expected tuple of size 1.  Found ", str(tup)]))
        if tup[0] == "Address":
            return Address(tup[0])
        else:
            raise ValueError("".join(["Do not know how to build address type: ", tup[0]]))

    @staticmethod
    def from_str(st):
        return Address.from_dict(ast.literal_eval(st))

    def get_top_level_key(self):
        return self.coverage_uid

    def __lt__(self, other):
        return self.__key__() < other.__key__()

    def __eq__(self, other):
        return self.as_dict() == other.as_dict()

    def __key__(self):
        return self.as_dict()

    def __hash__(self):
        return hash(self.__key__())

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self.__key__())
import os


class IDAddress(Address):
    def __init__(self, id):
        Address.__init__(self, id)
        self.id

    def as_dict(self):
        return {'type': IDAddress.__name__,
                'id': self.id}

    @staticmethod
    def from_dict(dic):
        if 'type' in dic and dic['type'] == IDAddress.__name__:
            if 'id' in dic:
                return IDAddress(dic['id'])
        raise ValueError("Do not now how to build %s from %s" % (IDAddress.__name__, str(dic)))

    def as_tuple(self):
        tup = IDAddress.__name__, self.id
        return tup

    @staticmethod
    def from_tuple(tup):
        if len(tup) != 2:
            raise ValueError("".join(["Expected tuple of size 2.  Found ", str(tup)]))
        if tup[0] == IDAddress.__name__:
            return IDAddress(tup[1])
        else:
            raise ValueError("".join(["Do not know how to build address type: ", tup[0]]))

    @staticmethod
    def from_str(st):
        return IDAddress.from_dict(ast.literal_eval(st))

    def get_top_level_key(self):
        return self.id


class FileAddress(Address):
    def __init__(self, coverage_uid, file_path, begin=0, end=-1, validate=False):
        Address.__init__(self, coverage_uid)
        if validate:
            if not os.path.exists(file_path):
                raise ValueError("".join(["File does not exist at path: ", file_path]))
        self.file_path = file_path
        self.begin = begin
        self.end = end

    def as_dict(self):
        return {'type': FileAddress.__name__,
                'coverage_uid': self.coverage_uid,
                'file_path': self.file_path,
                'begin': self.begin,
                'end': self.end}

    @staticmethod
    def from_dict(dic):
        if 'type' in dic and dic['type'] == FileAddress.__name__:
            if 'coverage_uid' in dic and 'file_path' in dic and 'begin' in dic and 'end' in dic:
                return FileAddress(dic['coverage_uid'], dic['file_path'], dic['begin'], dic['end'])
        raise ValueError("Do not know how to build address from %s ", str(dic))

    def as_tuple(self):
        tup = "FileAddress", self.coverage_uid, self.file_path, self.begin, self.end
        return tup

    @staticmethod
    def from_tuple(tup):
        if len(tup) != 5:
            raise ValueError("".join(["Expected tuple of size 5.  Found ", str(tup)]))
        if tup[0] == "FileAddress":
            return FileAddress(tup[1], tup[2], tup[3], tup[4])
        else:
            raise ValueError("".join(["Do not know how to build address type: ", tup[0]]))

    @staticmethod
    def from_str(st):
        return BrickAddress.from_dict(ast.literal_eval(st))

    def get_top_level_key(self):
        return self.file_path

    def __lt__(self, other):
        return self.__key__() < other.__key__()

    def __eq__(self, other):
        return self.as_dict() == other.as_dict()

    def __key__(self):
        return self.as_dict()

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

    def as_tuple(self):
        tup = "BrickAddress", self.coverage_uid, self.brick_id, self.brick_slice
        return tup

    def as_dict(self):
        return {'type': BrickAddress.__name__,
                'coverage_uid': self.coverage_uid,
                'brick_id': self.brick_id,
                'brick_slice': self.brick_slice}

    @staticmethod
    def from_dict(dic):
        if 'type' in dic and dic['type'] == BrickAddress.__name__:
            if 'coverage_uid' in dic and 'brick_id' in dic and 'brick_slice':
                return BrickAddress(dic['coverage_uid'], dic['brick_id'], dic['brick_slice'])
        raise ValueError("Do not know how to build address from %s ", str(dic))

    @staticmethod
    def from_tuple(tup):
        if len(tup) != 4:
            raise ValueError("".join(["Expected tuple of size 5.  Found ", str(len(tup))]))
        if tup[0] == "BrickAddress":
            return BrickAddress(tup[1], tup[2], tup[3])
        else:
            raise ValueError("".join(["Do not know how to build address type: ", tup[0]]))

    @staticmethod
    def from_str(st):
        return BrickAddress.from_dict(ast.literal_eval(st))

    def get_top_level_key(self):
        return self.coverage_uid, self.brick_id

    def __lt__(self, other):
        return self.__key__() < other.__key__()

    def __eq__(self, other):
        return self.as_dict() == other.as_dict()

    def __key__(self):
        return self.as_dict()

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

    def as_tuple(self):
        tup = "BrickFileAddress", self.coverage_uid, self.brick_id
        return tup

    def as_dict(self):
        return {'type': BrickFileAddress.__name__,
                'coverage_uid': self.coverage_uid,
                'brick_id': self.brick_id}

    @staticmethod
    def from_dict(dic):
        if 'type' in dic and dic['type'] == BrickFileAddress.__name__:
            if 'coverage_uid' in dic and 'brick_id' in dic:
                return BrickFileAddress(dic['coverage_uid'], dic['brick_id'])
        raise ValueError("Do not know how to build address from %s ", str(dic))

    @staticmethod
    def from_tuple(tup):
        if len(tup) != 3:
            raise ValueError("".join(["Expected tuple of size 5.  Found ", str(tup)]))
        if tup[0] == "BrickFileAddress":
            return BrickFileAddress(tup[1], tup[2])
        else:
            raise ValueError("".join(["Do not know how to build address type: ", tup[0]]))

    def get_db_str(self):
        return ''.join([BrickFileAddress.__name__, ':::',
                        self.coverage_uid, ':::', self.brick_id])

    @staticmethod
    def from_db_str(db_str):
        try:
            tp, cov_id, brick_id = db_str.split(":::")
            if tp == BrickFileAddress.__name__:
                return BrickFileAddress(cov_id, brick_id)
        except Exception as ex:
            pass
        raise ValueError("Do not know how to build address from %s ", str(db_str))

    @staticmethod
    def from_str(st):
        return BrickFileAddress.from_dict(ast.literal_eval(st))

    def get_top_level_key(self):
        return self.coverage_uid + "::" + self.brick_id

    def __lt__(self, other):
        return self.__key__() < other.__key__()

    def __eq__(self, other):
        return self.as_dict() == other.as_dict()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __key__(self):
        return str(self.as_dict())

    def __hash__(self):
        return hash(self.__key__())

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self.__key__())