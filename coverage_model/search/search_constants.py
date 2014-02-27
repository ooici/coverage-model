__author__ = 'casey'


def constant(f):
    def fset(self, value):
        raise SyntaxError("Cannot set this value")
    def fget(self):
        return f()
    return property(fget,fset)


class SearchParameterNames(object):
    @constant
    def TIME():
        return 'time'

    @constant
    def LAT():
        return 'lat'

    @constant
    def LON():
        return 'lon'

    @constant
    def VERTICAL():
        return 'vertical'

    @constant
    def GEO_BOX():
        return 'geo_box'

    @constant
    def INTERNAL_TIME_KEY():
        return 'internal_time'

    @constant
    def GPS_LAT_KEY():
        return 'gps_lat'

    @constant
    def GPS_LON_KEY():
        return 'gps_lon'

    @constant
    def AT_LEAST_ONE_REQUIRED():
        names = SearchParameterNames()
        return [ names.TIME, names.LAT, names.LON, names.GEO_BOX, names.VERTICAL ]


class IndexParameterNames(object):
    @constant
    def INTERNAL_TIME_KEY():
        return 'time'

    @constant
    def TIME_KEY():
        return 'time'

    @constant
    def VERTICAL():
        return 'vertical'

    @constant
    def LAT_KEY():
        return 'lat'

    @constant
    def LON_KEY():
        return 'lon'