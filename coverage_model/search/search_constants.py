__author__ = 'casey'


def enum(**enums):
    return type('Enum', (), enums)

IndexedParameters = enum(Time='time', Latitude='lat', Longitude='lon', Vertical='vertical', GeoBox='geo_box', CoverageId='coverage_id')

MinimumOneParameterFrom = frozenset([IndexedParameters.Time,
                                     IndexedParameters.Latitude,
                                     IndexedParameters.Longitude,
                                     IndexedParameters.GeoBox,
                                     IndexedParameters.Vertical,
                                     IndexedParameters.CoverageId])

AllowedSearchParameters = frozenset([IndexedParameters.Time,
                                     IndexedParameters.Latitude,
                                     IndexedParameters.Longitude,
                                     IndexedParameters.GeoBox,
                                     IndexedParameters.Vertical,
                                     IndexedParameters.CoverageId])
