"""
  --------------------------------------------------
  File Name : preprocessing.py
  Creation Date : 26-05-2018
  Last Modified : 2019-01-28 Mon 02:15 pm
  Created By : Joonatan Samuel
  --------------------------------------------------
"""

from pprint import pprint
from sklearn_pandas import DataFrameMapper

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler

from .utils import *
from sklearn.base import BaseEstimator, TransformerMixin

class Mapper(DataFrameMapper):
    """Docstring for Mapper. """
    def __init__(self, df, force_mappings=None):
        """
        :df: Pandas dataframe that will be mutated with transform
        :force_mappings: dictionary of col_name: FeatureMapperClass()
        EXAMPLE:

        from pprint import pprint
        from sklearn.preprocessing import MultiLabelBinarizer

        df = w2v.transform( ["some random string to encode into embedding: We propose two efficient approximations to standard convolutional neural networks: Binary-Weight-Networks and XNOR-Networks. In Binary-WeightNetworks, the filters are approximated with binary values resulting in 32 memory saving. In XNOR-Networks, both the filters and the input to convolutional layers are binary. XNOR-Networks approximate convolutions using primarily binary operations"])

        mapper = preprocessing.Mapper( df , force_mappings= { 'word_id': MultiLabelBinarizer()})
        mapper.transform( df )


        """

        self._force_mappings = {} if force_mappings is None else force_mappings
        self._map = [ ]
        self._groups = {}

        _groups = df.columns.to_series().groupby(df.dtypes).groups

        # returns data in the form of:
        # {'object': ['C', 'D'], 'int64': ['A', 'E'], 'float64': ['B']}

        self._groups = {k.name: v for k, v in _groups.items()}

        # Dataframe mapper is usually used like this
        # >>> mapper = DataFrameMapper([
        # ...     ('pet', sklearn.preprocessing.LabelBinarizer()),
        # ...     (['children'], sklearn.preprocessing.StandardScaler())
        # ... ])

        pprint( self._groups )

        def last_element_info_print(dtype):
            if len( self._map) <= 0:
                eprint("[ERROR] self._map is of length 0")
            else:
                col_name, transform_class = self._map[-1]
                eprint("[INFO] column: %s interpreted as %s and %s was applied" % (col_name, dtype, transform_class.__class__.__name__))

        for col_name in df.columns:
            _dtype = None

            if col_name in self._force_mappings:
                self._map.append( ([ col_name ], self._force_mappings[col_name] ) )
                last_element_info_print('force_mapping')
                continue

            for dtype, col_names in self._groups.items():
                if col_name in col_names:
                    if dtype.startswith('int'):
                        unique_elements = df[col_name].unique()
                        if len(unique_elements) > len(df) / 10. or len(unique_elements) > 100:
                            eprint("[WARNING] column \"%s\" has %d number of labels, are you sure this is correct?" % ( col_name, len(unique_elements) ))
                            _dtype = 'int'
                            self._map.append( (col_name, None ) )
                        else:
                            _dtype = 'input as int, redefined to category'
                            self._map.append( ([ col_name ], MultiLabelBinarizer() ) )
                    elif dtype.startswith('float'):
                        _dtype = 'float'
                        self._map.append( ([ col_name ], StandardScaler()) )

                    elif dtype.startswith('categorical'):
                        _dtype = 'categorical'
                        unique_elements = df[col_name].unique()
                        if len(unique_elements) > len(df) / 10. or len(unique_elements) > 100:
                            eprint("[WARNING] column \"%s\" has %d number of labels, are you sure this is correct?" % ( col_name, len(unique_elements )))

                        self._map.append( ([ col_name ], MultiLabelBinarizer() ) )

                    elif dtype.startswith('string'):
                        _dtype = 'string'
                        unique_elements = df[col_name].unique()

                        if unique_elements > len(df) / 10. or unique_elements > 100:
                            eprint("[WARNING] column \"%s\" has %d number of labels, are you sure this is correct?" % ( col_name, unique_elements ))

                        self._map.append( (col_name, MultiLabelBinarizer() ) )

                    elif dtype.startswith('object'):
                        raise RuntimeError(" %s.%s failed, because column %s is of datatype object" % (self.__class__.__name__,  inspect.currentframe().f_code.co_name, col_name)  )
            last_element_info_print(_dtype)

        super(Mapper, self).__init__(features=self._map)
        super(Mapper, self).fit( df )
