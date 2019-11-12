from __future__ import division
"""
Main angira fuctions
Named Angira after https://en.wikipedia.org/wiki/Angiras_(sage)
"""
__author__ = "Swagatam M"
__copyright__ = "swagatam mukhopadhyay"
__credits__ = []
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Swagatam Mukhopadhyay"
__email__ = "swagatam.mukhopadhyay@gmail.com"
__status__ = "Dev"

# Initial imports
## STD
import os, sys, string
import warnings
import re
import six
import StringIO
import logging
from collections import defaultdict
import pandas as pd
import numpy as np
import warnings
from unidecode import unidecode  #you may need to pip install this
import numpy.ma as ma #masked arrays 
#LOCAL
from . import  validators as vals
from . import searchers as search


#####################################################################################################

def clean_text(r):
    # clean up M***f***g unicode!
    return (unidecode(r) if isinstance(r, six.string_types) else r)


#####################################################################################################
def column_splitter(DF, column, params = None):
    """
    A genral column splitter: splits columns using pattern (regex), taking care of Nones
    Args:
        DF
        column to split
        params:
            params.setdefault('pattern', '([-+]?\d*\.\d+|\d+)(\D*)') #this is the Number-Unit splitter
            params.setdefault('num_splits', 2)
            params.setdefault('map_col_names', None) #default will just be underscore column num like _1 _2
            params.setdefault('map_col_types', ['num', 'str'])

        Returns:
            Split column DataFrame, see params. for naming convention, returns original DF if nothing to split
    """
    if params is None:
        _params = dict()
    else:
        _params = params.copy()
    _params.setdefault('pattern', '([-+]?\d*\.\d+|\d+)(\D*)') #this is the num unit splitter
    _params.setdefault('num_splits', 2)
    _params.setdefault('map_col_names', None) #default will just be underscore column num like _1 _2
    _params.setdefault('map_col_types', ['num', 'str'])
    pattern = re.compile(_params['pattern'])
    inds = DF[column].notnull().values #Not null
    DFC = DF.copy()
    split_cols = []
    if np.any(inds):
        vals = DF.loc[inds, column].values.astype(str)
        matches = np.asarray([True if re.match(pattern, v.strip()) is not None else False for v in vals]) #pattern was found
        if np.any(matches):
            extracted = np.asarray([re.match(pattern, m.strip()).groups() for m in vals[matches]])
            assert np.shape(extracted)[1] == _params['num_splits'], "Num. splits inconsistent with pattern!"
            for i in range(_params['num_splits']):
                if _params['map_col_types'][i] == 'num':
                    col_loc = extracted[:, i].astype(float)
                elif _params['map_col_types'][i] == 'str':
                    col_loc = np.asarray(map(string.strip, extracted[:, i].astype(str)))
                else:
                    raise ValueError("unknown right column type, should be 'num' or 'str'")
                col_slot = np.asarray([None]*len(DF))
                col_slot[np.flatnonzero(inds)[np.flatnonzero(matches)]] = col_loc
                if _params['map_col_names'] is None:
                    col_name = column + '_' + str(i)
                else:
                    col_name = _params['map_col_names'][i]
                DFC[col_name] = col_slot
    return DFC


###############################################################################################################################

class Checker:
    """
    A general class for "Value" (data entries) type checks. Digests any function that takes in a list of entries and
    returns a list of True/False and associated error codes. A string function name is expected to be found in the function
    in the validator module.
    """
    def __init__(self, functname, params = None, error_code = ''):
        """
        Args:
            functname: a Checker function, takes in lst of values and returns a list of True/False and associated error codes
            params_dict: dict. of parameters
            error_code: Associated with this checker
        Returns:
            vector or True/False
            error_vec
        """
        self.params = params
        self.error_code = error_code
        if isinstance(functname, six.string_types):
            self.checker = lambda x: getattr(vals, functname)(x, params = params, error_code = error_code)
        else:
            self.checker = lambda x: functname(x, params = params, error_code = error_code)

    def __call__(self, vector):
        test_vec, error_vec = self.checker(vector)
        return test_vec, error_vec


###############################################################################################################################

class ValueValidator:
    """
    Acts on rows and columns, in a vectorized manner on a DF,
    always returning a DFnew, DFT, and error matrix: the three fundamental DFs in angira
    """
    def __init__(self, functname, params = None, error_code = ''):
        """
        Args: Args are typical config entry for valueValidators: ValidatorFx, params, errorcode
            functname: a Checker function
            params_dict: dict. of parameters
            error_code:

        """
        self.functname = functname
        self.params = params
        self.error_code = error_code
        self.validator = Checker(functname, params = params, error_code = error_code)


    def __call__(self, DF, rows, cols):
        """
        Call Validator:
            DF: Dataframe, NON-hierarcial indexing! Typically a cleaned table
            rows: list of rows
            cols: list of cols
        Return:
            DFnew: All tested but failed values set to None of the original DF,
                    untested values passed along
            DFT: True/False/None matrix, elements not tested are None
            error: error matrix, untested values are None

        """
        DFT = pd.DataFrame().reindex_like(DF)
        E = pd.DataFrame().reindex_like(DF)
        DFnew = DF.copy()
        assert (type(rows) == list) & (type(cols) == list), 'Rows and cols must be list'
        if (len(rows) == 0) and (len(cols) != 0): #only columnwise
            test  = set(cols) - set(DF.columns.values)
            assert len(test) == 0, "Cols " +  ' '.join(test) + ' does/do not exist in DF'
            M = DF.loc[:, cols].values
            X = np.ravel(M)
            test_vec, error_vec = self.validator(X)
            DFT.loc[:, cols] = test_vec.reshape(M.shape)
            E.loc[:, cols] = error_vec.reshape(M.shape)
            X[~test_vec] = None
            DFnew.loc[:, cols] = X.reshape(M.shape)

        elif (len(cols) == 0) and (len(rows)!= 0): # Only rowwise
            test  = set(rows) - set(DF.index.values)
            assert len(test) == 0, "Rows " +  ' '.join(test) + ' does/do not exist in DF'
            M = DF.loc[rows, :].values
            X = np.ravel(M)
            test_vec, error_vec = self.validator(X)
            DFT.loc[rows, :] = test_vec.reshape(M.shape)
            E.loc[rows, :] = error_vec.reshape(M.shape)
            X[~test_vec] = None
            DFnew.loc[rows, :] = X.reshape(M.shape)

        elif (len(cols) != 0) and (len(rows)!= 0):  # both rowwise and columnwise
            test  = set(cols) - set(DF.columns.values)
            assert len(test) == 0, "Cols " +  ' '.join(test) + ' does/do not exist in DF'
            test  = set(rows) - set(DF.index.values)
            assert len(test) == 0, "Rows " +  ' '.join(test) + ' does/do not exist in DF'
            M = DF.loc[rows, cols].values
            X = np.ravel(M)
            test_vec, error_vec = self.validator(X)
            DFT.loc[rows, cols] = test_vec.reshape(M.shape)
            E.loc[rows, cols] = error_vec.reshape(M.shape)
            X[~test_vec] = None
            DFnew.loc[rows, cols] = X.reshape(M.shape)
        else: #Nothing to test, empty validator
            pass
        return DFnew, DFT, E

###############################################################################################################################


def update_validator_bookkeeping(my_filter, DFT):
    """
    Update which columns/rows/elements were even seen by validators when validation is composed.
        my_fiter: the book keeper, 1 if the element was seen by a vaidator, 0 otherwise
        DFT: True/False/None matrix created by the validator class, a fundamental DF
    Returns:
        updated my_filter
    """
    my_filter = pd.DataFrame(np.select([(DFT.values.astype(str) != 'nan') , (my_filter.values.astype(str) != 'nan') ],\
                        [DFT.values, my_filter.values], default = None), index = my_filter.index, columns = my_filter.columns)
    #fully vectorized selector, for column you could have also done ...
    #my_filter = DPT.combine(my_filter, lambda x1, x2: 1 if (not np.all(pd.isna(x1)) or (not np.all(pd.isna(x2)))) else None)
    my_filter[~pd.isna(my_filter)] = True #I have trested all these elements
    return my_filter

###############################################################################################################################
###############################################################################################################################
###############################################################################################################################



class CompositeTransformTypeC:
    """
    COLUMNWISE composite operations, with AND or OR logic

    Class/Functor for most common operation after table extraction: assigning "meaning" (semantic type) to the
    columns, and validating all those semantic-typed columns. This class makes it easy to define a whole
    bunch of columns to have the same semantic type and compose columnar validation operations
    """
    def __init__(self, stype_config):
        """
        Args: A dict (read for json typically) of semantic types
        {
          <Name>: { "description" : <some-free-text-description-of-the-semantic-type",
          "defaultSearchFx": [(token, searchfx, extraParams, errorcode, weight), ..],
          "valueValidators": [(validatorFx, params, errorcode, weight), ...],
          "concepts":[<text describing concept>,...]
        }
        composition_logic for the list of validators for every sementic type is always OR, otherwise why not create a single validator??? 
        """
        self.operations = defaultdict(list)
        for s, v in stype_config.items():
            loc_value_vals = v['valueValidators']
            for entry in loc_value_vals:
                V = ValueValidator(entry[0], params = entry[1], error_code = entry[2])
                self.operations[s].append(V)

    def __call__(self, DF, stype_mapper_dict):
        """
        Args:
            DF:  Is just a DF with well-defined clean column headers, no duplicate columns etc. or multi-index.I don't do any checks on DF, be nice.
            stype_mapper_dict: a dict of lists, with list entries being DF column headers, keyed by semantic type assigned to these
                                column header names
        Returns: (fundamental dataframes)
            DFnew
            DFT
            Error

        """
        DFnew = DF.copy()
        error = pd.DataFrame().reindex_like(DFnew).fillna('')
        DFT = pd.DataFrame().reindex_like(DFnew).fillna(False) #OR composition logic, initiate to False 
        my_filter = pd.DataFrame().reindex_like(DFnew) #This keeps track of which columns are tested
        for sname, opts in self.operations.items():
            if sname in stype_mapper_dict:
                mapped_cols = np.unique(stype_mapper_dict[sname])
                if len(set(mapped_cols) - set(DFnew.columns)): 
                    raise ValueError('Column not found in Dataframe passed')     
                for loc_opt in opts: # all operations in opts list for every validator 
                    _, DFT1, error1 = loc_opt(DFnew, [], mapped_cols.tolist())
                    my_filter = update_validator_bookkeeping(my_filter, DFT1)
                    DFT =  np.logical_or(DFT, DFT1.fillna(False)) #This defines the composition
                    #error = error1.fillna('').combine(error, lambda x1, x2: x1 + ' ' + x2)
        DFT[my_filter.isna()] = None #Columns that hasn't been touched should be reverted to None
        error[my_filter.isna()] = None
        DFnew = DFnew[DFT.fillna(True)] #pass along untouched columns, but filter on validated ones, so fillna(True)
        return DFnew, DFT, error

###############################################################################################################################
###############################################################################################################################
###############################################################################################################################


class SType:
    """
    Instatitiate semantic type to type-check description/name/column header and vector of entires
    Implements OR logic on list of description match function, and values validator functions
    """
    def __init__(self, type_json):
        """
        {
        <Name>: { "description" : <some-free-text-description-of-the-semantic-type",
          "defaultSearchFx": [(token, searchfx, extraParams, errorcode, weight), ..],
          "valueValidators": [(validatorFx, params, errorcode, weight), ...],
          "concepts":[<text describing concept>,...]
        }
        """
        assert(len(type_json) == 1), "This just instatiates one semantic type, pass a dictionary keyed by semantic type name"
        self.name = type_json.keys()[0]
        entry = type_json.values()[0]
        if 'searchFx' in  entry:
            self.search_func_defs = entry['searchFx']
        else:
            self.search_func_defs = entry['defaultSearchFx']
        self.value_validator_func_defs = entry['valueValidators']
        self.Searchers = []
        self.searcher_weights = []
        self.Checkers = []
        self.checker_weights = []
        for i, K in enumerate(self.search_func_defs):
            setattr(self, 'Searcher' + str(i), search.Search(K[0], K[1], params = K[2]))
            self.Searchers.append(getattr(self,'Searcher' + str(i)))
            self.searcher_weights.append(K[4])
        for i, K in enumerate(self.value_validator_func_defs):
            setattr(self, 'Checker' + str(i), Checker(K[0], params = K[1], error_code = K[2]))
            self.Checkers.append(getattr(self,'Checker' + str(i)))
            self.checker_weights.append(K[3])

    def __call__(self, name, values):
        """
        Args:
            name of column/items
            values in column/vector of items
        Returns:
            searcher error (OR logic)
            vector of checker error (OR logic)
            error_tracer: composition of all errors for entries-vector validation
        """
        error_tracer = ['']*len(values)
        errors_searcher = []
        errors_checker  = []
        for opt in self.Searchers:
            errors_searcher.append(opt(name))
        for opt in self.Checkers:
            ans, temp = opt(values)
            errors_checker.append(ans)
            error_tracer = [a+' ' + b for a,b in zip(temp, error_tracer)]

        searcher_weightsum = np.asarray(errors_searcher).dot(np.asarray(self.searcher_weights))
        checker_weightsum = np.asarray(errors_checker).T.dot(np.asarray(self.checker_weights))
        detailed_results = {} 
        detailed_results['errors_searcher_ordered'] = errors_searcher
        detailed_results['errors_checker_ordered'] = errors_checker  
        detailed_results['searcher_weighted'] = searcher_weightsum
        detailed_results['checker_weighted'] = checker_weightsum
        return np.any(errors_searcher), np.any(errors_checker, axis = 0), map(str.strip, error_tracer), detailed_results 

###############################################################################################################################
###############################################################################################################################
###############################################################################################################################

def _early_succeed_search(search_tuples, header_cell):
    """
    early fail for ordereded list of searching for a header
    """
    for i, K in enumerate(search_tuples):
        searcher = search.Search(K[0], K[1], params = K[2])
        test = searcher(header_cell)
        if test:
            break
    return test, i
##########################################################################################################


def _loop_over_tokens(search_tuples, header_cells):
    """
    loop over search tupes for one token
    """
    mapped_to = []
    mapped_to_index = []
    for i, h in enumerate(header_cells):
        test, _ = _early_succeed_search(search_tuples, h)
        if test: #required token must succeed!
            mapped_to.append(h)
            mapped_to_index.append(i)
    return mapped_to, mapped_to_index

##########################################################################################################


def _early_fail_required_tokens_loop(required_token_dict, header_cells):
    """
    loop over function
    """
    my_mapper = {}
    my_mapper_index = {}
    for k, v in required_token_dict.items():
        mapped_to, mapped_to_index = _loop_over_tokens(v, header_cells)
        if len(mapped_to) == 0: #failed to find required token
            return dict(), dict()
        else:
            my_mapper[k] = mapped_to
            my_mapper_index[k] = mapped_to_index
    return my_mapper, my_mapper_index

##########################################################################################################


def _optional_tokens_loop(optional_token_dict, header_cells):
    """
    loop over function
    """
    my_mapper = {}
    my_mapper_index = {}
    for k, v in optional_token_dict.items():
        mapped_to, mapped_to_index = _loop_over_tokens(v, header_cells)
        if len(mapped_to) > 0:
            my_mapper[k] = mapped_to
            my_mapper_index[k] = mapped_to_index
    return my_mapper, my_mapper_index


##########################################################################################################

class TableKraken:
    """
    Table extractor from a specific sheet in excel, passed in as DataFrame
    """
    def __init__(self, stype_configs, stype_status, params = None):
        """

        Args:
            DF: dataframe: if an excel sheet <sheet name> please use 
            >> sheet = pd.ExcelFile(<file path>)
            >> M.parse(<sheet name>, index = None, header = None) 
            TableKraken treats the whole excel sheet as a table to look for headers, so avoid auto-header and indexing in Pandas. 
            stype_configs:  standard stype config dicts, keyed by stype, must have a list of search functions, under key 'searchFx'
                          see docs. on config file for details, can pass the whole list of all configs
            stype_status: class will only inspect these stypes, a dict with required ('R') or optional ('O') status
            params: {'drop_all_numeric': True, #drop columns that are all numeric in trying to match, if False will be slower/ambiguous
                        because this allows for headers names to be numbers
                        'header_thickness': 1}, Can handle search over multiple cells as header (for exmaple, 2 means headers with 2 levels)
        Returns:
            extracted_tables: extracted tables
            kraken_output['header_mapper_dict'] = dict of headers mapped to tokens
            kraken_output['header_mapper_index_dict'] = dict of header mapped index of column (of original DF)
            kraken_output['header_row_index_begin'] = list of header row index of the column header for table extracted
            kraken_output['header_row_index_end'] = list of extracted-table-end row index
            kraken_output['table_column_indices'] = column indices dict for each table
            kraken_output['extracted_tables'] = extracted tables


        """
        if params is None:
            self.params = dict()
        else:
            self.params = params.copy()
        self.params.setdefault('drop_all_numeric', True) #drop columns that are all numeric in trying to match, if False will be slower
        self.params.setdefault('header_thickness', 1)
        self.required_tokens = {}
        self.optional_tokens = {}
        assert len(set(stype_status.keys()) - set(stype_configs.keys())) == 0, "some stypes are undefined in configs"
        for k, v in stype_status.items():
            assert v in ['O', 'R'], 'Only two options for stype_required values: O = optional, R = required'
            if 'searchFx' in stype_configs[k]:
                temp = stype_configs[k]['searchFx']
            else:
                temp= stype_configs[k]['defaultSearchFx']
            if v == 'R':
                self.required_tokens[k] = temp
            else:
                self.optional_tokens[k] = temp
        if len(self.required_tokens) == 0:
            warnings.warn('Without required tokens, your search is likely to be very slow and very ambiguous')

        self.stype_status = stype_status
        self.stype_configs = stype_configs

    def __call__(self, DFin):
        """
        Args: excel file as dataframe, make sure you read excel with no header index and column index
        """
        DF = DFin.applymap(clean_text) # Get rid  of ascii, a few pesky problems with Ascii characters
        thickhead = self.params['header_thickness']
        #--------- My accumulators, list of dicts -------
        header_mapper_dicts = []
        header_mapper_index_dicts = []
        suspect_header_row_index = [] #stored the index of the header row detected
        suspect_table_end = []
        col_indices = []
        extracted_tables = []
        #-------------------------------------------------
        for i, (_, v) in enumerate(DF.iloc[:-thickhead,:].iterrows()): #i, v is running over rows
            # For seafety initisted to empty, this should not be necessary
            #---------------------------------------------
            header_mapped = {}
            header_mapped_index = {}
            header_mapped_optional = {}
            header_mapped_index_optional = {}
            header_mapped_index_original_frame = {}
            #----------------------------------------------
            if thickhead == 1:
                row = v.dropna()
                global_row_index = np.arange(DF.shape[1])
                global_row_index = global_row_index[~v.isna()] #index of surviicing columns
                L = len(row)
            else:
                row = DF.iloc[i:(i+thickhead), :]
                global_row_index = np.arange(DF.shape[1])
                global_row_index = global_row_index[~np.all(row.isna(), axis = 0)] #index of surviving columns
                row = row.dropna(axis = 1, how = 'all') #This is critical, cannot ffill (see below) before dropping NaNs
                L = row.shape[1]
            if L>0:
                if thickhead == 1:
                    if self.params['drop_all_numeric']:
                        test = pd.to_numeric(row, errors = 'coerce').isna() # How many of these columns in the row are
                        #non-numeric (string)values
                        non_num_check = np.sum(test) > len(self.required_tokens)
                        possible_header_cells = row[test].values.astype(str)
                        global_row_index = global_row_index[test]
                    else:
                        non_num_check = True
                        possible_header_cells = row.values.astype(str)
                else:
                    if self.params['drop_all_numeric']:
                        test = np.all(row.apply(pd.to_numeric, errors = 'coerce').isna(), axis = 0)
                        # How many of these columns in the row are non-numeric (string) values
                        non_num_check = np.sum(test) > len(self.required_tokens)
                        possible_header_cells = row.loc[:, test].fillna(method = 'ffill', axis = 1).fillna('').values.astype(str)
                        # forwardfill NaNs, this is because excel DOES NOT store the info of merged cells in both cells
                        # I fill the remaining NaNs with empty strings, think of a multi-index header, this the best you can do!
                        global_row_index = global_row_index[test]

                    else:
                        non_num_check = True
                        possible_header_cells = row.fillna(method = 'ffill', axis = 1).fillna('').values.astype(str)
                        # forwardfill nans, this is because excel DOES NOT store the info of merged cells in both cells
                        # I fill the remaining NaNs with empty strings, think of a multi-index header, this the best you can do!
                    possible_header_cells = [' '.join(x) for x in possible_header_cells.T]
                if non_num_check:  #Then this column is even worth considering
                    if len(self.required_tokens):
                        header_mapped, header_mapped_index = _early_fail_required_tokens_loop(self.required_tokens, possible_header_cells)
                        if len(header_mapped):
                            if len(self.optional_tokens):
                                header_mapped_optional, header_mapped_index_optional = \
                                _optional_tokens_loop(self.optional_tokens, possible_header_cells)
                                header_mapped.update(header_mapped_optional)
                                header_mapped_index.update(header_mapped_index_optional)
                    else:
                        header_mapped, header_mapped_index = _optional_tokens_loop(self.optional_tokens, possible_header_cells)

                    header_mapped_index_original_frame = {tk:global_row_index[np.asarray(tv)] for tk, tv in header_mapped_index.items()}
                    if len(header_mapped):
                        suspect_header_row_index.append(i)
                        header_mapper_dicts.append(header_mapped)
                        header_mapper_index_dicts.append(header_mapped_index_original_frame)

        if len(header_mapper_dicts):
            # I need to make sure now that the "matrices" defined by the col index and the row
            # index are non overlapping.
            col_indices = []
            for i, s in enumerate(suspect_header_row_index):
                col_index =  np.sort(np.unique([item for k, l in header_mapper_index_dicts[i].items() for item in l])) #sorted in
                #order of appearance in the the table columns
                col_indices.append(col_index)

            # I need to find nonoverlapping tables, recall supsect_header_row_index is ordered by the row index of DF
            suspect_table_end = np.ones_like(suspect_header_row_index)*len(DF) #Default table length is until end of sheet
            for i, s in enumerate(suspect_header_row_index):
                loc_cols = col_indices[i]
                for j, test_cols in enumerate(col_indices):
                    if (i!=j) & (suspect_header_row_index[j] > suspect_header_row_index[i]) & \
                    (len(set(test_cols).intersection(set(loc_cols))) > 0):
                    # i.e., a different table under consideration
                        suspect_table_end[i] = np.min((suspect_header_row_index[j],suspect_table_end[i])) #end the table before overlap
            # now you are ready to construct the tables
            for i, s in enumerate(suspect_header_row_index):
                #cols = np.unique([item for k, l in header_mapper_dicts[i].items() for item in l])
                col_index =  np.sort(np.unique([item for k, l in header_mapper_index_dicts[i].items() for item in l])) #sorted in order of \
                if thickhead == 1:
                    loc_df = DF.iloc[(suspect_header_row_index[i]+1):suspect_table_end[i], col_index]
                    loc_df.columns = DF.iloc[suspect_header_row_index[i], col_index].values
                else:
                    loc_df = DF.iloc[(suspect_header_row_index[i]+thickhead):suspect_table_end[i], col_index]
                    col_rows = DF.iloc[suspect_header_row_index[i]:(suspect_header_row_index[i]+thickhead), col_index].fillna\
                    (method = 'ffill', axis = 1).fillna('').values.astype(str)
                    loc_df.columns = pd.MultiIndex.from_arrays(col_rows)
                loc_df = loc_df.dropna(axis = 0, how = 'all')
                extracted_tables.append(loc_df)

        kraken_output_data = {}
        kraken_output_data['header_mapper_dict'] = header_mapper_dicts
        kraken_output_data['header_mapper_index_dict'] = header_mapper_index_dicts
        kraken_output_data['header_row_index_begin'] = suspect_header_row_index
        kraken_output_data['header_row_index_end'] = suspect_table_end
        kraken_output_data['table_column_indices'] = col_indices
        kraken_output_data['extracted_tables'] = extracted_tables
        return extracted_tables, kraken_output_data



#################################################################################################################################

def strong_type_table(DF, stype_configs, rename = True):
    """
    An opinionated function to impose strong semantic type on tables---i.e. table has one and only one column of any specific
    semantic type; used to clean up tables
    Args:
        stype_config: Typical Stype config for all of angira

    Returns:
        table: Cleaned table, with column headers, with columns if passed semantic type test, Default return is None 
        dropped_headers_dict: which headers, keyed by stype name, did I drop for each stype?
        header_map: which column was renamed/mapepd to by imposing strong stype?
        rename: If True, rename header by mapped semantic type
    """

    table = DF.applymap(clean_text) # Get rid  of ascii, a few pesky problems with Ascii characters
    # Take care of duplicate column names by adding fake space, should not affect any reasonable column search operations
    loc_cols = table.columns
    duplicated_cols = loc_cols[loc_cols.duplicated()] 
    table.columns = [c + ''.join([' ']*k) if c in duplicated_cols else c for k, c in enumerate(loc_cols)] 
    and_indicator = np.ones(len(table)).astype(bool)
    error_collector = ['']*len(table)
    drop_headers_dict = {}
    header_map_dicts = []
    new_table = None
    for t, d in stype_configs.items():
        score = 0
        final_test_data = None
        final_error = None
        drop_headers = []
        header_map = {}
        type_me = SType({t: d}) #This will check the Stype of each column!
        for v in table.columns:
            entries = table.loc[:, v].values
            test_header, test_data, test_error, _ = type_me(v, entries)
            if np.any(test_data) & test_header:
                new_score = np.sum(test_data)
                if new_score > score:
                    score = new_score
                    final_test_data = np.copy(test_data)
                    final_error = np.copy(test_error)
                    header_map[v] = t
                else:
                    drop_headers.append(v)
            elif test_header:
                drop_headers.append(v)

        if len(header_map):
            if len(drop_headers):
                table.drop(drop_headers, axis = 1, inplace = True)
            if rename: 
                table.rename(header_map, inplace = True, axis = 1)
            and_indicator = np.logical_and(and_indicator, final_test_data)
            error_collector = [a + ' ' + b for a, b in zip(final_error, error_collector)]
            drop_headers_dict[t] = drop_headers
            header_map_dicts.append(header_map)
    if len(header_map_dicts) > 0:
        new_table = table.loc[and_indicator]
    return new_table, drop_headers_dict, header_map_dicts, np.asarray(map(str.strip, error_collector))

#################################################################################################################################

def weak_type_table(DF, stype_configs):
    """
    Impose weak semantic type a table from a list of semantic types defined
    Args:
        stype_config: Typical Stype config for all of angira
    Returns:
        table: muti-indexed with semantic type inferred, None if nothign could be inferred
        header_map: map of header to semantic type, only for unique mappers
        multi_index: Used to multi_index table
    """

    table = DF.applymap(clean_text) # Get rid  of ascii, a few pesky problems with Ascii characters
    # Take care of duplicate column names by adding fake space, should not affect any reasonable column search operations
    loc_cols = table.columns
    duplicated_cols = loc_cols[loc_cols.duplicated()] 
    table.columns = [c + ''.join([' ']*k) if c in duplicated_cols else c for k, c in enumerate(loc_cols)] 
    and_indicator = np.ones(len(table)).astype(bool)
    error_collector = ['']*len(table)
    header_maps = defaultdict(list)
    new_table = None
    for t, d in stype_configs.items():
        header_map = {}
        type_me = SType({t: d}) #This will check the Stype of each column!
        for v in table.columns:
            entries = table.loc[:, v].values
            test_header, test_data, test_error, _ = type_me(v, entries)
            if np.any(test_data) & test_header:
                score = np.sum(test_data)
                if score > 0:
                    header_maps[v].append([t, score/float(len(test_data))])
    multi_index = []
    if len(header_maps) > 0:
        for v in table.columns:
            if v in header_maps and (len(header_maps[v]) == 1): #found and unique
                multi_index.append((header_maps[v][0][0], v))
            else:
                multi_index.append(('', v))
        new_table = table.copy()
        new_table.columns = pd.MultiIndex.from_tuples(multi_index)
    return new_table, header_maps, multi_index

#################################################################################################################################

def map_header(header, stype_configs):
    """
    Use standard config file to check which headers map to which SemanticTypes 
    Args: 
        list of header 
        stype_configs: stdard angira confg json, must have SearchFx or DefaultSearchFx entry for every semantic
        type 
    Returns: 
        all_sementic_type_maps: (dict of list) semantic type to list of headers mapping
        all_colukn_maps: (dict of list) all header to semantic type mapping 
    """
    
    all_semantic_maps = defaultdict(list)
    all_column_maps = defaultdict(list) 
    for t, entry in stype_configs.items():
        if 'searchFx' in  entry:
            searcher_tuples = entry['searchFx']
        else:
            searcher_tuples = entry['defaultSearchFx']
        for v in header:
            tests = []
            for K in searcher_tuples: 
                func = search.Search(K[0], K[1], params = K[2])
                tests.append(func(v))
            if np.any(tests):
                all_semantic_maps[t].append(v) 
                all_column_maps[v].append(t)
    return dict(all_semantic_maps), dict(all_column_maps) 
    
#################################################################################################################################


class CompositeTransformNumTypeD:
    """
    A Compostite Transform functor to reduce the number of mapped semntic type column to single one
    by passing an operation of aggregation over those types 
    aggregation operation must be avaiable in numpy masked array!
    
    THIS IS ONLY FOR NUMERIC Semantic types, there is not way for the class to check this 
    but will exit ungracefully with error in trying to convert to float 
    
    """
    def __init__(self, stype_config, aggregator = 'max'): 
        """
        
        """
        self.operations = defaultdict(list)
        self.aggregator = aggregator 
        for s, v in stype_config.items():
            loc_value_vals = v['valueValidators']
            for entry in loc_value_vals:
                V = Checker(entry[0], params = entry[1], error_code = entry[2])
                self.operations[s].append(V)

    def __call__(self, DF, stype_mapper_dict):
        """
        Args:
            DF:  Is just a DF with well-defined clean column headers, no duplicate columns etc. or multi-index. I don't do any checks on DF, be nice.
            stype_mapper_dict: a dict of lists, with list entries being DF column headers, keyed by semantic type assigned to these
                                column header names
        Returns: 
            DFnew: only the data frame type checked and renamed 
        """
        DFnew = pd.DataFrame()  
        for sname, opts in self.operations.items():
            if sname in stype_mapper_dict:
                mapped_cols = np.unique(stype_mapper_dict[sname])
                if len(set(mapped_cols) - set(DF.columns)): 
                    raise ValueError('Column not found in Dataframe passed')   
                test_mat = [] 
                for loc_opt in opts: # all operations in opts list for every validator 
                    M = DF.loc[:, mapped_cols].values 
                    X = np.ravel(M)
                    test_vec, _ = loc_opt(X) #vectorized over the whole sub-matrix M which is the matrix of all headers of a certian semantic type! 
                    test_mat.append(test_vec)
                test_reduced = np.any(test_mat, axis = 0) #COMPOSITION LOGIC IS OR
                test_reshaped = test_reduced.reshape(M.shape)
                K = np.empty(M.shape)
                K[test_reshaped] = M[test_reshaped].astype(float)
                values = ma.masked_array(K, ~test_reshaped, fill_value = np.nan)
                masked_max =getattr(ma, self.aggregator)(values, axis = 1) #get the right fucntion for masked array numpy 
                DFnew[sname] = pd.Series(masked_max).values 
        return DFnew
#################################################################################################################################


def aggregateOverColumns(DF, col_names, target_col_name, aggregator_func_name = 'min'): 
    """
    Aggregate over user defined columns handling NaNs/missing values 
    Args:
        DF: 
        col_names: to aggregate over 
        target_col_name: aggregated columns will be delated and return col is named as this 
        aggregator_func_name: aggregator function in masked array (numpy) 
    
    """
    agg_DF = DF.copy() 
    loc_g = DF[col_names].values 
    test = ~pd.isna(loc_g)
    K = np.empty(loc_g.shape)
    K[test] = loc_g[test].astype(float)
    val = ma.masked_array(K, mask = ~test, fill_value = np.nan)
    X = getattr(ma, aggregator_func_name)(val, axis = 1)
    agg_DF[target_col_name] = pd.Series(X).values
    agg_DF.drop(col_names, axis = 1, inplace = True)
    return agg_DF 
    
    
##################################################################################################################################

def columnDownFiller(DF, column): 
    """
    Very immoral function that does one reprehensible job: given missing data in a column this will fill the column 
    with the last legal value seen, where "last" is with respect to row index. Treats empty string and null as "missing value". 
    Args: 
        DF: Data frame 
        column: column to shamefully fill out 
    Returns: 
        Copy of the assaulted DF 
    """ 
    DF_copy = DF.copy()
    DF_copy.replace(r'^\s*$', np.NaN, regex=True, inplace = True) #this is to make sure that empty spaces are interpretted as NaN
    data = DF_copy[column] 
    DF_copy[column] = data.fillna(method = 'ffill')
    return DF_copy    
