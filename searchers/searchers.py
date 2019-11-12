"""
searchers: All fucntions to search for tokens/names in fil headers etc. String search operaitons galore 
""" 
__author__ = "Swagatam M"
__copyright__ = "@Swagatam mukhopadhyay"
__credits__ = []
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Swagatam Mukhopadhyay"
__email__ = "smukhopadhyay@ionisph.com"
__status__ = "Dev"

import re
import regex
import six
from fuzzywuzzy import fuzz 


#################################################################

def my_wrapper(pattern, x, module, func, params= None): 
    """
    wrapper for fuzzy from fuzzywuzzy 
    """
    if params is None: 
        _params = {}
    else:
        _params = params.copy() 
            
    if module == 'fuzz': 
        if 'fuzzy_thresh' not in _params: 
            fuzzy_thresh = 90 
        else: 
            fuzzy_thresh = _params['fuzzy_thresh']
            _params.pop('fuzzy_thresh') #make params digestible by fuzzy 

        func = getattr(fuzz, func)(x, pattern, **_params)
        ans = True if func > fuzzy_thresh else False 
    elif module == 're':
        pattern = re.compile(pattern)
        func = getattr(re, func)(pattern, x, **_params)
        ans = True if func is not None else False 
    elif module == 'regex':
        pattern = regex.compile(pattern)
        func = getattr(regex, func)(pattern, x, **_params)
        ans = True if func is not None else False   
    else: 
        raise NotImplementedError("Only modules are re regex and fuzz from fuzzywuzzy")
    return ans


#############################################################################################



class Search: 
    """
    A general class for searcher functions for strings, matching, fuzzy matching, searching, etc.
    Knows how to handle re, regex and fuzzywuzzy by default
    initiated by the a string of the form re.match and it will parse the module (re) and the function (match) 
    or you can pass your own search function as a python function
    Trace of function must be (pattern, to_match_string, **params)
    Return of function should be True, False 
    """
    def __init__(self, token, func, params = None): 
        """
        Args: 
            token: I will compile this into a pattern if you give me re or regex, or use it as a token for fuzzywuzzy 
            if you are using re package, remember that if you want case=insensitive and ignore space write, \s*(?i)your pattern\s*
            func: exmaple, re.match or fuzz.ratio 
            params_dict: fuzzy score threshold to consider a match for fuzzy, and params I will simply pass on to the function you write 
            will be ignored for re and regex because they don't take any params
            Please be nice and try to use match, search, findall... avoid finditer (i.e., avoid generators)... I DON"T DO ANY CHECKS!!! 
        """
        if params is None: 
            _params = {}
        else:
            _params = params.copy() 
        if isinstance(func, six.string_types):
            module, fname = func.split('.')
            self.searcher = lambda x: my_wrapper(token, x, module, fname, params = _params)
        else: 
            self.searcher = lambda x: func(token, x, params = _params)

    def __call__(self, s): 
        assert isinstance(s, six.string_types), "expect string"
        ans = self.searcher(s)
        return ans
        
    
    
