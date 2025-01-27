import os
import xml.etree.ElementTree as ET

from sudonim import getLogger

log = getLogger()

def xmlToJson(tree, nan=[], blacklist=[], rename={}):
    """ 
    Convert XML to JSON and filter the keys.
    (this gets used to parse nvidia-smi output)
    """
    response = {}

    if not nan:
        nan = ['N/A', 'Unknown Error', 'None', None]

    if not blacklist:
        blacklist = [
            'gpu_reset_status', 'ibmnpu', 'temperature',
            'gpu_power_readings', 'module_power_readings'
        ]

    if not rename:
        rename = {
            'product_name': 'name',
            'product_architecture': 'arch',
        }

    def is_nan(text):
        text = text.lower()
        for n in nan:
            if n:
                if n.lower() in text:
                    return True
            else:
                if not text:
                    return True
        return False
    
    if isinstance(tree, str):
        tree = ET.fromstring(tree)

    for child in tree:
        if child.tag in blacklist:
            continue
        if child.tag in rename:
            child.tag = rename[child.tag]

        if len(list(child)) > 0:
            children = xmlToJson(child)
            if children:
                if child.tag in response:
                    if isinstance(response[child.tag], list):
                        response[child.tag].append(children)
                    else:
                        response[child.tag] = [response[child.tag], children]
                else:      
                    response[child.tag] = children
        else:
            text = child.text.strip()
            if not is_nan(text):
                response[child.tag] = text

    return response

class NamedDict(dict):
    """
    A dict where keys are available as named attributes:
    
      https://stackoverflow.com/a/14620633
      
    So you can do things like:
    
      x = NamedDict(a=1, b=2, c=3)
      x.d = x.c - x['b']
      x['e'] = 'abc'
      
    This is using the __getattr__ / __setattr__ implementation
    (as opposed to the more concise original commented out below)
    because of memory leaks encountered without it:
    
      https://bugs.python.org/issue1469629
    """
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, value):
        self.__dict__ = value

'''    
class NamedDict(dict):
    def __init__(self, *args, **kwargs):
        super(NamedDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
'''