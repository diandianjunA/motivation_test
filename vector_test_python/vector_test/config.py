import configparser
from typing import Dict


def read_config(filename: str) -> Dict[str, str]:
    """
    Read configuration from file.
    
    Args:
        filename: path to the configuration file
        
    Returns:
        Dictionary mapping config keys to values
    """
    config = configparser.ConfigParser()
    config.read(filename)
    
    result = {}
    for section in config.sections():
        for key, value in config.items(section):
            result[key] = value
            
    return result
