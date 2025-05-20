"""
This is the 'code' implementation for the scenario of Bob
"""

# Import packages
import twain.moct as moct  # FIXME: The package name 'twain' is not available in PyPI. Maybe 'twainwfc' as an alternative?
import numpy as np
import pandas as pd
import getpass

# Establish a connection to the TWAIN environment
username = 'BobDoe'
password = hash(getpass.getpass(f"Enter password for user '{username}': "))

# Launch the GUI
moct.open(connect={'username': username, 'password': password}, dir_local='./data/example2/')