"""
This script explains how the multi-objective control toolbox can interact with the TWAIN environment and returns the result to the users.
"""

# Import packages
import pandas as pd
import numpy as np
import twain.moct as moct

# Give your credentials
USERNAME, PASSWORD = 'test-user-001', 'test-password-001'

# Launch the GUI
moct.open()