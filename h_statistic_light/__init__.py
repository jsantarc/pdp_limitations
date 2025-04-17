"""
h_statistic_light - A lightweight library for feature interaction analysis.
"""

__version__ = '0.1.0'

# Import main components
from .h_statistic import HStatisticCalculator
from .utils import compute_pdp_interaction, check_pdp_additivity, compute_pdp_ice
from .data_generation import generate_interaction_data, generate_linear_interaction_data
