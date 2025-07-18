import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/nicolai/work/dnv/SensorFusion/revolt_state_estimator/install/revolt_state_estimator'
