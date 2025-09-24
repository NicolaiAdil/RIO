from setuptools import setup, find_packages
import glob

package_name = 'revolt_state_estimator'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(where='.'),  # picks up revolt_state_estimator and state_estimate_tuning
    include_package_data=True,
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob.glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob.glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='NicolaiAdil',
    maintainer_email='NicolaiAdil.Aatif@gmail.com',
    description='ES-EKF for ship state estimator in ROS2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'revolt_ekf_node = revolt_state_estimator.revolt_es_ekf_node:main',
            'revolt_ekf_debugger_node = state_estimate_tuning.revolt_es_ekf_debugger_node:main',
        ],
    },
)

