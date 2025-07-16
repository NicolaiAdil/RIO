from setuptools import setup
import glob

package_name = 'revolt_state_estimator'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    include_package_data=True,
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob.glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob.glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='martynas',
    maintainer_email='smilingis.martynas@gmail.com',
    description='EKF‐based ship state estimator ROS2 node',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'revolt_ekf_node = revolt_state_estimator.revolt_ekf_node:main',
        ],
    },
)

