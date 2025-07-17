from setuptools import setup

package_name = 'hardware_launch'

setup(
    name=package_name,
    version='0.0.0',
    packages=[],
    py_modules=[],
    data_files=[
        # register this package with ament_index
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        # install your package.xml
        (f'share/{package_name}', ['package.xml']),
        # install your launch files
        (f'share/{package_name}/launch', ['launch/hardware.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='revoltrevolt',
    maintainer_email='NicolaiAdil.Aatif@gmail.com',
    description='Main launch files for the hardware system',
    license='MIT',
    entry_points={
        # No entry points needed for this package
    },
)
