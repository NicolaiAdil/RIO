from setuptools import setup

setup(
    name="autoseapy_tracking",
    version="0.1",
    description="Tracking packages from the Autosea project",
    author="Autosea",
    author_email="andreas.flaten@ntnu.no",
    packages=["autoseapy_tracking"],
    install_requires=[
        "scipy",
        "numpy",
        "pandas",
        "pyyaml",
    ],
    include_package_data=True,
)
