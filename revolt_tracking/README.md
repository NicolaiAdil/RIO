# Revolt tracking

This is a package that contains implementations of various tracking algorithms from the Autosea project.
Since the Autosea project is as comprehensive as it is, this package was made to only contain the absolute bare
necessary components for a fully-operative tracking system.

If you wish to expand upon this package, check out the Autosea source code in its entirety under the `Archive` repository.
The branch `/project/2019/knut/sensor_fusion_camera-lidar` contains the work done by Knut Turøy before it was moved over to this package.
Chances are, you will find the Autosea source code in multiple other branches, but the aforementioned one is the one that contained the
most recent (working and tested!) updates prior to making this package.

***

## Notes
The autoseapy_tracking folder contains python scrips that are built as a python package, and included whereever needed. This means that any changes made to these files **will not be reflected in the rest of the system until the package is rebuilt!**

You may rebuild the package by either running the ´build_script.sh´, or by executing

    cd ~/revolt_ws/src/sensor_fusion/revolt_tracking
    python2.7 -m pip install --user .

