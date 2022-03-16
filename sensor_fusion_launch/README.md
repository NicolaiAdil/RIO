# Launchfiles

The purpose of this package is to define which, how and with what parameters other packages of the system are to be launched. If you have just written a package, you would need to edit parts of this package to actually run it with the rest of the system. More details on how to do this can be found in the *Details about the launchfiles* section.

***

## Main launch commands
Everything described above is defined in the `revolt_launch` file. Since 
this file is located in a ROS compliant package, it can be reached from anywhere, using the following command:
    
    roslaunch revolt_launch revolt.launch 

to launch on the physical vessel, and


    roslaunch revolt_launch revolt.launch main:=simulator

to launch in the simulator.
***

## Launching optional nodes 

Not all nodes are necessary for the system to run or even desired to run every time we launch the system. To allow for more flexibility in choosing which nodes to run we therefore use arguments to launch optional features. The arguments are defined at the top of `revolt.launch` like this: 

    <arg name = "rqt" default="false"/>

which in this case means that the argument rqt has default value "false" (will not run unless specified). To run rqt we have to pass "true" to our launch command like this: 

    roslaunch revolt_launch revolt.launch rqt:=true

Optionally, you can set the default value to true temporarily if you mostly want it to run while working on your project.

For more information on how to add your own options and launchfiles, see the sections below. 

***

## Details about the launchfiles


In the following section we give a brief introduction on:
- how the launchfiles are structured.
- how to read them.
- how to create a new launchfile and add it to the program.
- how to interpret arguments differently to achieve the desired behavior. 

### Toplevel launch
The toplevel main launchfile is called `revolt.launch` and can be found in the source directory. Its purpose is to let the user decide which functionalities in the control-system that should be run based on the arguments passed. It does this by having predefined **'arguments'** at the top of the file and conditional **'includes'** that decide which additional launchfiles that should also run. 

### Arguments
The arguments that can be passed to the launchfile are written at the top of the file. An example of how this statement looks: 

    <arg name = "main" default="vessel"/>

This means that the argument with the name "main", has a default value of "vessel". The default value can be overriden through passing the argument 'main'. Example:

    roslaunch revolt_launch revolt.launch main:=simulator

which will run the nodes the program needs to run on the simulator instead of on the Revolt vessel.

### Running nodes directly
If your new package is not very complicated, chances are that you will not need an entirely separate launchfile for your node(s). To launch your node(s) directly, you may add    

    <node pkg="<your_pkg_name>" type="<file_to_run>" name="<name_in_ros>" output="<screen/log>"/>

directly to the `revolt.launch` file. Note that you should make sure to put it in the correct group, and add it as an optional launch if your package is not always required for operation.

### Including launchfiles
If you have written a new package that has it's own launchfile(s), you may wish to run these instead of defining your launch directly in the `revolt.launch` file. To do this, you will need to "include" your launchfile. An example from the `revolt.launch` file:

    <include file="$(find revolt_launch)/launch/tracking.launch" pass_all_args="true"/>

the `file` argument should point to another launchfile, either via a relative path from the parent launchfile, or as shown in the example above, relative to a given package. 

The optional argument `pass_all_args` can be used to pass arguments defined in the parent launchfile, into the included launchfile. 

If your launchfile does not fit within one single package, it can be put in the `/launch` directory of this package. Otherwise, an attempt should be made to keep the launchfiles themselves contained to their respective packages, and include them as outlined above. Launchfiles used for testing should **always** be placed in separate /launch folders in each individual package. This is to avoid clutter in this package.

### Conditional statements
By evaluating the arguments in conditional statements, we can then run the desired launchfiles. An example from the revolt.launch file:

    <group if $(eval main == "vessel")> 
        <!-- Vessel-specific nodes can be launched here -->
    </group>

this simply runs the revolt-main.launch file when 'main' is set to it's default value. Through this method we can run any combination of launch files! We can even make custom groups of files that we want to run together and create an argument for running that specific group. This is useful if you are working on your own project and only need parts of the functionality Revolt has to offer. 

*If you want to learn more about writing launchfiles and their possibilities, check out 'additional sources' at the bottom of the page.*

## Common Errors

- Note that if a node is already running and you try to run a node of the same name, roscore will kill all nodes of the same name currently running. If there are similar nodes in two launchfiles, and you try to launch both files, this problem may arise. A solution could be to comment out one of the nodes in either file or even use it as an opportunity to organize it better. Many of the files are created by different students trying to implement different things, so it's probably not uncommon that this is the case. Source: https://stackoverflow.com/questions/50954601/launch-node-if-not-already-running


***

### Additional sources

http://wiki.ros.org/roslaunch/Tutorials/Roslaunch%20tips%20for%20larger%20projects

https://sir.upc.edu/projects/rostutorials/4-launch_tutorial/index.html

https://www.chenshiyu.top/blog/2020/09/29/Basic-Grammar-of-ROS-Launch/

***

*Author: Eldar Sandanger*

*Email: <Eldar.sandanger@gmail.com>*
