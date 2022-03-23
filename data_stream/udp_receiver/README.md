# UDP receiver

##Table of content
- [Overview](#Overview)
- [Interactions](#Interactions)
- [Purpose](#Purpose)
- [Sources](#Sources)

## Overview 
A User Datagram Protocol (UDP) receiver for obstacle-data sent from the ReVolt Vessel Simulator.

## Interactions
Recives data from simulator. Decodes and sends obstacle data to /ais_dynamic topic. See thesis by Tonje Midjås for an extensive explenation.

## Purpose  
Enable transmission of data between the Simulator and ReVolt.

## Sources 
* Tonje's thesis: "SBMPC Collision Avoidance for the ReVolt Model-Scale Ship" - Tonje Midjås
