# User Datagram Protocol (UDP) Video Stream

##Table of content
- [Overview](#Overview)
- [Interactions](#Interactions)
- [Purpose](#Purpose)
- [Sources](#Sources)

## Overview 
Creates the output stream to send to the RMC Station through for the video stream through a UDP connection.

A library called ”Practical C++ Sockets” is used to create communication sockets to transmit and receive data.

## Interactions
Subscribes to necassary data to send to RMC station. Sends data to RMC station thorugh the UDP socket. See thesis written by Albert Havnegjerde for an extensive explenation.

## Purpose  
Enable transmission of data between the RMC station and ReVolt for the video stream.

## Sources 
* "Remote Control and Path Following for the ReVolt Model Ship", Albert Havnegjerde. June 2018. TODO add link