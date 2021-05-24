# AVD 2021 Final Project - Group 6
Final Autonomous Vehicle Driving Project.

## Introduction

### Directory tree

Please be sure to clone this repository under `PythonClient` directory:

```text
CarlaSimulator
│   ...
├───PythonClient
│   ├───carla
│   ...
│   ├───final_project
```

### Requirements

Please be sure to install the proper [requirements](requirements.txt):

```bash
pip install -r requirements.txt
```

## Server

Run the server with the following command:

* Windows:

```bash
..\..\CarlaUE4.exe /Game/Maps/Town01 -windowed -carla-server -benchmark -fps=<FPS>
```

* Linux:

```bash
../../CarlaUE4.sh /Game/Maps/Town01 -windowed -carla-server -benchmark -fps=<FPS>
```

**[IMPORTANT]** Remember to use the same `fps` paramater in both server and client.

## Client

Under development...
