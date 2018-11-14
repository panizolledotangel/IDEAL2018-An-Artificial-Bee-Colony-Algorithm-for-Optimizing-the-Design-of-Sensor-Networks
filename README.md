# IDEAL2018-An-Artificial-Bee-Colony-Algorithm-for-Optimizing-the-Design-of-Sensor-Networks
Code and data for the article "An Artificial Bee Colony Algorithm for Optimizing the Design of Sensor Networks"<br>     
DOI: 10.1007/978-3-030-03496-2_35

## Requisites
Having installed docker (https://www.docker.com/) and docker-compose (https://docs.docker.com/compose/install/).

## Running
1. clone the repository.
2. go to folder with the repository and run "docker-compose up".
3. In the terminal find the line "Copy/paste this URL into your browser when you connect for the first time,to login with a token:"
copy the url and paste it un your broswer.
4. open host_data->notebooks->*.ipynb" to run the different experiments.

## Structure
Inside jupyter home is the proyect structure: <br>
├── historic_evolution<br>
├── Hive<br>
├── notebooks<br>
│   ├── Case1_evolution.ipynb<br>
│   ├── Case_1_experiments.ipynb<br>
│   ├── Case2_evolution.ipynb<br>
│   ├── Case_2_experiments.ipynb<br>
│   ├── Case3_evolution.ipynb<br>
│   └── Case_3_experiments.ipynb<br>
├── sources<br>
│   ├── mongo_connection<br>
│   ├── plotting<br>
│   ├── problem_formulation<br>
|   ├── parallel_executions.py<br>
│   ├── SensorNetworkDesignABC.py<br>
│   └── settings.py<br>
└── test<br>

* **sources**: contains the code of the proyect.
* **Hive**: Custom Hive library for the ABC optimization.
* **historic_evolution**: contains the output pictures of the an execution evolution.
* **notebooks**: this folder stores the notebooks that allows to run the different experiments. The ones called evolution generates pictures with the fitness of the ABC for each iteration.
* **test**: this folder stores python unittest files.

## Sources estructure
* **mongo_connection**: contains all the code related to experiment execution/storage/loading
* **plotting**: code related to plot different metrics.
* **problem_formulation**: code for the three different SNDP that are tested, includes the feasibility tests.
* **parallel_executions.py**: code for loading several experiments in different threads.
* **SensorNetworkDesignABC.py**: code for the ABC for solving the SNDP.
* **settings.py**: python class for configuring the ABC for solving the SNDP
