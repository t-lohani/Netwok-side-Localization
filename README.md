# LocalizationProject

The testandtrain.py script is used for simulation of the project. To start the script for populating the database with locations and signal strength vector execute the below command :

python testandtrain.py -p -r <Number of Data points> -t <Number of transmitters>

To start the application in learning mode, execute the below command :

python testandtrain.py -l 1

To start the scripting in testing mode, execute the below command :

python testandtrain.py -c 1 -d <Test Data(Signal strength vector)>
