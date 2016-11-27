#!/bin/bash
echo "$*"
export NeuralNetworks_HOME=/Users/sameer.khan/Documents/workspace/NeuralNetworks
echo $NeuralNetworks_HOME
java -Xmx2048m -Dlog4j.configuration=file:///$NeuralNetworks_HOME/resources/log4j.properties -cp $NeuralNetworks_HOME/lib/log4j-1.2.15.jar:$NeuralNetworks_HOME/lib/nn.jar -Duser.dir=$NeuralNetworks_HOME training.data.load.DigitalLabelLoadingService
