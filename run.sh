#!/bin/bash
echo "$*"
export NeuralNetworks_HOME=/Users/sameer.khan/Documents/workspace/NeuralNetworks
echo $NeuralNetworks_HOME
java -Xmx2048m -Dlog4j.configuration=file:///$NeuralNetworks_HOME/resources/log4j.properties -cp $NeuralNetworks_HOME/lib/jgraph-5.13.0.0.jar:$NeuralNetworks_HOME/lib/jgrapht-core-0.9.2.jar:$NeuralNetworks_HOME/lib/jgrapht-demo-0.9.2.jar:$NeuralNetworks_HOME/lib/jgrapht-ext-0.9.2-uber.jar:$NeuralNetworks_HOME/lib/jgrapht-ext-0.9.2.jar:$NeuralNetworks_HOME/lib/jgraphx-2.0.0.1.jar:$NeuralNetworks_HOME/lib/log4j-1.2.15.jar:$NeuralNetworks_HOME/lib/nn.jar -Duser.dir=$NeuralNetworks_HOME core.NeuralNetworkDriver
