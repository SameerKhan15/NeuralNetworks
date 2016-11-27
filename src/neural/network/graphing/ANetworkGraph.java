package neural.network.graphing;

import java.util.List;
import java.util.Map;

import core.Neuron;

public abstract class ANetworkGraph 
{
	//[Map<Integer, List<String>>] Key = Level, Value = List<String (Neuron Id)> of Neurons within the level
	protected Map<Integer, List<String>> neuronsByLevelMap;
	
	//[Map<String, Neuron>] Key = Neuron Id, Value = Neuron
	protected Map<String, Neuron> neuronsListMap;
	
	protected Map<Integer, List<String>> biasNodesLevel;
	
	public ANetworkGraph(Map<Integer, List<String>> neuronsByLevelMap, Map<String, Neuron> neuronsListMap, Map<Integer, List<String>> biasNodesLevel)
	{
		this.biasNodesLevel = biasNodesLevel;
		this.neuronsByLevelMap = neuronsByLevelMap;
		this.neuronsListMap = neuronsListMap;
	}
	
	public abstract void plotGraph();
}
