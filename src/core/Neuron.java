package core;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class Neuron 
{
	public class NeuronTrainingSetCalcHolder
	{
		private double output, sum, lastCalcNodeDelta, idealOutput, error;
		
		private void setError(double error){ this.error = error; }
		public double getError(){ return this.error; }
		
		private void setOutput(double output){ this.output = output; }
		public double getOutput(){ return this.output; }
		
		private void setSum(double sum){ this.sum = sum; }
		public double getSum(){ return this.sum; }
		
		private void setLastCalcNodeDelta(double delta){ this.lastCalcNodeDelta = delta; }
		public double getLastCalcNodeDelta(){ return this.lastCalcNodeDelta; }
		
		private void setIdealOutput(double idealOutput){ this.idealOutput = idealOutput; }
		public double getIdealOutput(){ return this.idealOutput; }
	}
	
	public enum Type {
		INPUT_NONBIAS, INPUT_BIAS, HIDDEN_NONBIAS, HIDDEN_BIAS, OUTPUT
	}
	
	//idealOutput variable will only have value if the neuron is an output neuron
	private double output, sum, lastCalcDelta, idealOutput;
	private List<Connection> incomingConnections, outgoingConnections;
	private String id;
	private Type type;
	
	//K => Unique identifier for a given training input dataset, V => Calculations holder for the training input 
	private Map<Integer,NeuronTrainingSetCalcHolder> perInputCalculationsMap;
	
	public Neuron(String id, Type type)
	{
		this.type = type;
		perInputCalculationsMap = new HashMap<>();
		this.id = id;
		incomingConnections = new ArrayList<Connection>();
		outgoingConnections = new ArrayList<Connection>();
	}
	
	/*TODO: Deprecated*/
	public Neuron(String id)
	{
		perInputCalculationsMap = new HashMap<>();
		this.id = id;
		incomingConnections = new ArrayList<Connection>();
		outgoingConnections = new ArrayList<Connection>();		
	}
	
	public Set<Integer> getTrainingSetIds()
	{
		return Collections.unmodifiableSet(perInputCalculationsMap.keySet());
	}
	
	public void removeTrainingDataSet(int trainingDataSetId)
	{
		perInputCalculationsMap.remove(trainingDataSetId);
	}
	
	public Type getType(){ return this.type; }
	
	public String getId(){ return this.id; }
	
	public void addIncomingConnection(Connection iConnection)
	{
		this.incomingConnections.add(iConnection);
	}
	
	public List<Connection> getIncomingConnectionsList(){ return this.incomingConnections; }
	
	public void addOutgoingConnection(Connection oConnection)
	{
		this.outgoingConnections.add(oConnection);
	}
	
	public List<Connection> getOutgoingConnectionsList(){ return this.outgoingConnections; }
	
	private void ensureEntryInMapExists(int trainingInputId)
	{
		if(!perInputCalculationsMap.containsKey(trainingInputId))
		{
			perInputCalculationsMap.put(trainingInputId, new NeuronTrainingSetCalcHolder());
		}
	}
	
	public void addError(int trainingInputId, double error)
	{
		ensureEntryInMapExists(trainingInputId);
		perInputCalculationsMap.get(trainingInputId).setError(error);
	}
	
	public double getError(int trainingInputId)
	{
		return perInputCalculationsMap.get(trainingInputId).getError();
	}
	
	public void addOutput(int trainingInputId, double output)
	{
		ensureEntryInMapExists(trainingInputId);
		perInputCalculationsMap.get(trainingInputId).setOutput(output);
	}
	
	public double getOutput(int trainingInputId)
	{
		return perInputCalculationsMap.get(trainingInputId).output;
	}
	
	public void addSum(int trainingInputId, double sum)
	{
		ensureEntryInMapExists(trainingInputId);
		perInputCalculationsMap.get(trainingInputId).setSum(sum);
	}
	
	public double getSum(int trainingInputId)
	{
		return perInputCalculationsMap.get(trainingInputId).getSum();
	}
	
	public void addLastCalcNodeDelta(int trainingInputId, double lastCalcNodeDelta)
	{
		ensureEntryInMapExists(trainingInputId);
		perInputCalculationsMap.get(trainingInputId).setLastCalcNodeDelta(lastCalcNodeDelta);
	}
	
	public double getLastCalcNodeDelta(int trainingInputId)
	{
		return perInputCalculationsMap.get(trainingInputId).getLastCalcNodeDelta();
	}
	
	public void addIdealOutput(int trainingInputId, double idealOutput)
	{
		ensureEntryInMapExists(trainingInputId);
		perInputCalculationsMap.get(trainingInputId).setIdealOutput(idealOutput);
	}
	
	public double getIdealOutput(int trainingInputId)
	{
		return perInputCalculationsMap.get(trainingInputId).idealOutput;
	}
	
	/*Needs to be deprecated*/
	public void setOutput(double output){ this.output = output; }
	public double getOutput(){ return this.output; }
	
	/*Needs to be deprecated*/
	public void setSum(double sum){ this.sum = sum; }
	public double getSum(){ return this.sum; }
	
	/*Needs to be deprecated*/
	public void setLastCalcDelta(double delta){ this.lastCalcDelta = delta; }
	public double getLastCalcDelta(){ return this.lastCalcDelta; }
	
	/*Needs to be deprecated*/
	public void setIdealOutput(double idealOutput){ this.idealOutput = idealOutput; }
	public double getIdealOutput(){ return this.idealOutput; }
}