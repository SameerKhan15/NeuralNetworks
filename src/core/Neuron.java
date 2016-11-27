package core;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

public class Neuron 
{
	//idealOutput variable will only have value if the neuron is an output neuron
	private double output, sum, lastCalcDelta, idealOutput;
	private List<Connection> incomingConnections, outgoingConnections;
	private String id;
	
	public Neuron(String id)
	{
		this.id = id;
		incomingConnections = new ArrayList<Connection>();
		outgoingConnections = new ArrayList<Connection>();
	}
	
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
	
	public void setOutput(double output){ this.output = output; }
	public double getOutput(){ return this.output; }
	
	public void setSum(double sum){ this.sum = sum; }
	public double getSum(){ return this.sum; }
	
	public void setLastCalcDelta(double delta){ this.lastCalcDelta = delta; }
	public double getLastCalcDelta(){ return this.lastCalcDelta; }
	
	public void setIdealOutput(double idealOutput){ this.idealOutput = idealOutput; }
	public double getIdealOutput(){ return this.idealOutput; }
}