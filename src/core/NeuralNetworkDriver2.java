package core;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.StringTokenizer;
import java.util.Map.Entry;
import org.apache.log4j.Logger;

import activation.Functions;

public class NeuralNetworkDriver2 
{
	private Logger log;
	
	//[Input Nodes Map] Key = input id, Value = input value
	private Map<String, Float> inputsMap; 
	
	//[Neurons Map] Key = Neuron Id, Value = Neuron
	private Map<String, Neuron> neuronsListMap;
	
	//[Bias Nodes] Key = Bias Node Id, Value = input value
	private Map<String, Float> biasNodesListMap;
	
	//[Weights] Key = Weight Id, Value = weight value
	private Map<String, Float> weightsListMap;
	
	//Output nodes list - String = key
	private List<String> outputNeuronsList;
	
	//[Neurons Level] Key = Neuron Id, Value = level
	private Map<String, Integer> neuronsLevel;
	
	//[Neurons Level] Key = Level, Value = List<String (Neuron Id)> of Neurons within the level
	private Map<Integer, List<String>> neuronsByLevelMap;
	
	private int highestLevelNeuronLayer = 0;
	
	private double learningRate, momentum;
	
	private double error = 100;
			
	public NeuralNetworkDriver2()
	{
		inputsMap = new HashMap<String, Float>();
		neuronsListMap = new HashMap<String, Neuron>();
		biasNodesListMap = new HashMap<String, Float>();
		weightsListMap = new HashMap<String, Float>();
		outputNeuronsList = new ArrayList<String>();
		neuronsLevel = new HashMap<String, Integer>();
		neuronsByLevelMap = new HashMap<Integer, List<String>>();
		log = Logger.getLogger(Driver.class.getName());
	}
	
	public void setLearningRate(double rate){ this.learningRate = rate; }
	public double getLearningRate(){ return this.learningRate; }
	
	public void setMomentum(double momentum){ this.momentum = momentum; }
	public double getMomentum(){ return this.momentum; }
	
	public static void main(String args[])
	{
		NeuralNetworkDriver2 neuralNetworkDriver = new NeuralNetworkDriver2();
		neuralNetworkDriver.setLearningRate(0.7);
		neuralNetworkDriver.setMomentum(0.3);
		
		Logger log = Logger.getLogger(Driver.class.getName());
		
		try{
			int iter = 0;
			neuralNetworkDriver.buildNetwork();
			while(Math.abs(neuralNetworkDriver.error) >= .01)
			{
				neuralNetworkDriver.solve();
				neuralNetworkDriver.printNetwork();
				iter++;
				log.info("iteration# "+iter);
			}
			log.info("total # of epochs:"+iter);
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException io) {
			io.printStackTrace();
	    	StringWriter sw = new StringWriter();
			io.printStackTrace(new PrintWriter(sw));
			log.error(sw.toString());
		} catch(Exception e) {
			e.printStackTrace();
	    	StringWriter sw = new StringWriter();
			e.printStackTrace(new PrintWriter(sw));
			log.error(sw.toString());
		}
	}
	
	private void getNeuronLevel(String node)
	{
		log.debug("network walkthrough for level calc... [node id] "+node);
		Neuron neuron = neuronsListMap.get(node);
		List<Connection> incomingConnections = neuron.getIncomingConnectionsList();
		Iterator<Connection> incomingConnectionsIter = incomingConnections.iterator();
		while(incomingConnectionsIter.hasNext())
		{
			Connection connection = incomingConnectionsIter.next();
			String srcNode = connection.getSrcNodeId();
			if(!inputsMap.containsKey(srcNode) && !biasNodesListMap.containsKey(srcNode))
			{
				int targetNodeLevel = neuronsLevel.get(connection.getTargetNodeId());
				if(!neuronsLevel.containsKey(srcNode))
				{
					neuronsLevel.put(srcNode, targetNodeLevel+1);
				}else
				{
					int currentLevel = neuronsLevel.get(srcNode);
					if(currentLevel < (targetNodeLevel + 1))
					{
						neuronsLevel.put(srcNode, currentLevel);
					}
				}
				getNeuronLevel(srcNode);
			}
		}
	}
	
	private void calculateNeuronsLevel()
	{
		Iterator<String> outputNeuronsListIter = outputNeuronsList.iterator();
		while(outputNeuronsListIter.hasNext())
		{
			Neuron outermostNeuron = neuronsListMap.get(outputNeuronsListIter.next());
			this.neuronsLevel.put(outermostNeuron.getId(), 1);
			log.debug("network walkthrough for level calc... [node id] "+outermostNeuron.getId());
			List<Connection> incomingConnections = outermostNeuron.getIncomingConnectionsList();
			Iterator<Connection> incomingConnectionsIter = incomingConnections.iterator();
			while(incomingConnectionsIter.hasNext())
			{
				Connection connection = incomingConnectionsIter.next();
				String srcNode = connection.getSrcNodeId();
				
				if(!inputsMap.containsKey(srcNode) && !biasNodesListMap.containsKey(srcNode))
				{
					int targetNodeLevel = neuronsLevel.get(connection.getTargetNodeId());
					if(!neuronsLevel.containsKey(srcNode))
					{
						neuronsLevel.put(srcNode, targetNodeLevel+1);
					}else
					{
						int currentLevel = neuronsLevel.get(srcNode);
						if(currentLevel < (targetNodeLevel + 1))
						{
							neuronsLevel.put(srcNode, currentLevel);
						}
					}
					getNeuronLevel(srcNode);
				}
			}
		}
		
		Iterator<Entry<String, Integer>> neuronsLevelIter = neuronsLevel.entrySet().iterator();
		while(neuronsLevelIter.hasNext())
		{
			Map.Entry<String, Integer> pair = (Map.Entry<String, Integer>)neuronsLevelIter.next();
			String neuronId = pair.getKey();
			int level = pair.getValue();
			if(!neuronsByLevelMap.containsKey(level))
			{
				if(level > highestLevelNeuronLayer)
				{
					highestLevelNeuronLayer = level;
				}
				neuronsByLevelMap.put(level, new ArrayList<String>());
			}
			neuronsByLevelMap.get(level).add(neuronId);
		}
	}
	
	private void trainNetwork()
	{
		//Compute node delta for output (layer 1) level neurons
		List<String> outputNeuronsList = neuronsByLevelMap.get(1);
		Iterator<String> outputNeuronsListIter = outputNeuronsList.iterator();
		while(outputNeuronsListIter.hasNext())
		{
			Neuron neuron = neuronsListMap.get(outputNeuronsListIter.next());
			double error = neuron.getOutput() - neuron.getIdealOutput();
			this.error = error;
			this.log.info("Error:"+error);
			//Functions.sigmoidD(..) takes in the output of the sigmoid function
			double nodeDelta = -(error) * (Functions.sigmoidD(neuron.getOutput()));
			neuron.setLastCalcDelta(nodeDelta);
		}
		
		for(int a = 2 ; a <=  highestLevelNeuronLayer; a++)
		{
			List<String> hiddenNeuronsList = neuronsByLevelMap.get(a);
			Iterator<String> hiddenNeuronsListIter = hiddenNeuronsList.iterator();
			while(hiddenNeuronsListIter.hasNext())
			{
				Neuron neuron = neuronsListMap.get(hiddenNeuronsListIter.next());
				double nodeDerivative = Functions.sigmoidD(neuron.getOutput());
				double wToDeltaProd = 0;
				List<Connection> outboundConnectionsList = neuron.getOutgoingConnectionsList();
				Iterator<Connection> outboundConnectionsListIter = outboundConnectionsList.iterator();
				while(outboundConnectionsListIter.hasNext())
				{
					Connection connection = outboundConnectionsListIter.next();
					wToDeltaProd = wToDeltaProd + connection.getConnectionWeight() * neuronsListMap.get(connection.getTargetNodeId()).getLastCalcDelta();
				}
				neuron.setLastCalcDelta(nodeDerivative * wToDeltaProd);
			}
		}
		
		//Calculate individual gradient 
		for(int a = highestLevelNeuronLayer ; a >= 1 ; a--)
		{
			List<String> memberNeuronsList = neuronsByLevelMap.get(a);
			Iterator<String> memberNeuronsListIter = memberNeuronsList.iterator();
			while(memberNeuronsListIter.hasNext())
			{
				List<Connection> connectionsList = neuronsListMap.get(memberNeuronsListIter.next()).getIncomingConnectionsList();
				Iterator<Connection> connectionsListIter = connectionsList.iterator();
				while(connectionsListIter.hasNext())
				{
					Connection connection = connectionsListIter.next();
					
					double targetNodeLastCalcDelta = neuronsListMap.get(connection.getTargetNodeId()).getLastCalcDelta();
					double srcNodeLastOutput = 0;
					if(connection.getSrcNodeId().startsWith("i"))
					{
						srcNodeLastOutput = this.inputsMap.get(connection.getSrcNodeId());
						
					}else if(connection.getSrcNodeId().startsWith("b"))
					{
						srcNodeLastOutput = this.biasNodesListMap.get(connection.getSrcNodeId());
					}
					else
					{
						srcNodeLastOutput = neuronsListMap.get(connection.getSrcNodeId()).getOutput();
					}
					
					double gradient = targetNodeLastCalcDelta * srcNodeLastOutput;
					double newDelta = (learningRate * gradient) + momentum * connection.getDelta();
					connection.setGradient(gradient);
					connection.setDelta(newDelta);
					connection.setConnectionWeight(connection.getConnectionWeight() + newDelta);
				}
			}
		}
	}
	
	public void solve()
	{
		for(int a = highestLevelNeuronLayer ; a >= 1 ; a--)
		{
			List<String> memberNeuronsList = neuronsByLevelMap.get(a);
			Iterator<String> memberNeuronsListIter = memberNeuronsList.iterator();
			while(memberNeuronsListIter.hasNext())
			{
				double sum = 0;
				String neuronId = memberNeuronsListIter.next();
				log.info("Processing Neuron "+neuronId+" at level "+a);
				Neuron neuron = neuronsListMap.get(neuronId);
				List<Connection> incomingConnections = neuron.getIncomingConnectionsList();
				Iterator<Connection> incomingConnectionsIter = incomingConnections.iterator();
				while(incomingConnectionsIter.hasNext())
				{
					Connection conn = incomingConnectionsIter.next();
					double weight = conn.getConnectionWeight();
					if(conn.isSrcNodeInputNode())
					{
						log.info("processing connection source node id "+conn.getSrcNodeId());
						if(this.inputsMap.containsKey(conn.getSrcNodeId()))
						{
							sum = sum + (this.inputsMap.get(conn.getSrcNodeId()) * weight);
						}else
						{
							sum = sum + (this.biasNodesListMap.get(conn.getSrcNodeId()) * weight);
						}
					}else
					{
						log.info("processing connection source node id "+conn.getSrcNodeId());
						Neuron sourceNeuron = this.neuronsListMap.get(conn.getSrcNodeId());
						sum = sum + (sourceNeuron.getOutput() * weight);
					}
				}
				neuron.setOutput(Functions.sigmoid(sum));
				neuron.setSum(sum);
			}
		}
		
		List<String> memberNeuronsList = neuronsByLevelMap.get(1);
		Iterator<String> memberNeuronsListIter = memberNeuronsList.iterator();
		while(memberNeuronsListIter.hasNext())
		{
			String neuronId = memberNeuronsListIter.next();
			Neuron neuron = neuronsListMap.get(neuronId);
			DecimalFormat numberFormat = new DecimalFormat("#.0000");
			log.info("Level 1 Neuron Id "+neuronId+" has output "+numberFormat.format(neuron.getOutput()));
			log.info("Level 1 Neuron Id "+neuronId+" has sum "+numberFormat.format(neuron.getSum()));
		}
		
		log.info("Training the network...");
		trainNetwork();
	}
	
	public void printNetwork()
	{
		DecimalFormat numberFormat = new DecimalFormat("#.0000");
		log.info("[highest layer of NN] "+highestLevelNeuronLayer);
		
		//Calculate individual gradient 
		for(int a = highestLevelNeuronLayer ; a >= 1 ; a--)		
		{
			List<String> memberNeuronsList = neuronsByLevelMap.get(a);
			Iterator<String> memberNeuronsListIter = memberNeuronsList.iterator();
			while(memberNeuronsListIter.hasNext())		
			{
				Neuron neuron = neuronsListMap.get(memberNeuronsListIter.next());
				log.info("==============================================================");
				log.info("[Neuron id] "+neuron.getId()+" [LastCalcDelta] "+neuron.getLastCalcDelta()+" [Output] "+neuron.getOutput()+
						" [Sum] "+neuron.getSum()+" [Incoming Connections List Size] "+neuron.getIncomingConnectionsList().size()+" [Outgoing Connections List Size] "
						+neuron.getOutgoingConnectionsList().size());
				List<Connection> connectionsList = neuron.getIncomingConnectionsList();
				Iterator<Connection> connectionsListIter = connectionsList.iterator();
				while(connectionsListIter.hasNext())		
				{
					Connection connection = connectionsListIter.next();					
					log.info("Connection Source Node Id:"+connection.getSrcNodeId()+","+"Connection Target Node Id:"+connection.getTargetNodeId()+
							"Gradient:"+connection.getGradient());
				}					
			}				
		}
	}
	
	public void buildNetwork() throws Exception
	{
		//Step# 1: read inputs and weights seed values file and store within in-memory data structures
		Properties props = new Properties();
		props.load(new FileInputStream(System.getProperty("user.dir") + "/config/inputs.properties"));
		Iterator<Entry<Object, Object>> propsIter = props.entrySet().iterator();
		while(propsIter.hasNext())
		{
			String key = (String) propsIter.next().getKey();
			// i = input node
			if(key.startsWith("i"))
			{
				Float val = Float.parseFloat(props.getProperty(key));
				inputsMap.put(key, val);
				log.debug("[Input] "+"[Name] "+key+" [Value] "+val);
			// b = bias node
			}else if(key.startsWith("b"))
			{
				Float val = Float.parseFloat(props.getProperty(key));
				biasNodesListMap.put(key, val);
				log.debug("[Input] "+"[Name] "+key+" [Value] "+val);
			// w = weight
			}else if(key.startsWith("w"))
			{
				Float val = Float.parseFloat(props.getProperty(key));
				weightsListMap.put(key, val);
				log.debug("[Input] "+"[Name] "+key+" [Value] "+val);
			}
		}
		
		//Step# 2: read connections config and store in in-memory data structures
		List<String> srcNodesList = new ArrayList<String>();
		props = new Properties();
		props.load(new FileInputStream(System.getProperty("user.dir") + "/config/connections.properties"));
		Iterator<Entry<Object, Object>> connPropsIter = props.entrySet().iterator();
		while(connPropsIter.hasNext())
		{
			String key = (String) connPropsIter.next().getKey();
			String val = props.getProperty(key);
			log.debug("[Input] "+"[Name] "+key+" [Value] "+val);
			
			Connection connection = new Connection(key);
			
			StringTokenizer strTok = new StringTokenizer(val,",");
			String srcNode = strTok.nextToken();
			String weight = strTok.nextToken();
			String targetNode = strTok.nextToken();
			
			if(!inputsMap.containsKey(srcNode) && !biasNodesListMap.containsKey(srcNode))
			{
				if(!neuronsListMap.containsKey(srcNode))
				{
					Neuron neuron = new Neuron(srcNode);
					neuronsListMap.put(srcNode, neuron);
					srcNodesList.add(srcNode);
				}					
			}else
			{
				connection.setSrcNodeAsInputNode();
			}
			
			log.debug("[srcNode] "+srcNode);
			connection.setSrcNodeId(srcNode);
			connection.setTargetNodeId(targetNode);
			connection.setConnectionWeight(weightsListMap.get(weight));
			
			if(!srcNodesList.contains(targetNode) && !outputNeuronsList.contains(targetNode))
			{
				log.debug("[adding targetNode] "+targetNode);
				outputNeuronsList.add(targetNode);
			}
			
			if(outputNeuronsList.contains(srcNode))
			{
				outputNeuronsList.remove(srcNode);
			}
			
			if(!neuronsListMap.containsKey(targetNode))
			{
				Neuron neuron = new Neuron(targetNode);
				neuronsListMap.put(targetNode, neuron);
			}
			
			if(!neuronsListMap.containsKey(srcNode))
			{
				Neuron neuron = new Neuron(srcNode);
				neuronsListMap.put(srcNode, neuron);
			}
			neuronsListMap.get(targetNode).addIncomingConnection(connection);
			neuronsListMap.get(srcNode).addOutgoingConnection(connection);
		}
		
		props = new Properties();
		props.load(new FileInputStream(System.getProperty("user.dir") + "/config/outputs.properties"));
		propsIter = props.entrySet().iterator();
		while(propsIter.hasNext())
		{
			String neuronId = (String) propsIter.next().getKey();
			Double idealVal = Double.parseDouble((String) props.get(neuronId));
			this.neuronsListMap.get(neuronId).setIdealOutput(idealVal);
		}
		calculateNeuronsLevel();
	}
}