package core;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileReader;
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

import neural.network.graphing.ANetworkGraph;
import neural.network.graphing.JGraphNetwork;

import org.apache.log4j.Logger;

import activation.Functions;

public class NeuralNetworkDriver 
{	
	private Logger log;
	
	//Map<K:Training_Set_Id[Integer], V:Map<K:Input_Id[Integer], V:Input_value>>
	private Map<Integer,Map<String, Double>> inputsMap, outputMap;
	
	//Map<K:Training_Set_Id[Integer], V:Map<K:Neuron_Id[Integer], V:Neuron>>		
	private Map<Integer, Map<String, Neuron>> neuronsListMap;
	
	//[Bias Nodes] Key = Bias Node Id, Value = input value
	private Map<String, Double> biasNodesListMap;
	
	//[Weights] Key = Weight Id, Value = weight value
	private Map<String, Double> weightsListMap;
	
	//Output nodes list - String = key
	private List<String> outputNeuronsList;
	
	//[Neurons Level] Key = Neuron Id, Value = level
	private Map<String, Integer> neuronsLevel;
	
	//[Neurons Level] Key = Level, Value = List<String (Neuron Id)> of Neurons within the level
	private Map<Integer, List<String>> neuronsByLevelMap, biasNodesByLevelsMap;
	
	private int highestLevelNeuronLayer = 0;
	
	private double error = 100;
		
	private double learningRate, momentum;
	
	private void setLearningRate(double rate){ this.learningRate = rate; }
	public double getLearningRate(){ return this.learningRate; }
	
	private void setMomentum(double momentum){ this.momentum = momentum; }
	public double getMomentum(){ return this.momentum; }
	
	protected Logger getLogger(){ return this.log; }
	
	public NeuralNetworkDriver() throws Exception
	{
		log = Logger.getLogger(NeuralNetworkDriver.class.getName());
		
		biasNodesByLevelsMap = new HashMap<Integer, List<String>>();
		inputsMap = new HashMap<Integer, Map<String, Double>>();
		outputMap = new HashMap<Integer, Map<String, Double>>();
		outputNeuronsList = new ArrayList<String>();
		biasNodesListMap = new HashMap<String, Double>();
		weightsListMap = new HashMap<String, Double>();
		neuronsListMap = new HashMap<Integer, Map<String, Neuron>>();
		neuronsLevel = new HashMap<String, Integer>();
		neuronsByLevelMap = new HashMap<Integer, List<String>>();
		
		Properties miscProps = new Properties();
		miscProps.load(new FileInputStream(System.getProperty("user.dir") + "/config/misc.properties"));
		setLearningRate(Double.valueOf(miscProps.getProperty("LearningRate")));
		setMomentum(Double.valueOf(miscProps.getProperty("Momentum")));
	}
	
	private void getNeuronLevel(String node)
	{
		log.debug("network walkthrough for level calc... [node id] "+node);
		Entry<Integer, Map<String, Neuron>> neuronsListMapEntry = neuronsListMap.entrySet().iterator().next();
		Map<String, Neuron> nMap = neuronsListMapEntry.getValue();
		Neuron neuron = nMap.get(node);
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
			}else if(biasNodesListMap.containsKey(srcNode))
			{
				int targetNodeLevel = neuronsLevel.get(connection.getTargetNodeId());
				if(!biasNodesByLevelsMap.containsKey(targetNodeLevel + 1))
				{
					biasNodesByLevelsMap.put(targetNodeLevel + 1, new ArrayList<String>());
				}
				
				if(!biasNodesByLevelsMap.get(targetNodeLevel + 1).contains(srcNode))
				{
					biasNodesByLevelsMap.get(targetNodeLevel + 1).add(srcNode);
				}
			}
		}
	}
	
	private void trainNetwork()
	{
		//Compute node delta for output (layer 1) level neuron(s), for each training set
		List<String> outputNeuronsList = neuronsByLevelMap.get(1);
		Iterator<String> outputNeuronsListIter = outputNeuronsList.iterator();
		while(outputNeuronsListIter.hasNext())
		{
			String neuronId = outputNeuronsListIter.next();
			Iterator<Entry<Integer, Map<String, Neuron>>> neuronsListMapIter =  neuronsListMap.entrySet().iterator();
			while(neuronsListMapIter.hasNext())
			{
				Neuron neuron = neuronsListMapIter.next().getValue().get(neuronId);
				double error = neuron.getOutput() - neuron.getIdealOutput();
				this.error = error;
				this.log.info("Error:"+error);
				//Functions.sigmoidD(..) takes in the output of the sigmoid function
				double nodeDelta = - (error) * (Functions.sigmoidD(neuron.getOutput()));
				neuron.setLastCalcDelta(nodeDelta);
			}
		}
		
		for(int a = 2 ; a <=  highestLevelNeuronLayer; a++)
		{
			List<String> hiddenNeuronsList = neuronsByLevelMap.get(a);
			Iterator<String> hiddenNeuronsListIter = hiddenNeuronsList.iterator();
			while(hiddenNeuronsListIter.hasNext())
			{
				String neuronId = hiddenNeuronsListIter.next();
				Iterator<Entry<Integer, Map<String, Neuron>>> neuronsListMapIter =  neuronsListMap.entrySet().iterator();
				while(neuronsListMapIter.hasNext())
				{
					Map<String, Neuron> neuronsListMap = neuronsListMapIter.next().getValue();
					Neuron neuron = neuronsListMap.get(neuronId);
					double nodeDerivative = Functions.sigmoidD(neuron.getOutput());
					double wToDeltaProd = 0;
					List<Connection> outboundConnectionsList = neuron.getOutgoingConnectionsList();
					Iterator<Connection> outboundConnectionsListIter = outboundConnectionsList.iterator();
					while(outboundConnectionsListIter.hasNext())
					{
						Connection connection = outboundConnectionsListIter.next();
						wToDeltaProd = wToDeltaProd + connection.getConnectionWeight() * neuronsListMap.get(connection.getTargetNodeId()).
								getLastCalcDelta();
					}
					neuron.setLastCalcDelta(nodeDerivative * wToDeltaProd);
				}
			}
		}
		

		//Calculate individual gradient 
		for(int a = highestLevelNeuronLayer ; a >= 1 ; a--)
		{
			List<String> memberNeuronsList = neuronsByLevelMap.get(a);
			Iterator<String> memberNeuronsListIter = memberNeuronsList.iterator();
			while(memberNeuronsListIter.hasNext())
			{
				//Get the connections list from the first training set. Note: all training sets share the same connection objects
				List<Connection> connectionsList = neuronsListMap.entrySet().iterator().next().getValue().get(memberNeuronsListIter.next()).
						getIncomingConnectionsList();
				Iterator<Connection> connectionsListIter = connectionsList.iterator();
				while(connectionsListIter.hasNext())
				{
					Connection connection = connectionsListIter.next();
					double aggregatedGradient = 0;
					//Now that we have the connection object, lets iterate over the training sets, in order to compute the gradients
					Iterator<Entry<Integer, Map<String, Neuron>>> neuronsListMapIter =  neuronsListMap.entrySet().iterator();
					while(neuronsListMapIter.hasNext())
					{
						Entry<Integer, Map<String, Neuron>> entry = neuronsListMapIter.next();
						int trainingSetNumber = entry.getKey();
						Map<String, Neuron> neuronsListMap = entry.getValue();
						double targetNodeLastCalcDelta = neuronsListMap.get(connection.getTargetNodeId()).getLastCalcDelta();
						double srcNodeLastOutput = 0;
						if(connection.getSrcNodeId().startsWith("i"))
						{
							srcNodeLastOutput = this.inputsMap.get(trainingSetNumber).get(connection.getSrcNodeId());
							
						}else if(connection.getSrcNodeId().startsWith("b"))
						{
							srcNodeLastOutput = this.biasNodesListMap.get(connection.getSrcNodeId());
						}
						else
						{
							srcNodeLastOutput = neuronsListMap.get(connection.getSrcNodeId()).getOutput();
						}						
						aggregatedGradient = aggregatedGradient + (targetNodeLastCalcDelta * srcNodeLastOutput);
					}
					double newDelta = (learningRate * aggregatedGradient) + momentum * connection.getDelta();
					connection.setGradient(aggregatedGradient);
					connection.setDelta(newDelta);
					connection.setConnectionWeight(connection.getConnectionWeight() + newDelta);
				}
			}
		}
	}
	
	private void calculateNeuronsLevel()
	{
		Entry<Integer, Map<String, Neuron>> neuronsListMapEntry = neuronsListMap.entrySet().iterator().next();
		Map<String, Neuron> nMap = neuronsListMapEntry.getValue();
		
		Iterator<String> outputNeuronsListIter = outputNeuronsList.iterator();
		while(outputNeuronsListIter.hasNext())
		{
			Neuron outermostNeuron = nMap.get(outputNeuronsListIter.next());
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
				}else if(biasNodesListMap.containsKey(srcNode))
				{
					int targetNodeLevel = neuronsLevel.get(connection.getTargetNodeId());
					if(!biasNodesByLevelsMap.containsKey(targetNodeLevel))
					{
						biasNodesByLevelsMap.put(targetNodeLevel + 1, new ArrayList<String>());
					}
					
					if(!biasNodesByLevelsMap.get(targetNodeLevel + 1).contains(srcNode))
					{
						biasNodesByLevelsMap.get(targetNodeLevel + 1).add(srcNode);
					}
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
	
	private void loadBiasNodesData() throws Exception
	{
		Properties biasNodesProps = new Properties();		
		biasNodesProps.load(new FileInputStream(System.getProperty("user.dir") + "/data/bias_nodes.properties"));
		Iterator<Entry<Object, Object>> propsIter = biasNodesProps.entrySet().iterator();
		while(propsIter.hasNext())
		{
			String key = (String) propsIter.next().getKey();
			biasNodesListMap.put(key, Double.valueOf((String) biasNodesProps.get(key)));
		}
	}
	
	private void loadTrainingData() throws Exception
	{
		//Read the training data
		BufferedReader br = null;		
		try{	
			br = new BufferedReader(new FileReader(System.getProperty("user.dir") + "/data/training_data.csv"));
			String line = null;
			this.log.info("Loading training data...");
	        this.log.debug("Printing training data...");
	        while ((line = br.readLine()) != null) 
			{
	        	if(!line.startsWith("#"))		    	
	        	{			       		
	        		StringTokenizer strTok = new StringTokenizer(line,",");
			        int trainingSetNumber = Integer.valueOf(strTok.nextToken());			     		
			        this.neuronsListMap.put(trainingSetNumber, new HashMap<String, Neuron>());
			        int val = strTok.countTokens();			        		
			        int tokenCounter=0;
			        log.debug("[Training Set#]"+trainingSetNumber);
			        inputsMap.put(trainingSetNumber, new HashMap<String,Double>());
			        outputMap.put(trainingSetNumber, new HashMap<String,Double>());
			        while(strTok.hasMoreTokens())
			        {			        			
			        	double value = Double.valueOf((String) strTok.nextElement());
			        	tokenCounter++;
			        	if(tokenCounter < val)
			        	{
			        		inputsMap.get(trainingSetNumber).put("i"+tokenCounter, value);
			        		log.debug("[Input#]"+tokenCounter+"[Input Value]"+value);        							        			
			        	}else			        			
			        	{
			        		outputMap.get(trainingSetNumber).put("o"+1, value);
			        		log.debug("[Output#]"+tokenCounter+"[Output Value]"+value);			        			
			        	}			        		
			        }			        	
	        	}			        
			}				
		}finally				
		{					
			if(br != null)					
			{						
				br.close();					
			}				
		}
	}
	
	private void loadConnectionWeights() throws Exception
	{
		Properties connectionWeightsProps = new Properties();		
		connectionWeightsProps.load(new FileInputStream(System.getProperty("user.dir") + "/data/weights.properties"));
		Iterator<Entry<Object, Object>> propsIter = connectionWeightsProps.entrySet().iterator();
		while(propsIter.hasNext())
		{
			String key = (String) propsIter.next().getKey();
			weightsListMap.put(key, Double.valueOf((String) connectionWeightsProps.get(key)));
		}
	}
	
	private void setIdealOutput() throws Exception
	{
		int numberOfTrainingSets = this.neuronsListMap.size();	
		for(int a = 1 ; a <= numberOfTrainingSets ; a++)
		{
			Map<String, Double> outputM = this.outputMap.get(a);
			Iterator outputMIter = outputM.entrySet().iterator();
			while(outputMIter.hasNext())
			{
				Map.Entry pair = (Map.Entry)outputMIter.next();
				String outputNeuronId = (String) pair.getKey();
				Double outputNeuronIdealValue = (Double) pair.getValue();
				this.neuronsListMap.get(a).get(outputNeuronId).setIdealOutput(outputNeuronIdealValue);
			}
		}
	}
	
	@SuppressWarnings("unchecked")
	private void loadConnections() throws Exception
	{
		List<String> srcNodesList = new ArrayList<String>();
		Properties connectionsProps = new Properties();
		connectionsProps.load(new FileInputStream(System.getProperty("user.dir") + "/data/connections.properties"));
		Iterator<Entry<Object, Object>> propsIter = connectionsProps.entrySet().iterator();
		while(propsIter.hasNext())
		{
			String key = (String) propsIter.next().getKey();
			String val = connectionsProps.getProperty(key);
			
			Connection connection = new Connection(key);
			
			StringTokenizer strTok = new StringTokenizer(val,",");
			String srcNode = strTok.nextToken();
			String weight = strTok.nextToken();
			String targetNode = strTok.nextToken();
			
			Map iMap = inputsMap.get(1);
			
			if(!iMap.containsKey(srcNode) && !biasNodesListMap.containsKey(srcNode))
			{
				if(!iMap.containsKey(srcNode))
				{
					Iterator<Entry<Integer, Map<String, Neuron>>> neuronsListMapIter = neuronsListMap.entrySet().iterator();
					while(neuronsListMapIter.hasNext())
					{
						Neuron neuron = new Neuron(srcNode);
						Entry<Integer, Map<String, Neuron>> pair = neuronsListMapIter.next();
						pair.getValue().put(srcNode, neuron);
						srcNodesList.add(srcNode);
					}
				}
			}else
			{
				connection.setSrcNodeAsInputNode();
			}
			
			log.debug("[srcNode]"+srcNode+"[targetNode]"+targetNode+"[weight]"+weight);
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
			
			Map nMap = neuronsListMap.get(1);
			
			if(!nMap.containsKey(targetNode))
			{
				Iterator<Entry<Integer, Map<String, Neuron>>> neuronsListMapIter = neuronsListMap.entrySet().iterator();
				while(neuronsListMapIter.hasNext())
				{
					Neuron neuron = new Neuron(targetNode);
					Entry<Integer, Map<String, Neuron>> pair = neuronsListMapIter.next();
					pair.getValue().put(targetNode, neuron);				
				}
			}
			
			if(!nMap.containsKey(srcNode))
			{
				Iterator<Entry<Integer, Map<String, Neuron>>> neuronsListMapIter = neuronsListMap.entrySet().iterator();
				while(neuronsListMapIter.hasNext())
				{
					Neuron neuron = new Neuron(srcNode);
					Entry<Integer, Map<String, Neuron>> pair = neuronsListMapIter.next();
					pair.getValue().put(srcNode, neuron);			
				}
			}
			
			Iterator<Entry<Integer, Map<String, Neuron>>> neuronsListMapIter = neuronsListMap.entrySet().iterator();
			while(neuronsListMapIter.hasNext())
			{
				Map<String, Neuron> neuronsListMap = neuronsListMapIter.next().getValue();				
				neuronsListMap.get(targetNode).addIncomingConnection(connection);
				neuronsListMap.get(srcNode).addOutgoingConnection(connection);
			}
		}
	}
	
	public void buildNetwork() throws Exception
	{
		this.log.info("Loading Training Data...");
		loadTrainingData();
		
		this.log.info("Loading Bias Nodes Data...");
		loadBiasNodesData();
		
		this.log.info("Loading Connections Weights Data...");
		loadConnectionWeights();
		
		this.log.info("Loading Connections Data..");
		loadConnections();
		
		this.log.info("Setting ideal output..");
		setIdealOutput();
		
		this.log.info("Calculating Neurons hierarchy level..");
		calculateNeuronsLevel();
		
		this.log.info("Printing Network...");
		printNetwork();
		
		this.log.info("Solving the Network...");
		solve();
		
		this.log.info("Printing Neuron Data...");
		printNeuronData();
	}
	
	public void solve()
	{
		for(int a = highestLevelNeuronLayer ; a >= 1 ; a--)
		{
			List<String> memberNeuronsList = neuronsByLevelMap.get(a);
			Iterator<String> memberNeuronsListIter = memberNeuronsList.iterator();
			while(memberNeuronsListIter.hasNext())
			{
				String neuronId = memberNeuronsListIter.next();
				Iterator<Entry<Integer, Map<String, Neuron>>> neuronsListMapIter = neuronsListMap.entrySet().iterator();
				while(neuronsListMapIter.hasNext())
				{
					double sum = 0;
					Entry<Integer, Map<String, Neuron>> pair = neuronsListMapIter.next();
					Integer trainingSetId = pair.getKey();
					Map<String, Neuron> neuronsListMap = pair.getValue();
					log.info("Processing Neuron "+neuronId+" at level "+a+" for training set id# "+trainingSetId);
					Neuron neuron = neuronsListMap.get(neuronId);
					
					List<Connection> incomingConnections = neuron.getIncomingConnectionsList();
					Iterator<Connection> incomingConnectionsIter = incomingConnections.iterator();
					while(incomingConnectionsIter.hasNext())
					{
						Connection conn = incomingConnectionsIter.next();
						double weight = conn.getConnectionWeight();
						if(conn.isSrcNodeInputNode())
						{
							if(this.inputsMap.get(1).containsKey(conn.getSrcNodeId()))
							{
								sum = sum + (this.inputsMap.get(trainingSetId).get(conn.getSrcNodeId()) * weight);
							}else
							{
								sum = sum + (this.biasNodesListMap.get(conn.getSrcNodeId()) * weight);
							}
						}else
						{
							Neuron sourceNeuron = this.neuronsListMap.get(trainingSetId).get(conn.getSrcNodeId());
							sum = sum + (sourceNeuron.getOutput() * weight);
						}
					}
					neuron.setOutput(Functions.sigmoid(sum));
					neuron.setSum(sum);
				}
			}
		}
		
		log.info("Training the network...");
		trainNetwork();
	}
	
	public void printNeuronData()
	{
		Iterator<Entry<Integer, Map<String, Neuron>>> neuronsListMapIter = neuronsListMap.entrySet().iterator();
		while(neuronsListMapIter.hasNext())
		{
			Entry<Integer, Map<String, Neuron>> pair = neuronsListMapIter.next();
			Integer trainingSetId = pair.getKey();
			Map<String, Neuron> neuronsListMap = pair.getValue();
			List<String> memberNeuronsList = neuronsByLevelMap.get(1);
			Iterator<String> memberNeuronsListIter = memberNeuronsList.iterator();
			log.info("Printing Neuron Data for Training Set# "+trainingSetId);
			while(memberNeuronsListIter.hasNext())
			{
				String neuronId = memberNeuronsListIter.next();
				Neuron neuron = neuronsListMap.get(neuronId);
				DecimalFormat numberFormat = new DecimalFormat("#.0000");
				log.info("Level 1 Neuron Id "+neuronId+" has output "+numberFormat.format(neuron.getOutput()));
				log.info("Level 1 Neuron Id "+neuronId+" has sum "+numberFormat.format(neuron.getSum()));
			}
		}		
	}
	
	public void printNetwork()
	{
		DecimalFormat numberFormat = new DecimalFormat("#.0000");
		log.info("[highest layer of NN] "+highestLevelNeuronLayer);
		 
		for(int a = highestLevelNeuronLayer ; a >= 1 ; a--)		
		{
			List<String> memberNeuronsList = neuronsByLevelMap.get(a);
			Iterator<String> memberNeuronsListIter = memberNeuronsList.iterator();
			
			Entry<Integer, Map<String, Neuron>> neuronsListMapEntry = neuronsListMap.entrySet().iterator().next();
			Map<String, Neuron> nMap = neuronsListMapEntry.getValue();
			
			while(memberNeuronsListIter.hasNext())		
			{
				Neuron neuron = nMap.get(memberNeuronsListIter.next());
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
	
	public void drawNetworkGraph()
	{
		ANetworkGraph graph = new JGraphNetwork(this.neuronsByLevelMap, this.neuronsListMap.get(1), biasNodesByLevelsMap);
		graph.plotGraph();
	}
	
	public void printOutputLayer()
	{
		List<String> memberNeuronsList = neuronsByLevelMap.get(1);
		Iterator<String> memberNeuronsListIter = memberNeuronsList.iterator();
		while(memberNeuronsListIter.hasNext())
		{
			String neuronId = memberNeuronsListIter.next();
			Iterator<Entry<Integer, Map<String, Neuron>>>  neuronsListMapIter = neuronsListMap.entrySet().iterator();
			while(neuronsListMapIter.hasNext())
			{	
				Entry<Integer, Map<String, Neuron>> entry = neuronsListMapIter.next();
				Integer trainingSetKey = entry.getKey();
				Neuron neuron = entry.getValue().get(neuronId);
				log.info("======================================================");
				log.info("Training Set Key:"+trainingSetKey+", Neuron Id:"+neuron.getId()+", Neuron Actual Output:"+neuron.getOutput()+", Neuron Ideal Output:"+neuron.getIdealOutput());
			}
		}
	}
	
	public static void main(String args[])
	{
		NeuralNetworkDriver nnDriver = null;
		Logger log = Logger.getLogger(Driver.class.getName());	
		try{
			int iter = 0;
			nnDriver = new NeuralNetworkDriver();
			nnDriver.getLogger().info("[Learning Rate]"+nnDriver.getLearningRate());
			nnDriver.getLogger().info("[Momentum]"+nnDriver.getMomentum());	
			nnDriver.buildNetwork();
			while(Math.abs(nnDriver.error) >= .01)	
			{					
				nnDriver.solve();
				nnDriver.printNetwork();
				nnDriver.printOutputLayer();
				iter++;
				log.info("iteration# "+iter);
			}	
			log.info("total # of epochs:"+iter);
			
			log.info("Drawing network graph...");
			nnDriver.drawNetworkGraph();
		}
		catch(Exception e)
		{
			StringWriter sw = new StringWriter();
			e.printStackTrace(new PrintWriter(sw));
			nnDriver.getLogger().error(sw.toString());	
		}			
	}
}