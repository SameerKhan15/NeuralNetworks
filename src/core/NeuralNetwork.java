package core;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import activation.Functions;
import core.Neuron.Type;

import java.util.concurrent.ThreadLocalRandom;

import neural.network.graph.NeuralNetworkGraph;

public class NeuralNetwork {
	
	private String networkId = null;
	private double globalError = 100, globalError2 = 100;
	
	/*Map<K => SrcNode->TargetNode, V => Connection Weight*/
	private Map<String,Double> preCalculatedWeights;
	
	//Map<K => Neuron_Id [Integer], V => Neuron>>				
	private Map<String, Neuron> neuronsListMap;
	
	//Map<K => Neurons Level, V => List of Neurons>
	private Map<Integer, LinkedHashSet<String>> neuronsByLevelMap;
	
	private double learningRate, momentum;
	
	private void setLearningRate(double rate){ this.learningRate = rate; }
	public double getLearningRate(){ return this.learningRate; }
	
	private void setMomentum(double momentum){ this.momentum = momentum; }
	public double getMomentum(){ return this.momentum; }
	
	/*Hidden Layer: Map<K => Hidden Layer Number (where K >= 2, since layer 1 is for reserved for input neurons, 
	 * V => Number of neurons in the layer>*/	
	public NeuralNetwork(int numInputNeurons, int numHiddenLayerNeurons, 
			int numOutputNeurons, String networkId) throws FileNotFoundException, IOException
	{			
		neuronsListMap = new HashMap<String, Neuron>();
		neuronsByLevelMap = new HashMap<>();
		
		/*Load the weights from the file, as oppose to initializing randomly*/
		if(networkId != null)
		{
			this.networkId = networkId;
			preCalculatedWeights = new HashMap<>();
			BufferedReader br = null;
			try{		
				br = new BufferedReader(new FileReader(System.getProperty("user.dir")+"/"+networkId));
				String line;
				while((line = br.readLine()) != null)
				{
					String srcNode = line.split("->")[0];
					String targetNode = line.split("->")[1].split("=")[0];
					double weight = Double.valueOf(line.split("=")[1].trim());					
					preCalculatedWeights.put(srcNode+"->"+targetNode, weight);
				}
			}finally
			{
				if(br != null)
				{
					try {
						br.close();
					} catch (IOException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				}
			}		
		}
		/*Neuron Naming convention: Input Neurons = i<neuron number>
		 * 							Hidden Neurons = h<layer number>+"_"+<neuron number>
		 * 							Output Neurons = o<layer number>+"_"+<neuron number>*/
		
		/*Create the input layer*/
		for(int a = 1 ; a <= numInputNeurons ; a++)
		{
			if(!neuronsByLevelMap.containsKey(1))
			{
				neuronsByLevelMap.put(1, new LinkedHashSet<String>());			
			}
			
			String neuronName = "i"+(a);
			neuronsByLevelMap.get(1).add(neuronName);
			Neuron neuron = new Neuron(neuronName, Type.INPUT_NONBIAS);
			neuronsListMap.put(neuronName, neuron);		
			
		}
		/*Add the bias neuron to the input layer*/
		String iLayerbiasNeuronName = "b1";
		neuronsByLevelMap.get(1).add(iLayerbiasNeuronName);
		Neuron iLayerBiasNeuron = new Neuron(iLayerbiasNeuronName, Type.INPUT_BIAS);
		neuronsListMap.put(iLayerbiasNeuronName, iLayerBiasNeuron);
		
		/*Create the hidden layers*/
		for(int a = 1 ; a <= numHiddenLayerNeurons ; a++)
		{				
			if(!neuronsByLevelMap.containsKey(2))				
			{					
				neuronsByLevelMap.put(2, new LinkedHashSet<String>());							
			}
								
			String neuronName = "h"+"1"+"_"+a;					
			neuronsByLevelMap.get(2).add(neuronName);				
			Neuron neuron = new Neuron(neuronName, Type.HIDDEN_NONBIAS);					
			neuronsListMap.put(neuronName, neuron);				
		}
		/*Add the bias neuron to the input layer*/
		String hLayerBiasNeuronName = "b2";
		neuronsByLevelMap.get(2).add(hLayerBiasNeuronName);
		Neuron hLayerBiasNeuron = new Neuron(hLayerBiasNeuronName, Type.HIDDEN_BIAS);
		neuronsListMap.put(hLayerBiasNeuronName, hLayerBiasNeuron);
		
		/*Create the output neurons*/
		for(int a = 1 ; a <= numOutputNeurons ; a++)
		{
			if(!neuronsByLevelMap.containsKey(3))
			{
				neuronsByLevelMap.put(3, new LinkedHashSet<String>());			
			}
				
			String neuronName = "o"+a;
			neuronsByLevelMap.get(3).add(neuronName);
			Neuron neuron = new Neuron(neuronName, Type.OUTPUT);
			neuronsListMap.put(neuronName, neuron);
		}
		
		/*Establish synapses, with random weights*/
		/*Input neurons -> layer2 hidden neurons*/
		LinkedHashSet<String> inputNeuronsList = neuronsByLevelMap.get(1);
		Iterator<String> inputNeuronsListIter = inputNeuronsList.iterator();
		while(inputNeuronsListIter.hasNext())
		{
			String inputNeuronName = inputNeuronsListIter.next();
			Neuron inputNeuron = this.neuronsListMap.get(inputNeuronName);
			LinkedHashSet<String> hiddenNeuronsLayer1List = neuronsByLevelMap.get(2);
			Iterator<String> hiddenNeuronsLayer1ListIter = hiddenNeuronsLayer1List.iterator();
			while(hiddenNeuronsLayer1ListIter.hasNext())
			{
				String hiddenLayer1NeuronName = hiddenNeuronsLayer1ListIter.next();
				Neuron hiddenLayer1Neuron = this.neuronsListMap.get(hiddenLayer1NeuronName);
				if(hiddenLayer1Neuron.getType() != Type.HIDDEN_BIAS)
				{
					Connection connection = new Connection(inputNeuronName+"->"+hiddenLayer1NeuronName);
					connection.isSrcNodeInputNode();
					if(preCalculatedWeights != null)
					{
						connection.setConnectionWeight(preCalculatedWeights.get(inputNeuronName+"->"+hiddenLayer1NeuronName));
					}else
					{
						connection.setConnectionWeight(ThreadLocalRandom.current().nextDouble(-1, 1.01));
					}
					System.out.println("[Initial Synapses Weights]"+ inputNeuronName+"->"+hiddenLayer1NeuronName+" - "+connection.getConnectionWeight());
					connection.setSrcNodeId(inputNeuronName);
					connection.setTargetNodeId(hiddenLayer1NeuronName);
					inputNeuron.addOutgoingConnection(connection);
					hiddenLayer1Neuron.addIncomingConnection(connection);
				}			
			}
		}
		
		/*Hidden Neurons -> Output Neurons*/
		LinkedHashSet<String> hiddenNeuronsList = neuronsByLevelMap.get(2);
		Iterator<String> hiddenNeuronsListIter = hiddenNeuronsList.iterator();
		while(hiddenNeuronsListIter.hasNext())
		{
			String hiddenLayerNeuronName = hiddenNeuronsListIter.next();
			Neuron hiddenLayerNeuron = this.neuronsListMap.get(hiddenLayerNeuronName);
			LinkedHashSet<String> outputLayerNeuronsList = neuronsByLevelMap.get(3);
			Iterator<String> outputLayerNeuronsListIter = outputLayerNeuronsList.iterator();
			while(outputLayerNeuronsListIter.hasNext())
			{
				String outputLayerNeuronName = outputLayerNeuronsListIter.next();
				Neuron outputLayerNeuron = this.neuronsListMap.get(outputLayerNeuronName);
				Connection connection = new Connection(hiddenLayerNeuronName+"->"+outputLayerNeuronName);
				if(preCalculatedWeights != null)
				{
					connection.setConnectionWeight(preCalculatedWeights.get(hiddenLayerNeuronName+"->"+outputLayerNeuronName));
				}else
				{
					connection.setConnectionWeight(ThreadLocalRandom.current().nextDouble(-1, 1.01));
				}
				System.out.println("[Initial Synapses Weights]"+ hiddenLayerNeuronName+"->"+outputLayerNeuronName+" - "+connection.getConnectionWeight());
				connection.setSrcNodeId(hiddenLayerNeuronName);
				connection.setTargetNodeId(outputLayerNeuronName);
				hiddenLayerNeuron.addOutgoingConnection(connection);
				outputLayerNeuron.addIncomingConnection(connection);
			}
		}
	}
	
	public void plotGraph() throws Exception
	{
		Map<Integer, String> biasedNeuronsMap = new HashMap<>();
		biasedNeuronsMap.put(1, "b1");
		biasedNeuronsMap.put(2, "b2");
		
		Map<String,neural.network.graph.Neuron> neuronsMap = new HashMap<>();
		Iterator<Entry<String, Neuron>> neuronsLevelIter = this.neuronsListMap.entrySet().iterator();
		while(neuronsLevelIter.hasNext())
		{
			Neuron neuron = neuronsLevelIter.next().getValue();
			neural.network.graph.Neuron neuronForGraph = new neural.network.graph.Neuron(neuron.getId());
			neuronsMap.put(neuron.getId(), neuronForGraph);
			Iterator<Connection> neuronConnectionsListIter = neuron.getOutgoingConnectionsList().iterator();
			while(neuronConnectionsListIter.hasNext())
			{
				Connection con = neuronConnectionsListIter.next();
				neuronForGraph.addOutgoingConnection(con.getTargetNodeId(), con.getConnectionWeight());			
			}
		}
		
		NeuralNetworkGraph.plot(neuronsByLevelMap, neuronsMap, biasedNeuronsMap);
	}
	
	private void calculateNodeDeltas()
	{
		int numSamplesForGlobalErrorCalc=0;
		/*For each training set, compute the node delta for the output layer*/
		/*for now, we *only* support 3 layer neural network so layer 3 is always the output layer*/
		LinkedHashSet<String> outputLayerMemberNeuronsList = neuronsByLevelMap.get(3);
		Iterator<String> outputLayerMemberNeuronsListIter = outputLayerMemberNeuronsList.iterator();
		while(outputLayerMemberNeuronsListIter.hasNext())
		{
			String neuronId = outputLayerMemberNeuronsListIter.next();
			Neuron neuron = this.neuronsListMap.get(neuronId);
			Iterator<Integer> trainingSetIdsIter = neuron.getTrainingSetIds().iterator();
			while(trainingSetIdsIter.hasNext())
			{
				int trainingDataSetId = trainingSetIdsIter.next();
				double err = neuron.getOutput(trainingDataSetId) - neuron.getIdealOutput(trainingDataSetId);
				//Functions.sigmoidD(..) takes in the output of the sigmoid function
				double nodeDelta = - (err) * (Functions.sigmoidD(neuron.getOutput(trainingDataSetId)));
				neuron.addLastCalcNodeDelta(trainingDataSetId, nodeDelta);
				numSamplesForGlobalErrorCalc++;
				globalError = err;
				globalError2 = globalError2 + Math.pow(err,2);
			}		
		}
		globalError2 = globalError2 / numSamplesForGlobalErrorCalc;
		
		/*For each training set, compute the node delta for the hidden layer
		 * for now, we *only* support 3 layer neural network so layer 2 is always the hidden layer*/
		LinkedHashSet<String> hiddenLayerMemberNeuronsList = neuronsByLevelMap.get(2);
		Iterator<String> hiddenLayerMemberNeuronsListIter = hiddenLayerMemberNeuronsList.iterator();
		while(hiddenLayerMemberNeuronsListIter.hasNext())
		{
			String neuronId = hiddenLayerMemberNeuronsListIter.next();
			Neuron neuron = this.neuronsListMap.get(neuronId);
			Iterator<Integer> trainingSetIdsIter = neuron.getTrainingSetIds().iterator();
			while(trainingSetIdsIter.hasNext())
			{
				int trainingDataSetId = trainingSetIdsIter.next();
				double nodeDerivative = Functions.sigmoidD(neuron.getOutput(trainingDataSetId));
				double wToDeltaProd = 0;
				List<Connection> outboundConnectionsList = neuron.getOutgoingConnectionsList();
				Iterator<Connection> outboundConnectionsListIter = outboundConnectionsList.iterator();
				while(outboundConnectionsListIter.hasNext())
				{
					Connection connection = outboundConnectionsListIter.next();
					wToDeltaProd = wToDeltaProd + connection.getConnectionWeight() * neuronsListMap.get(connection.getTargetNodeId()).
							getLastCalcNodeDelta(trainingDataSetId);
				}
				neuron.addLastCalcNodeDelta(trainingDataSetId, nodeDerivative * wToDeltaProd);
			}
		}
	}
	
	private void computeGradients()
	{
		for(int a = 1 ; a <= 2 ; a++)
		{
			LinkedHashSet<String> inputLayerMemberNeuronsList = neuronsByLevelMap.get(a);
			Iterator<String> inputLayerMemberNeuronsListIter = inputLayerMemberNeuronsList.iterator();
			while(inputLayerMemberNeuronsListIter.hasNext())
			{
				String neuronId = inputLayerMemberNeuronsListIter.next();
				Neuron neuron = this.neuronsListMap.get(neuronId);
				List<Connection> outboundConnectionsList = neuron.getOutgoingConnectionsList();
				Iterator<Connection> outboundConnectionsListIter = outboundConnectionsList.iterator();		
				
				while(outboundConnectionsListIter.hasNext())
				{
					Connection connection = outboundConnectionsListIter.next();
					Iterator<Integer> trainingSetIdsIter = neuron.getTrainingSetIds().iterator();
					double aggregatedGradient = 0;
					while(trainingSetIdsIter.hasNext())
					{
						int trainingDataSetId = trainingSetIdsIter.next();
						double srcNodeLastOutput = neuron.getOutput(trainingDataSetId);
						double targetNodeLastCalcDelta = this.neuronsListMap.get(connection.getTargetNodeId()).getLastCalcNodeDelta(trainingDataSetId);
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
	
	public String train(double[][] input, double[][] output, double desiredErrorVal, int maxEpochs) 
			throws EpochsExceededException, Exception
	{
		int epochNumber = 0;
		while(Math.abs(this.globalError2) >= .0001)	
		{	
			runEpoc(input, output);					
			epochNumber++;
			System.out.print("$> epoch# < " + epochNumber + " > records.\r");
			
			if(epochNumber > maxEpochs)
			{
				throw new EpochsExceededException("Unable to train the network within the desired # of epochs");
			}
		}
		System.out.println("\nTotal # of epochs:"+epochNumber);
		return persistNetwork();
	}
	
	//the training set passed in will be processed as a single batch, for training
	private void runEpoc(double[][] input, double[][] output)
	{
		this.globalError = 0;
		this.globalError2 = 0;
		computeNodes(input, output, true);
		calculateNodeDeltas();
		computeGradients();
	}
	
	public String persistNetwork() throws Exception
	{
		BufferedWriter dataWriter = null;
		String networkFileName = null;
		try{
			File networkFile;		
			
			if(networkId != null)
			{
				networkFileName = networkId;
			}else
			{
				networkFileName = String.valueOf("neural.network."+System.currentTimeMillis());	
			}
			
			networkFile = new File(System.getProperty("user.dir")+"/"+networkFileName);
			networkFile.createNewFile();
			dataWriter = new BufferedWriter(new FileWriter(networkFile.getAbsoluteFile()));
			
			//hidden and output layer
			for(int a = 2 ; a <= 3 ; a++)
			{
				LinkedHashSet<String> memberNeuronsList = neuronsByLevelMap.get(a);
				Iterator<String> memberNeuronsListIter = memberNeuronsList.iterator();
				while(memberNeuronsListIter.hasNext())
				{
					String neuronId = memberNeuronsListIter.next();
					Neuron neuron = this.neuronsListMap.get(neuronId);
					
					if(a == 3)
					{
						System.out.println("==================================================================================================================================");
						System.out.println("[Neuron id] "+neuron.getId()+" [Type] "+neuron.getType()+" [Incoming Connections List Size] "+neuron.getIncomingConnectionsList().size()
								+" [Outgoing Connections List Size] "+neuron.getOutgoingConnectionsList().size());
						Iterator<Integer> trainingSetIdsIter = neuron.getTrainingSetIds().iterator();
						while(trainingSetIdsIter.hasNext())
						{
							int trainingDataSetId = trainingSetIdsIter.next();
							System.out.println("[trainingDataSetId] "+trainingDataSetId+" [Sum] "+neuron.getSum(trainingDataSetId)+
									" [Node Delta] "+neuron.getLastCalcNodeDelta(trainingDataSetId)+" [Output] "+neuron.getOutput(trainingDataSetId)+
									" [IdealOutput] "+neuron.getIdealOutput(trainingDataSetId));
						}
					}
					
					List<Connection> connectionsList = neuron.getIncomingConnectionsList();
					Iterator<Connection> connectionsListIter = connectionsList.iterator();
					while(connectionsListIter.hasNext())		
					{
						Connection connection = connectionsListIter.next();					
						System.out.println("Connection Source Node Id:"+connection.getSrcNodeId()+","+"Connection Target Node Id:"+connection.getTargetNodeId()+
								"Gradient:"+connection.getGradient()+","+"Connection weight:"+connection.getConnectionWeight());
						dataWriter.write(connection.getSrcNodeId()+"->"+connection.getTargetNodeId()+"="+connection.getConnectionWeight());
						dataWriter.write("\n");
					}
				}
			}
			System.out.println("Global Error: "+this.globalError+", Global Error2: "+this.globalError2);
		}finally
		{
			if(dataWriter != null)
			{
				dataWriter.close();
			}
		}
		return networkFileName;
	}
	
	public double[] getOutput(double[] input)
	{
		double[][] inputForComputation = new double[1][input.length];
		for(int a = 0 ; a < input.length ; a++)
		{
			inputForComputation[0][a] = input[a];
		}
		computeNodes(inputForComputation, null, false);
		
		LinkedHashSet<String> memberNeuronsList = neuronsByLevelMap.get(3);
		double[] output = new double[memberNeuronsList.size()];
		Iterator<String> memberNeuronsListIter = memberNeuronsList.iterator();
		int outputNeuronsCounter=0;
		while(memberNeuronsListIter.hasNext())
		{
			String neuronId = memberNeuronsListIter.next();
			Neuron neuron = this.neuronsListMap.get(neuronId);
			output[outputNeuronsCounter] = neuron.getOutput();
			outputNeuronsCounter++;
		}
		return output;
	}
	
	private void computeNodes(double[][] input, double[][] output, boolean dataForTraining)
	{
		//for now, we *only* support 3 layer neural network
		//Start with layer 2 and move forward
		for(int a = 2 ; a <= 3 ; a++)
		{
			//Go over the training dataset
			for(int b = 0 ; b < input.length ; b++)
			{
				LinkedHashSet<String> memberNeuronsList = neuronsByLevelMap.get(a);
				Iterator<String> memberNeuronsListIter = memberNeuronsList.iterator();
				while(memberNeuronsListIter.hasNext())
				{
					String neuronId = memberNeuronsListIter.next();
					Neuron neuron = this.neuronsListMap.get(neuronId);
					
					List<Connection> incomingConnections = neuron.getIncomingConnectionsList();
					Iterator<Connection> incomingConnectionsIter = incomingConnections.iterator();
					int incomingConnectionsIterPointer = 0;
					double sum = 0;
					while(incomingConnectionsIter.hasNext())
					{
						Connection connection = incomingConnectionsIter.next();
						double weight = connection.getConnectionWeight();
						/*This is the hidden layer, therefore use input neurons values to compute the weighted sum*/
						if(a == 2)
						{
							//This is the bias neuron. Hence, use 1 as the input
							if(incomingConnectionsIterPointer == input[b].length)
							{
								sum = sum + ((1) * weight);
								if(dataForTraining)
								{
									/*Set input neuron value for the training set*/
									neuronsListMap.get(connection.getSrcNodeId()).addOutput(b, 1);
								}else
								{
									neuronsListMap.get(connection.getSrcNodeId()).setOutput(1);
								}								
							}else
							{
								if(dataForTraining)
								{
									/*Set input neuron value for the training set*/
									neuronsListMap.get(connection.getSrcNodeId()).addOutput(b, input[b][incomingConnectionsIterPointer]);
								}else
								{
									neuronsListMap.get(connection.getSrcNodeId()).setOutput(input[b][incomingConnectionsIterPointer]);
								}								
								sum = sum + (input[b][incomingConnectionsIterPointer]) * weight;
								incomingConnectionsIterPointer++;
							}		
						}
						/*This is the output layer, therefore use the hidden layer neurons activation output to compute the weighted sum*/
						else if(a == 3)
						{
							if(incomingConnectionsIterPointer == input[b].length)
							{
								sum = sum + ((1) * weight);
							}else
							{
								if(dataForTraining)
								{
									sum = sum + (this.neuronsListMap.get(connection.getSrcNodeId()).getOutput(b)) * weight;
								}else
								{
									sum = sum + (this.neuronsListMap.get(connection.getSrcNodeId()).getOutput()) * weight;
								}
								
								incomingConnectionsIterPointer++;
							}	
						}
					}
					if(dataForTraining)
					{
						neuron.addOutput(b,Functions.sigmoid(sum));
						neuron.addSum(b,sum);
					}else
					{
						neuron.setOutput(Functions.sigmoid(sum));
						neuron.setSum(sum);
					}					
				}
			}
		}
		
		if(dataForTraining)
		{
			/*Set the ideal outputs to the output neuron layer*/
			int outputLayerNeuronCounter = 0;
			LinkedHashSet<String> memberNeuronsList = neuronsByLevelMap.get(3);
			Iterator<String> memberNeuronsListIter = memberNeuronsList.iterator();
			while(memberNeuronsListIter.hasNext())
			{
				String neuronId = memberNeuronsListIter.next();
				Neuron neuron = this.neuronsListMap.get(neuronId);
				for(int c = 0 ; c < output.length ; c++)
				{
					neuron.addIdealOutput(c, output[c][outputLayerNeuronCounter]);
				}
				outputLayerNeuronCounter++;
			}
		}	
	}
	
	public static void main(String args[])
	{		
		try{
			NeuralNetwork neuralNetwork = new NeuralNetwork(2,2,2,"neural.network.1485141383960");
			neuralNetwork.setLearningRate(3);
			neuralNetwork.setMomentum(0.3);
			
			/*double[][] input = new double[4][2];
			double[][] output = new double[4][1];*/
			
			/*Example training dataset for [2,2,1] nn*/
			/*input[0][0] = 1;
			input[0][1] = 0;
			output[0][0] = 1;
					
			input[1][0] = 0;
			input[1][1] = 0;
			output[1][0] = 0;
			
			input[2][0] = 0;
			input[2][1] = 1;
			output[2][0] = 1;
			
			input[3][0] = 1;
			input[3][1] = 1;
			output[3][0] = 0;*/
			
			double[][] input = new double[2][2];
			double[][] output = new double[2][2];
			
			/*Example training dataset for [2,2,2] nn*/
			input[0][0] = 0.2;
			input[0][1] = 0.5;
			output[0][0] = 0;
			output[0][1] = 1;
			
			input[1][0] = 0.3;
			input[1][1] = 0.6;
			output[1][0] = 0;
			output[1][1] = 1;
			
			/*input[2][0] = 0.02;
			input[2][1] = 0.8;
			output[2][0] = 0;
			output[2][1] = 1;
			
			input[3][0] = 0.9;
			input[3][1] = 0.4;
			output[3][0] = 1;
			output[3][1] = 0;*/
			
			//System.out.println("Network Name:"+neuralNetwork.train(input, output, .0001, 10000));
			
			//System.out.println("Network Name:"+neuralNetwork.train(input, output, .0001, 10000));
			double[] input3 = new double[2];
			input3[0] = 0.05;
			input3[1] = 0.4;
			double[] output3 = neuralNetwork.getOutput(input3);
			System.out.println("ideal output[0,1]");
			for(int a = 0 ; a < output3.length ; a++)
			{
				System.out.println("output["+a+"]"+output3[a]);
			}
			
			double[] input4 = new double[2];
			input4[0] = 0.5;
			input4[1] = 0.1;
			double[] output4 = neuralNetwork.getOutput(input4);
			System.out.println("ideal output[1,0]");
			for(int a = 0 ; a < output4.length ; a++)
			{
				System.out.println("output["+a+"]"+output4[a]);
			}
			
			double[] input5 = new double[2];
			input5[0] = 0.5;
			input5[1] = 0.8;
			double[] output5 = neuralNetwork.getOutput(input5);
			System.out.println("ideal output[0,1]");
			for(int a = 0 ; a < output5.length ; a++)
			{
				System.out.println("output["+a+"]"+output5[a]);
			}
			
			double[] input6 = new double[2];
			input6[0] = 0.9;
			input6[1] = 0.1;
			double[] output6 = neuralNetwork.getOutput(input6);
			System.out.println("ideal output[1,0]");
			for(int a = 0 ; a < output6.length ; a++)
			{
				System.out.println("output["+a+"]"+output6[a]);
			}
			
			neuralNetwork.plotGraph();
		}catch(Exception e)
		{
			e.printStackTrace();
		}		
	}
}