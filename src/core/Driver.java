package core;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.HashMap;
import java.util.Iterator;
import org.apache.log4j.Logger;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Properties;
import java.util.StringTokenizer;

public class Driver 
{	
	public static void main(String args[])
	{
		//[Input Nodes Map] Key = input id, Value = input value
		Map<String, Float> inputsMap = new HashMap<String, Float>();
		
		//[Neurons Map] Key = Neuron Id, Value = Neuron
		Map<String, Neuron> neuronsListMap = new HashMap<String, Neuron>();
		
		//[Bias Nodes] Key = Bias Node Id, Value = input value
		Map<String, Float> biasNodesListMap = new HashMap<String, Float>();
		
		//[Weights] Key = Weight Id, Value = weight value
		Map<String, Float> weightsListMap = new HashMap<String, Float>();
		
		//Step# 1: read Inputs and Weights seed values file and store in in-memory data structures
		//The files should reside within the 'config' folder of the tool
		Logger log = Logger.getLogger(Driver.class.getName());
		Properties props = new Properties();
		
		try {
			props.load(new FileInputStream(System.getProperty("user.dir") + "/config/inputs.properties"));
			Iterator<Entry<Object, Object>> propsIter = props.entrySet().iterator();
			while(propsIter.hasNext())
			{
				String key = (String) propsIter.next().getKey();
				if(key.startsWith("i"))
				{
					Float val = Float.parseFloat(props.getProperty(key));
					inputsMap.put(key, val);
					log.debug("[Input] "+"[Name] "+key+" [Value] "+val);
				}else if(key.startsWith("b"))
				{
					Float val = Float.parseFloat(props.getProperty(key));
					biasNodesListMap.put(key, val);
					log.debug("[Input] "+"[Name] "+key+" [Value] "+val);
				}else if(key.startsWith("w"))
				{
					Float val = Float.parseFloat(props.getProperty(key));
					weightsListMap.put(key, val);
					log.debug("[Input] "+"[Name] "+key+" [Value] "+val);
				}
			}
			
			//Step# 2: read connections config and store in in-memory data structures
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
				
				if(!inputsMap.containsKey(srcNode))
				{
					if(!neuronsListMap.containsKey(srcNode))
					{
						Neuron neuron = new Neuron(srcNode);
						neuronsListMap.put(srcNode, neuron);
					}					
				}else
				{
					connection.setSrcNodeAsInputNode();
				}
				
				connection.setSrcNodeId(srcNode);
				connection.setTargetNodeId(targetNode);
				connection.setConnectionWeight(weightsListMap.get(weight));
				
				if(!neuronsListMap.containsKey(targetNode))
				{
					Neuron neuron = new Neuron(targetNode);
					neuronsListMap.put(targetNode, neuron);
				}
				neuronsListMap.get(targetNode).addIncomingConnection(connection);
			}
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
	    	StringWriter sw = new StringWriter();
			e.printStackTrace(new PrintWriter(sw));
			log.error(sw.toString());
		}
	}
}