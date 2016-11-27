package neural.network.graphing;

import java.awt.Color;
import java.awt.geom.Rectangle2D;
import java.text.DecimalFormat;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import javax.swing.JFrame;

import org.apache.log4j.Logger;
import org.jgraph.JGraph;
import org.jgraph.graph.DefaultGraphCell;
import org.jgraph.graph.GraphConstants;
import org.jgrapht.ext.JGraphModelAdapter;
import org.jgrapht.graph.ListenableDirectedWeightedGraph;

import core.Connection;
import core.Neuron;

public class JGraphNetwork extends ANetworkGraph
{
	private class Point
	{
		private int x,y;
		
		private Point(int x, int y)
		{
			this.x = x;
			this.y = y;
		}
		
		private int getX(){ return x; }
		private int getY(){ return y; }
		
		private void setX(int x){ this.x = x; }
		private void setY(int y){ this.y = y; }
	}
	
	private JFrame frame;
	private ListenableDirectedWeightedGraph<String,MyWeightedEdge> directedWeightedGraph;
	private int neuronsHighestLayer = 1;
	private Logger log;
	
	/*Key = Network layer, Value = Last returned Point obj*/
	private Map<Integer,Point> layerLatestPointMap;
	
	/*Key = Network layer, Value = First Point obj*/
	private Map<Integer,Point> layerFirstPointMap;
	
	/*Key = Network layer, Value = Vertex last incremented value*/
	private Map<Integer,Integer> vertexLastIncrementedValue;
	
	private static int VERTEX_INITIAL_LOC_X = 300, VERTEX_INITIAL_LOC_Y = 50, LAYER_INCREMENT_X = 300, LAYER_INCREMENT_Y = 200, ROW_WITHIN_LAYER_X_ADJUSTMENT_FACTOR_PERC[] = {0,-33,-50},
			ROW_WITHIN_LAYER_Y_ADJUSTMENT_VALUE = 200;
	
	/*Note: y values will continue to increment linearly with the length of the network*/
	private static int NUM_X_ADJUSTMENTS_WITHIN_LAYER_= 3;

	public JGraphNetwork(Map<Integer, List<String>> neuronsByLevelMap,
			Map<String, Neuron> neuronsListMap, Map<Integer, List<String>> biasNodesLevel) 
	{
		super(neuronsByLevelMap, neuronsListMap, biasNodesLevel);
		frame = new JFrame("Neural Network Plot");
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		directedWeightedGraph = new ListenableDirectedWeightedGraph<String,MyWeightedEdge>(MyWeightedEdge.class);
		layerLatestPointMap = new HashMap<Integer,Point>();
		layerFirstPointMap = new HashMap<Integer,Point>();
		vertexLastIncrementedValue = new HashMap<Integer,Integer>();
		
		log = Logger.getLogger(JGraphNetwork.class.getName());
		
		/* The width of the neural network is expected to be small enough that calculating highest layer number 
		 * via linear iteration is not going to be a performance issue 
		 */
		for (Map.Entry<Integer, List<String>> entry : neuronsByLevelMap.entrySet())
		{
			if(neuronsHighestLayer < entry.getKey())
			{
				neuronsHighestLayer = entry.getKey();
			}
		}
	}

	private static void positionAtVertexAt(JGraphModelAdapter m_jgAdapter, Object vertex, int x, int y, boolean bridgeNeuron)
	{
		DefaultGraphCell cell = m_jgAdapter.getVertexCell(vertex);
        Map attr = cell.getAttributes();
        Rectangle2D r2d = GraphConstants.getBounds(attr);
        
        double width = r2d.getWidth();
        double height = r2d.getHeight();
        if(bridgeNeuron)
        {
        	width = width*2;
        	height = height*2;
        	GraphConstants.setBackground(attr, Color.black);
        }else
        {
        	GraphConstants.setBackground(attr, Color.gray);
        }
        r2d.setRect(x, y, width, height);
        GraphConstants.setBounds(attr, r2d);
        
        Map cellAttr = new HashMap();
        cellAttr.put( cell, attr );
        m_jgAdapter.edit(cellAttr, null, null, null);
	}
	
	@Override
	public void plotGraph() 
	{
		//Build Vertices and Edges
		buildNetwork();		
		JGraphModelAdapter m_jgAdapter = new JGraphModelAdapter(directedWeightedGraph);
		JGraph jgraph = new JGraph( m_jgAdapter );	        
		jgraph.setEdgeLabelsMovable(true);
	    jgraph.setName("Neural Network");
	    
	    System.out.println(neuronsByLevelMap.size());
	    for(int a = 1 ; a <= neuronsHighestLayer ; a++)
	    {
	    	List<String> neuronslistIter = neuronsByLevelMap.get(a);
	    	for (String neuron : neuronslistIter)
	    	{
	    		if(neuron.contains("i1"))
	    		{
	    			JGraphNetwork.positionAtVertexAt(m_jgAdapter, neuron, 300, 40, false);
	    		}
	    		
	    		if(neuron.contains("i2"))
	    		{
	    			JGraphNetwork.positionAtVertexAt(m_jgAdapter, neuron, 100, 140, false);
	    		}
	    		
	    		if(neuron.contains("h1"))
	    		{
	    			JGraphNetwork.positionAtVertexAt(m_jgAdapter, neuron, 600, 40, false);
	    		}
	    		
	    		if(neuron.contains("h2"))
	    		{
	    			JGraphNetwork.positionAtVertexAt(m_jgAdapter, neuron, 600, 140, false);
	    		}
	    		
	    		if(neuron.contains("o1"))
	    		{
	    			JGraphNetwork.positionAtVertexAt(m_jgAdapter, neuron, 900, 140, false);
	    		}	        
	    	}
	    }
	    
	    for(int a = neuronsHighestLayer ; a > 1 ; a--)
	    {
	    	List<String> biasNodesListIter = this.biasNodesLevel.get(a);
	     	for(String biasNode : biasNodesListIter)
	     	{
	     		if(biasNode.contains("b1"))
	     		{
	     			JGraphNetwork.positionAtVertexAt(m_jgAdapter, biasNode, 100, 500, false);
	     		}
	     		
	     		if(biasNode.contains("b2"))
	     		{
	     			JGraphNetwork.positionAtVertexAt(m_jgAdapter, biasNode, 600, 240, false);
	     		}
	     	}
	    }
	   
	    
	    frame.add(jgraph);
        frame.pack();
	    frame.setLocationByPlatform(true);
	    frame.setVisible(true);
	    frame.setSize(2048, 2048);
	}
	
	/*[x,l1] - [300 -> -33%(200) -> -50%(100) -> (x,l1)(1) -> (x,l1)(2) -> (x,l1)(3) -> ...]
			[y,l1] - [50 -> +200 -> +200 -> ...]

			[x,l2] - [((x,l1)(1) + 300) -> -200 -> -160 -> ((x,l1)(1) + 300)]
			[y,l2] - [(y,l1) -> +200 -> +200 ...]

			[x,l3] - [(x,l3(1) + 300) -> -200 -> -160 -> (x,l3(1) + 300)]
			[y,l3] - [50 -> +200 -> +200 ...]

			.
			.
			.
	*/
	
	public Point getVertexPosition(int nnLayer)
	{
		Point point=null;
		/*Is it the first neuron in this layer*/
		if(!layerLatestPointMap.containsKey(nnLayer))
		{
			/*If so, do we have a previous layer*/
			if(!layerLatestPointMap.containsKey(nnLayer - 1))
			{
				/*First neuron in the first layer*/
				point = new Point(VERTEX_INITIAL_LOC_X, VERTEX_INITIAL_LOC_Y);				
			}else
			{
				/*Get the first Point object from the previous layer*/
				Point prevLayerFirstPoint = this.layerFirstPointMap.get(nnLayer - 1);
				point = new Point(prevLayerFirstPoint.getX() + LAYER_INCREMENT_X, prevLayerFirstPoint.getY() + LAYER_INCREMENT_Y);
				
			}
			layerLatestPointMap.put(nnLayer, point);
			layerFirstPointMap.put(nnLayer, point);
			vertexLastIncrementedValue.put(nnLayer, 1);
		}else
		{
			point = layerLatestPointMap.get(nnLayer);			
			int lastValue = vertexLastIncrementedValue.get(nnLayer);
			lastValue++;
			
			if(lastValue > NUM_X_ADJUSTMENTS_WITHIN_LAYER_)
			{
				lastValue = 1;
			}
			
			int newX = point.getX() + (point.getX() * (ROW_WITHIN_LAYER_X_ADJUSTMENT_FACTOR_PERC[lastValue] / 100));
			int newY = point.getY() + ROW_WITHIN_LAYER_Y_ADJUSTMENT_VALUE;
			point.setX(newX);
			point.setY(newY);
			layerLatestPointMap.put(nnLayer, point);
			vertexLastIncrementedValue.put(nnLayer, lastValue);
		}
		return point;
	}
	
	private void buildNetwork()
	{
		 DecimalFormat twoDForm = new DecimalFormat("#.##");
		//Start with first hidden layer and move rightwards up to the output layer
		log.debug("NN highest layer# "+neuronsHighestLayer);
		for(int a = 1 ; a <= neuronsHighestLayer ; a++)
		{
			/* a) compute source layer 
			 * b) get (connected) source node(s) for each neuron in the layer
			 * c) get connection weight(s) for each connection
			 * d) add source node to ListenableDirectedWeightedGraph, if not already present
			 * e) compute and configure node graph position, based on layer the node belongs in
			 * f) add target node to ListenableDirectedWeightedGraph    
			 * f) add edge and set edge weight for each connection, in the ListenableDirectedWeightedGraph 
			 * */
			int sourceLayer = (a - 1);
			List<String> neuronsList = neuronsByLevelMap.get(a);
			Iterator<String> neuronsListIter = neuronsList.iterator();
			while(neuronsListIter.hasNext())
			{
				Neuron neuron = this.neuronsListMap.get(neuronsListIter.next());
				List<Connection> inboundConnectionsList = neuron.getIncomingConnectionsList();
				Iterator<Connection> inboundConnectionsListIter = inboundConnectionsList.iterator();
				while(inboundConnectionsListIter.hasNext())
				{
					Connection connection = inboundConnectionsListIter.next();
					String srcNodeId = connection.getSrcNodeId();
					if(!directedWeightedGraph.containsVertex(srcNodeId))
					{
						directedWeightedGraph.addVertex(srcNodeId);
					}
					
					if(!directedWeightedGraph.containsVertex(neuron.getId()))
					{
						directedWeightedGraph.addVertex(neuron.getId());
					}
					directedWeightedGraph.setEdgeWeight(directedWeightedGraph.addEdge(srcNodeId, neuron.getId()), 
							Double.valueOf(twoDForm.format(connection.getConnectionWeight())));
				}
			}
		}
	}
}