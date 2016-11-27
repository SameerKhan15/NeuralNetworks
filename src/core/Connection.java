package core;


public class Connection 
{
	private boolean srcNodeIsInputNode = false;
	private double weight;
	private double gradient, delta=0;
	private String srcNodeId, targetNodeId;
	private String id;
	
	public Connection(String id)
	{
		this.id = id;
	}
	
	public String getId(){ return this.id; }
	
	public void setSrcNodeAsInputNode(){ this.srcNodeIsInputNode = true; }
	public boolean isSrcNodeInputNode(){ return this.srcNodeIsInputNode; }
	
	public void setConnectionWeight(double weight){ this.weight = weight; }
	public double getConnectionWeight(){ return weight; }
	
	public void setSrcNodeId(String srcNodeId){ this.srcNodeId = srcNodeId; }
	public String getSrcNodeId(){ return this.srcNodeId; }
	
	public void setTargetNodeId(String targetNodeId){ this.targetNodeId = targetNodeId; }
	public String getTargetNodeId(){ return this.targetNodeId; }			
	
	public void setGradient(double gradient){ this.gradient = gradient; }
	public double getGradient(){ return this.gradient; }
	
	public void setDelta(double delta){ this.delta = delta; }
	public double getDelta(){ return this.delta; }
}