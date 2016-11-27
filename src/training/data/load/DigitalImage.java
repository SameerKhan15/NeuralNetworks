package training.data.load;

public class DigitalImage 
{
	private int[][] pixelsData;
	private int height, width;
	
	public DigitalImage(int height, int width)
	{
		this.height = height;
		this.width = width;
		pixelsData = new int[height][width];
	}
	
	public int getPixelValue(int height, int width)
	{
		return pixelsData[height][width];
	}
	
	public void setPixelValue(int height, int width, int value)
	{
		pixelsData[height][width] = value;
	}
	
	public int getWidth() { return this.width; }
	
	public int getHeight() { return this.height; }
}
