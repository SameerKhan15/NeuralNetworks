package training.data.load;

import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import javax.imageio.ImageIO;
import org.apache.log4j.Logger;

public class DigitalLabelLoadingService 
{
	private Logger log;
	private String labelFileName, imageFileName;
	private static final int OFFSET_SIZE = 4; //in bytes
	private static final int LABEL_MAGIC_NUMBER = 2049;
    private static final int IMAGE_MAGIC_NUMBER = 2051;
    private static final int IMAGE_OFFSET = 16; //in bytes
    private ArrayList<DigitalImage> imagesList;
    private ArrayList<Integer> labelsList;
    
	public  DigitalLabelLoadingService(String labelFileName, String imageFileName)
	{
		log = Logger.getLogger(DigitalLabelLoadingService.class.getName());
		this.labelFileName = labelFileName;
		this.imageFileName = imageFileName;
		imagesList = new ArrayList<DigitalImage>();
		labelsList = new ArrayList<Integer>();
	}
	
	protected Logger getLogger(){ return this.log; }
	
	public void loadLabels() throws Exception 
	{
		ByteArrayOutputStream labelBuffer = new ByteArrayOutputStream();
		
		int read;
		byte[] buffer = new byte[16384];
		InputStream labelInputStream = null;
		
		try{
			labelInputStream = new FileInputStream(labelFileName);
			while((read = labelInputStream.read(buffer, 0, buffer.length)) != -1) 
			{
				labelBuffer.write(buffer, 0, read);	        
			}
			labelBuffer.flush();
			this.getLogger().info("labelBuffer size:"+labelBuffer.size());
			
			byte[] labelBytes = labelBuffer.toByteArray();
			byte[] labelMagic = Arrays.copyOfRange(labelBytes, 0, OFFSET_SIZE);
			
			if(ByteBuffer.wrap(labelMagic).getInt() != LABEL_MAGIC_NUMBER)  
			{
	            throw new Exception("Bad magic number in label file!");
	        }
			
			byte[] numOfLabels = Arrays.copyOfRange(labelBytes, 4, 4 + OFFSET_SIZE);
			this.getLogger().info(ByteBuffer.wrap(numOfLabels).getInt());
			
			int intialOffSet = 8;
			
			for(int a = 0 ; a < ByteBuffer.wrap(numOfLabels).getInt() ; a++)
			{
				int label = labelBytes[intialOffSet + a];
				labelsList.add(label);
			}
		}finally
		{
			if(labelInputStream != null)
			{
				labelInputStream.close();
			}
		}	
	}
	
	public void printRandomImage() throws Exception
	{
		Random rand = new Random();
		int imageLocation = rand.nextInt(this.imagesList.size());
		DigitalImage image = imagesList.get(imageLocation);
		int label = labelsList.get(imageLocation);
		
		BufferedImage bufferedImage = new BufferedImage(image.getWidth(), image.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
		WritableRaster raster = bufferedImage.getRaster(); 
		
		for(int a = 0 ; a < bufferedImage.getHeight() ; a++)
		{
			for(int b = 0 ; b < bufferedImage.getWidth(); b++)
			{
				int value = image.getPixelValue(a, b);
				raster.setSample(a, b, 0, value);
			}
		}
		
		File f = new File(System.getProperty("user.dir")+"/"+label+".png");
		this.getLogger().info("printing image at location:"+System.getProperty("user.dir")+"/"+label+".png");
		ImageIO.write(bufferedImage, "PNG", f);
	}
	
	public void loadImages() throws Exception
	{
		ByteArrayOutputStream imageBuffer = new ByteArrayOutputStream();
		
		int read;
		byte[] buffer = new byte[16384];
		InputStream imageInputStream = null;
		long t1 = System.currentTimeMillis();
		try{
			imageInputStream = new FileInputStream(imageFileName);
			while((read = imageInputStream.read(buffer, 0, buffer.length)) != -1) 
			{
				imageBuffer.write(buffer, 0, read);	        
			}
			imageBuffer.flush();
			this.getLogger().info("Image Buffer size:"+imageBuffer.size());
			
			byte[] imageBytes = imageBuffer.toByteArray();
			byte[] imageMagic = Arrays.copyOfRange(imageBytes, 0, OFFSET_SIZE);
			
			if(ByteBuffer.wrap(imageMagic).getInt() != IMAGE_MAGIC_NUMBER)  
			{
	            throw new Exception("Bad magic number in image file!");
	        }
			
			byte[] numOfImages = Arrays.copyOfRange(imageBytes, 4, 4 + OFFSET_SIZE);
			byte[] numOfRows = Arrays.copyOfRange(imageBytes, 8, 8 + OFFSET_SIZE);
			byte[] numOfCols = Arrays.copyOfRange(imageBytes, 12, 12 + OFFSET_SIZE);
			
			this.getLogger().info("# of images:"+ByteBuffer.wrap(numOfImages).getInt()+", # of rows:"+ByteBuffer.wrap(numOfRows).getInt()+", # of cols:"+
					ByteBuffer.wrap(numOfCols).getInt());
			
			int numBytesRead=0;
			for(int a = 0 ; a < ByteBuffer.wrap(numOfImages).getInt() ; a++)
			{
				DigitalImage digitalImage = new DigitalImage(ByteBuffer.wrap(numOfRows).getInt(), ByteBuffer.wrap(numOfCols).getInt());
				for(int b = 0 ; b < ByteBuffer.wrap(numOfCols).getInt() ; b++)
				{
					for(int c = 0; c < ByteBuffer.wrap(numOfRows).getInt() ; c++)
					{
						int pixelValue = imageBytes[IMAGE_OFFSET + numBytesRead];
						digitalImage.setPixelValue(c, b, pixelValue);
						numBytesRead++;
					}
				}
				imagesList.add(digitalImage);
			}
		}
		finally
		{
			if(imageInputStream != null)
			{
				imageInputStream.close();
			}
			long t2 = System.currentTimeMillis();
			this.getLogger().info("Loading of "+this.imagesList.size()+" took "+(t2 - t1)+" msecs");
		}
	}
	
	public static void main(String args[])
	{
		DigitalLabelLoadingService digitalLabelLoadingService = new 
				DigitalLabelLoadingService("/Users/sameer.khan/Documents/workspace/NeuralNetworks/data/train-labels-idx1-ubyte", 
						"/Users/sameer.khan/Documents/workspace/NeuralNetworks/data/train-images-idx3-ubyte");
		try {
			digitalLabelLoadingService.loadLabels();
			digitalLabelLoadingService.loadImages();
			digitalLabelLoadingService.printRandomImage();
		} catch (Exception e) {
			StringWriter sw = new StringWriter();
			e.printStackTrace(new PrintWriter(sw));
			digitalLabelLoadingService.getLogger().error(sw.toString());	
		}
	}
}