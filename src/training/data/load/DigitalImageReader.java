package training.data.load;

import java.awt.image.BufferedImage;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;

import javax.imageio.ImageIO;

public class DigitalImageReader 
{
	public static void main(String args[])
	{
		try {
			BufferedImage imageIO = ImageIO.read(new FileInputStream("/Users/sameer.khan/Documents/workspace/NeuralNetworks/data/MyFile.png"));
			System.out.println("Height:"+imageIO.getHeight()+" Weight:"+imageIO.getWidth());
			double[] dArray = null;			
			System.out.println(imageIO.getData().getPixel(27, 27, dArray)[0]);
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
}
