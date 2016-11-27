package training.data.load;

import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.File;
import javax.imageio.ImageIO;

public class DigitalImageWriter 
{
	public static void main(String args[])
	{
		try{
			BufferedImage img = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
			WritableRaster raster = img.getRaster(); 
			for(int a = 0 ; a < img.getHeight() ; a++)
			{
				for(int b = 0 ; b < img.getWidth(); b++)
				{
					int value = 127+(int)(128*Math.sin(b/32.)*Math.sin(a/32.)); // Weird sin pattern. 
					raster.setSample(a, b, 0, value);
					System.out.println("Setting value:"+value);
				}
			}
			File f = new File("/Users/sameer.khan/Documents/workspace/NeuralNetworks/5.png");
			ImageIO.write(img, "PNG", f);
			img.getData();
		}catch(Exception e)
		{
			e.printStackTrace();
		}		
	}
}
