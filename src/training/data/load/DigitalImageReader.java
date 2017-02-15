package training.data.load;

import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.File;
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
			int minValue=0, maxValue=0;
			BufferedImage imageIO = ImageIO.read(new FileInputStream("/Users/sameer.khan/Documents/workspace/OCR/parsed_images_output"
					+ "/Sameer_HandWriting_2.jpg/char_0.png"));
			System.out.println("Height:"+imageIO.getHeight()+" Weight:"+imageIO.getWidth());
			double[] dArray = null;
			System.out.println(imageIO.getData().getPixel(27, 27, dArray)[0]);
			
			BufferedImage bufferedImage = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
			WritableRaster raster = bufferedImage.getRaster();
			for(int a = 0 ; a < bufferedImage.getHeight() ; a++)
			{
				for(int b = 0 ; b < bufferedImage.getWidth(); b++)
				{
					int value = (int) imageIO.getData().getPixel(a, b, dArray)[0];
					//System.out.println(value);
					raster.setSample(a, b, 0, value);
					if(minValue > value)
					{
						minValue = value;
					}
					
					if(maxValue < value)
					{
						maxValue = value;
					}
				}
			}
			File f = new File(System.getProperty("user.dir")+"/"+"sameer"+".png");
			ImageIO.write(bufferedImage, "PNG", f);
			System.out.println("minVal:"+minValue+","+"maxVal:"+maxValue);
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
}
