package activation;

public class Functions 
{
	public static double sigmoid(double x) 
	{
	    return (1/( 1 + Math.pow(Math.E,(-1*x))));
	}
	
	//x = output of the sigmoid function
	public static double sigmoidD(double x)
	{
		return x * (1.0 - x);
	}
}
