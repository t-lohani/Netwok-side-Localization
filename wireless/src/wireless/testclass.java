package wireless;

//import org.json.JSONObject;

public class testclass 
{
	public static void main(String[] args)
	{
		insertDataToDB();
	}
	public static void insertDataToDB()
	{
		DAO.insertData("40.84,-73.87,-65");
	}
}
