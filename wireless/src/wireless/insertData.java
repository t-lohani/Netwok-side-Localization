package wireless;

import java.sql.Connection;
import java.sql.SQLException;
import java.sql.Statement;

import javax.ws.rs.Consumes;
import javax.ws.rs.GET;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.QueryParam;
import javax.ws.rs.core.MediaType;

import org.json.JSONObject;

@Path("/test")
public class insertData {
	@GET
	@Consumes(MediaType.APPLICATION_JSON)
	@Produces(MediaType.TEXT_PLAIN)
	@Path("/getContacts")
    public String getContacts(@QueryParam("json_obj") JSONObject json_obj)
    {
	    if(json_obj != null)
	    {
	    	System.out.println("OK");
	    }
	    else
	    {
	    	System.out.println("Not ok");
	    }
	    return "Domething";
    }
	@GET
	@Consumes(MediaType.APPLICATION_JSON)
	@Produces(MediaType.TEXT_PLAIN)
	@Path("/testServ")
    public String testserv(@QueryParam("json_obj") String json_obj)
    {
		DAO.insertData(json_obj);
		return "success";
    }	
}
class DAO
{
	static Connection currentCon = null;
	
	public static void insertData(String s)
	{
		try 
		{
			currentCon = ConnectionManager.getConnection();
			Statement stmt = null;
			String[] d = s.split(",");
			stmt = currentCon.createStatement();
			String sql_query = "INSERT INTO LocalizationData values(" + Double.parseDouble(d[0]) + "," + Double.parseDouble(d[1]) +"," + Double.parseDouble(d[2])+")";
			stmt.executeUpdate(sql_query);			
		}
		catch (SQLException e) 
		{
			e.printStackTrace();
		}
	}
}
class data
{
	public double lon;
	public double lat;
	public double rssi;
	public data(double l_p, double lat_p, double r_p)
	{
		this.lon = l_p;
		this.lat = lat_p;
		this.rssi = r_p;
	}
	public data(){}
}