package wireless;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;

public class ConnectionManager {
	static Connection conn;

	public static Connection getConnection() {
		String url = "jdbc:mysql://localhost:3306/";
		String dbName = "wireless";
		String uname = "root";
		String pwd = "123456";
		try {
			if (conn != null&& conn.isClosed()) {
				conn = DriverManager.getConnection(url + dbName, uname, pwd);
				return conn;
			}
			if (conn != null)
				return conn;
			Class.forName("com.mysql.jdbc.Driver");
			conn = DriverManager.getConnection(url + dbName, uname, pwd);

		} catch (ClassNotFoundException e) {
			System.out.println(e);
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return conn;
	}

	public static void closeConnection() {
		try {
			// Check of connection is null or not
			if (conn != null) {
				// If connection is not null, then commit the changes and cloase
				// the connection

				conn.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
