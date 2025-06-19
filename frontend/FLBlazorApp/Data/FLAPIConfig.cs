using DnsClient;
using Newtonsoft.Json;
using System.Net.NetworkInformation;
using System.Net.Sockets;
using System.Net;
using System.Runtime.InteropServices;
using System.Text;
using System.Linq;

namespace FLBlazorApp.Data
{
	public class FLAPIConfig
	{
		public static bool IsWindows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
		public static string ProjectRoot = IsWindows ? Directory.GetParent(Directory.GetParent(Directory.GetCurrentDirectory()).FullName).FullName : Directory.GetCurrentDirectory();
		public static string ReleaseVersion { get; set; } = "";

		public static string API_HOST { get; set; } = IsWindows ? "http://localhost:5279/" : "http://api:7166/";
		public static string ImgServer_HOST { get; set; } = IsWindows ? "http://localhost:64656/image/jellox/openslide/" : "http://192.168.88.83:64656/image/jellox/openslide/";
		public static string FL_Tools_Path { get; set; } = IsWindows ? Path.Combine(ProjectRoot, "Federated-Tools") : @"/Federated-Tools";
		public static string DirectorConnectionRecipePath { get; set; } = IsWindows ? Path.Combine(ProjectRoot, "Configuration", "Director", "Connection") : @"/mnt/Configuration/Director/Connection/";
		public static string DirectorExperimentRecipePath { get; set; } = IsWindows ? Path.Combine(ProjectRoot, "Configuration", "Director", "Experiment") : @"/mnt/Configuration/Director/Experiment/";
		public static string EnvoyConnectionRecipePath { get; set; } = IsWindows ? Path.Combine(ProjectRoot, "Configuration", "Envoy", "Connection") : @"/mnt/Configuration/Envoy/Connection/";
		public static string EnvoyExperimentRecipePath { get; set; } = IsWindows ? Path.Combine(ProjectRoot, "Configuration", "Envoy", "Experiment") : @"/mnt/Configuration/Envoy/Experiment/";
		public static string SlidesMountPath { get; set; } = IsWindows ? Path.Combine(ProjectRoot, "slides") : @"/home/riley/workspace/courses/555005_PP/final_project/last/slides/";
	}

	public static class Recipe
	{
		public static void ReflectionFromData<T>(ref T recipe, T data) where T : new()
		{
			Type dataType = typeof(T);

			foreach (var property in dataType.GetProperties())
			{
				var value = property.GetValue(data);
				dataType.GetProperty(property.Name).SetValue(recipe, value);
			}
		}
	}

	public class IP
	{
		public enum Type
		{
			Normal,
			Public,
		}
		public string Value { get; set; }
		public Type type { get; set; } = Type.Normal;
		public string Hostname { get; set; }
		public string DisplayText
		{
			get
			{
				List<string> values = new List<string> { Value };
				string? hostname = Hostname;
				if (hostname != null)
				{
					values.Add($"[{hostname}]");
				}
				switch (type)
				{
					case Type.Public:
						values.Add($"[{type}]");
						break;
				}
				return string.Join(' ', values);
			}
		}
		public static IP Parse(IPAddress ip)
		{
			return new IP { Value = ip.ToString() };
		}
		public static string? GetHostname(string ip)
		{
			try
			{
				var dns = new DnsClient.LookupClient();
				dns.Timeout = TimeSpan.FromMilliseconds(50);
				var dnsResult = dns.GetHostEntry(ip);
				if (dnsResult != null)
				{
					return dnsResult.HostName;
				}
			}
			catch
			{
				;
			}

			return null;
		}
	}
	
}
