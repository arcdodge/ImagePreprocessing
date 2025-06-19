using BootstrapBlazor.Components;
using FLBlazorApp.Component;
using FLBlazorApp.Data;
using FLBlazorApp.Pages;
using Microsoft.AspNetCore.Components.Forms;
using Microsoft.Extensions.FileSystemGlobbing.Internal;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Diagnostics;
using System.Diagnostics.Metrics;
using System.Reflection.Emit;
using System.Text.RegularExpressions;
using System.Xml.Linq;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace FLBlazorApp.ViewModel
{
	public class RelativePath
	{
		public string Value { get; set; } = "";
		public string RelativeTo { get; set; }
		public string Absolute
		{
			get
			{
				return Path.Combine(RelativeTo, Value);
			}
			set
			{
				if (string.IsNullOrEmpty(value))
				{
					Value = RelativeTo;
				}
				Value = Path.GetRelativePath(RelativeTo, value);
			}
		}
		public RelativePath(string RelativeTo)
		{
			this.RelativeTo = RelativeTo;
		}
	}
}
