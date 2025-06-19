using BootstrapBlazor.Components;
using BruTile.Wms;
using Newtonsoft.Json;
using OpenCvSharp;
using SixLabors.ImageSharp.Formats.Jpeg;
using System.Drawing;
using System.Drawing.Imaging;
using System.Reflection;
using System.Runtime.CompilerServices;
using static System.Net.Mime.MediaTypeNames;
using static System.Runtime.InteropServices.JavaScript.JSType;
using Image = SixLabors.ImageSharp.Image;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.Formats.Gif;
using System.Text;
using Newtonsoft.Json.Linq;
using System.ComponentModel;

namespace FLBlazorApp.Data
{

	public class ImageLoader : INotifyPropertyChanged
	{
		public event PropertyChangedEventHandler? PropertyChanged;
		public readonly int thumbnailMaxPixel = 4096;
		public int ThumbnailImgWidth { get; protected set; }
		public int ThumbnailImgHeight { get; protected set; }
		public int OriImageWidth { get; protected set; }
		public int OriImageHeight { get; protected set; }
		public string Base64EncodedThumbnail { get; protected set; }
		public string AfterImage { get; protected set; }
		public string AfterMask { get; protected set; }
		public string AfterShowImg { get; protected set; } = "";
		public string ImagePath { get; protected set; }
		public string AnnotationPath { get; protected set; }
		public string ImageFormat { get; protected set; }

		public async Task GetRegionAsync(int x, int y, int w, int h, Dictionary<string, string> preprocessingTask, string accelerationMethod)
		{
			using (var client = new HttpClient())
			{
				var values = new Dictionary<string, string>
				{
					{ "path", ImagePath },
					{ "maskPath",AnnotationPath },
					{ "format", ImageFormat },
		 				{ "x", x.ToString() },
		 				{ "y", y.ToString() },
		 				{ "w", w.ToString() },
		 				{ "h", h.ToString() },
					{ "acceleration", accelerationMethod }
				};

				foreach (var task in preprocessingTask)
				{
					values[task.Key] = task.Value;
				}
				//foreach (var item in values)
				//{

				//	System.Console.WriteLine($"!!!!![DEBUG]*****{item.Key} : {item.Value}*****!!!!!");
				//}

				var content = new FormUrlEncodedContent(values);

				var response = await client.PostAsync(FLAPIConfig.ImgServer_HOST + "getRegion/", content);

				if (response.IsSuccessStatusCode)
				{
					var responseContent = await response.Content.ReadAsStringAsync();

					//System.Console.WriteLine($"!!!!![DEBUG]responseContent={responseContent}");
					try
					{

					var obj = JsonConvert.DeserializeObject<ImageBase64Str>(responseContent);
					AfterImage = obj.ImageRegion;
					AfterMask = obj.MaskRegion;
					AfterShowImg = obj.IntegrationRegion;
					}
					catch (System.Exception ex)
					{
                        System.Console.WriteLine($"!!!!![DEBUG]**ERRORRRR***{ex.Message}*****!!!!!");
					}
				}
				else
				{
					throw new System.Exception("回應狀態碼: " + response.StatusCode);
				}
			}
		}

		public async Task LoadImageAsync(string imgPath, int thumbnailMaxPixel, string annotationPath = "")
		{
			try
			{
				ImagePath = imgPath;
				ImageFormat = System.IO.Path.GetExtension(ImagePath);
				AnnotationPath = annotationPath;

				using (var client = new HttpClient())
				{
					var values = new Dictionary<string, string>
				{
					{ "path", ImagePath },
					{ "maskPath",AnnotationPath },
					{ "format", ImageFormat },
					{ "maxPixel", thumbnailMaxPixel.ToString() },
				};

					var content = new FormUrlEncodedContent(values);
					System.Console.WriteLine($"呼叫 API: {FLAPIConfig.ImgServer_HOST + "loadImage/"}"); // 在執行請求前輸出 URL
					var response = await client.PostAsync(FLAPIConfig.ImgServer_HOST + "loadImage/", content);

					if (response.IsSuccessStatusCode)
					{
						var responseContent = await response.Content.ReadAsStringAsync();

						var obj = JsonConvert.DeserializeObject<ImageInfo>(responseContent);
						OriImageWidth = obj.originW;
						OriImageHeight = obj.originH;
						ThumbnailImgWidth = obj.ThumbnailW;
						ThumbnailImgHeight = obj.ThumbnailH;
						Base64EncodedThumbnail = obj.ThumbnailPNGImg;
					}
					else
					{
						throw new System.Exception("回應狀態碼: " + response.StatusCode);
					}
				}
			}
			catch (System.Exception ex)
			{

				throw ex;
			}

		}
		        
		public void SavePreprocessedImg(string saveDir, string fileName, string PrefixStr, int PrefixNo, string saveType)
        		{
            		switch (ImageFormat)
            		{
                			case ".dcm":
                    		byte[] saveMask = Convert.FromBase64String(AfterMask);

							string maskDir = Path.Combine(saveDir,"mask");
							if (!Path.Exists(maskDir))
								System.IO.Directory.CreateDirectory(maskDir);
				
                    		string saveMaskFullPath = Path.Combine(maskDir, $"{fileName}_{PrefixStr}_{PrefixNo}.{saveType}");

                    		using (FileStream fs = new FileStream(saveMaskFullPath, FileMode.Create))
                    		{
                        			fs.Write(saveMask, 0, saveMask.Length);
                    		}
                    		break;
            		}

            		byte[] saveImg = Convert.FromBase64String(AfterImage);
					string imageDir = Path.Combine(saveDir,"image");
            		string saveImgFullPath = Path.Combine(imageDir, $"{fileName}_{PrefixStr}_{PrefixNo}.{saveType}");
					System.Console.WriteLine($"**********[Img_Saved] SavePath = {saveImgFullPath}**********");
            		using (FileStream fs = new FileStream(saveImgFullPath, FileMode.Create))
            		{
                			fs.Write(saveImg, 0, saveImg.Length);
            		}
        		}
	}
	public class ImageBase64Str
	{
		public string ImageRegion { get; set; }
		public string? MaskRegion { get; set; }
		public string IntegrationRegion { get; set; }
	}
	public class ImageInfo
	{
		public int originW { get; set; }
		public int originH { get; set; }
		public int ThumbnailW { get; set; }
		public int ThumbnailH { get; set; }
		public string ThumbnailPNGImg { get; set; }
	}
}
