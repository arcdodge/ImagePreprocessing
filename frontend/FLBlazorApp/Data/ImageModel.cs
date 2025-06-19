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
		public string AfterShowImg { get; protected set; } = "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";
		public string ImagePath { get; protected set; }
		public string AnnotationPath { get; protected set; }
		public string ImageFormat { get; protected set; }

		public async Task GetRegionAsync(int x, int y, int w, int h, Dictionary<string, string> preprocessingTask, bool useCuda)
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
					{ "use_cuda", useCuda.ToString().ToLower() }
				};

				foreach (var task in preprocessingTask)
				{
					values.Add(task.Key, task.Value);
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
