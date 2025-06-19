namespace FLBlazorApp.Data
{
    public class ChartDataClass
    {
        public string XAxisName { get; set; } = "Round";
        public string YAxisName { get; set; }
        public Dictionary<string, List<object>> Datas { get; set; } = new Dictionary<string, List<object>>();
        public List<string> XLabels {get;set; } = new List<string>();
    }
}
