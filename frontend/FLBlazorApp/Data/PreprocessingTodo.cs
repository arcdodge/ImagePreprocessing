namespace FLBlazorApp.Data
{
    public class PreprocessingTodo
    {
        public FunctionType FunctionType { get; set; }
        public PreprocessingValueType TodoType { get; set; }
        public bool IsEnable { get; set; }

        public List<PreprocessingChose> Comboboxdata { get; set; }
        public List<PreprocessingSlider> Sliderdata { get; set; }
        public List<PreprocessingTextbox> Textboxdata { get; set; }
    }
    public enum PreprocessingValueType
    {
        Combobox,
        Slider,
        Textbox
    }
    public class PreprocessingChose
    {
        public string ValueTitle { get; set; }
        public List<string> ChoseItemList { get; set; }
        public string SelectedItem { get; set; }
    }
    public class PreprocessingSlider
    {
        public string ValueTitle { get; set; }
        public decimal Max { get; set; }
        public decimal Min { get; set; }
        public string Step { get; set; }
        public double Value { get; set; }
    }
    public class PreprocessingTextbox
    {
        public string ValueTitle { get; set; }
        public string Text { get; set; }
    }
    public enum FunctionType
    {
        TransformationDegrees,
        TransformationFlip,
        GammaCorrection,
        Guassian,
        Blur,
        BilateralFilter,
        Canny,
        WarpAffine
    }
}
