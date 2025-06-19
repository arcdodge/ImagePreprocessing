namespace FLBlazorApp.Data
{
	public class StepInfo
	{
		public string StepName { get; set; }
		public string IconName { get; set; }
		public StepInfoStatus Status { get; set; } = StepInfoStatus.pending;

		public List<StepInfo> Steps { get; set; } = new List<StepInfo>();
		public void InitSteps(List<(string name, string icon)> stepList)
		{
			Steps.Clear();
			stepList.ForEach(step => { Steps.Add(new StepInfo { StepName = step.name, IconName = step.icon }); });
		}
		public void ChangeLastStepStatus(int stepCount, StepInfoStatus status)
		{
			for (int i = 0; i < Steps.Count; i++)
			{
				if (i < stepCount) Steps[i].Status = StepInfoStatus.completed;

				else if(i== stepCount) Steps[i].Status = status;
				else Steps[i].Status = StepInfoStatus.pending;
			}
		}
		public void ClearStepStatus()
		{
			Steps.ForEach(_ => _.Status = StepInfoStatus.pending);
		}
	}
	public enum StepInfoStatus
	{
		/// <summary>
		/// 未開始
		/// </summary>
		pending,

		/// <summary>
		/// 進行中
		/// </summary>
		loading,

		/// <summary>
		/// 已完成
		/// </summary>
		completed,

		/// <summary>
		/// 因錯誤停止
		/// </summary>
		error,

        /// <summary>
		/// 使用者停止
		/// </summary>
		stop
    }
}
