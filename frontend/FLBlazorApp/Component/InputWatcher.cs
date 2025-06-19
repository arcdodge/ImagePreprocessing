using Microsoft.AspNetCore.Components.Forms;
using Microsoft.AspNetCore.Components;

namespace FLBlazorApp.Component
{
	public class InputWatcher : ComponentBase
	{
		private EditContext editContext;

		[CascadingParameter]
		protected EditContext EditContext
		{
			get => editContext;
			set
			{
				editContext = value;
				EditContextActionChanged?.Invoke(editContext);
			}
		}

		[Parameter]
		public Action<EditContext> EditContextActionChanged { get; set; }
	}
}
