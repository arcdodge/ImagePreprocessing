using FLBlazorApp.ViewModel;

namespace FLBlazorApp.Data
{

	public class RecipePath
	{
		public string SelectedPath { get; set; } = "";
		public string DirectoryPath { get; set; }
	}
	public class DirectorSettingData
	{
		public DirectorExperimentData DirectorExperimentData { get; set; } = new DirectorExperimentData();
		public DirectorConnectionData DirectorConnectionData { get; set; } = new DirectorConnectionData();
		public List<mTLSData> mTLSData { get; set; } = new List<mTLSData>();
	}
	public class DirectorExperimentData
	{
		public string UUID { get; set; } = "";

		public int CollaboratorsCount { get; set; } = 1;
		public bool IsRemoteInferencing { get; set; }
		public string ExperimentName { get; set; } = "New FL experiment";
		public string SavedModelName { get; set; } = "New FL experiment";
		public string ModelArchitecture { get; set; } = "Segmentation_HRNETv2";
		public string AggregationFunction { get; set; } = "Average";
		public int Batchsize { get; set; } = 3;
		public int Epoch { get; set; } = 3;
		public bool IsConvertOpenVINO { get; set; } = false;
		public string ConvertOpenVINOPath { get; set; }
	}
	public class DirectorConnectionData
	{
		public string UUID { get; set; } = "";

		public string UserIP { get; set; } = "127.0.0.1";
		public int ListeningPort { get; set; } = 50051;
		public int InsecurePort { get; set; } = 8899;
		public bool IsUseMTLS { get; set; }
	}
	public class CollaboratorSettingData
	{
		public CollaboratorExperimentData CollaboratorExperimentData { get; set; } = new CollaboratorExperimentData();
		public CollaboratorConnectionData CollaboratorConnectionData { get; set; } = new CollaboratorConnectionData();
		public mTLSData mTLSData { get; set; } = new mTLSData();
	}
	public class CollaboratorExperimentData
	{
		public string UUID { get; set; }
		public string LoadModelLoaderSetting { get; set; }
		public string TrainingDataDirectory { get; set; }
		public string ValidationDataDirectory { get; set; }

		public string ClassNum { get; set; }
	}
	public class CollaboratorConnectionData
	{
		public string UUID { get; set; }
		public string UserName { get; set; } = "test";
		public string ConnectedIP { get; set; } = "127.0.0.1";
		public int ConnectedPort { get; set; } = 50051;
		public bool IsUseMTLS { get; set; }
	}
	public class mTLSData
	{
		public string DirectorIP { get; set; } = "";
		public string EnvoyName { get; set; } = "";
		public string CSR { get; set; } = "";
		//Crt_chain
		public string Crt_Chain { get; set; } = "";
		//Collaborator_crt
		public string Collaborator_crt { get; set; } = "";
	}
}
