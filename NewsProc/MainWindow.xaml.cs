using Macademy;
using Microsoft.Win32;
using NewsAnalyzer;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace NewsProc
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private Network network = null;

        public MainWindow()
        {
            InitializeComponent();
            RecreateNetwork();

            var deviceList = Macademy.OpenCL.ComputeDevice.GetDevices();
            cmbComputeDevice.Items.Add("CPU Fallback Device");
            foreach (var device in deviceList)
            {
                string item = "[" + device.GetPlatformID() + ":" + device.GetDeviceID() + ", " + device.GetDeviceType().ToString() + "] " + device.GetName().Trim() + " " + (device.GetGlobalMemorySize() / (1024 * 1024)) + "MB";
                cmbComputeDevice.Items.Add(item);
            }
            cmbComputeDevice.SelectedIndex = 0;
        }

        private static string OpenFileWithDialog(string title, string filter)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Title = title;
            openFileDialog.Filter = filter;
            if (openFileDialog.ShowDialog() == true)
            {
                var content = File.ReadAllText(openFileDialog.FileName);
                return content;
            }
            return null;
        }


        private static void SaveFileWithDialog(string title, string content, string filter)
        {
            SaveFileDialog saveFileDialog = new SaveFileDialog();
            saveFileDialog.Title = title;
            saveFileDialog.Filter = filter;
            saveFileDialog.AddExtension = true;
            saveFileDialog.CheckFileExists = false;
            saveFileDialog.CreatePrompt = true;

            if (saveFileDialog.ShowDialog() == true)
            {
                File.WriteAllText(saveFileDialog.FileName, content);
            }
        }

        private static Network CreateNetworkFromConfig(string config)
        {
            var layers = config.Split('x').Select(num => int.Parse(num)).ToArray();
            return Network.CreateNetworkInitRandom(layers, new SigmoidActivation());
        }

        private static Calculator GetCalculator(int deviceId)
        {
            if (deviceId == 0)
                return null;
            return new Calculator(Macademy.OpenCL.ComputeDevice.GetDevices()[deviceId - 1]);
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            if (network == null)
                return;

            var trainingFileContent = OpenFileWithDialog("Open news training data", "*.csv|*.csv");

            if (trainingFileContent == null)
                return;

            var trainingData = new List<TrainingSuite.TrainingData>();

            FillTrainingDataFromFile(ref trainingData, trainingFileContent, network.GetInputSize());

            var trainingSuite = new TrainingSuite(trainingData);
            trainingSuite.config.learningRate = float.Parse(txtLearningRate.Text);
            trainingSuite.config.epochs = int.Parse(txtEpochs.Text);
            trainingSuite.config.miniBatchSize = int.Parse(txtMiniBatches.Text);
            trainingSuite.config.regularizationLambda = float.Parse(txtRegularizationLambda.Text);
            trainingSuite.config.shuffleTrainingData = chkShuffleMinibatches.IsChecked == true;

            switch (cmbCostFunction.SelectedIndex)
            {
                case 1:
                    trainingSuite.config.costFunction = new MeanSquaredErrorFunction();
                    break;
                case 0:
                default:
                    trainingSuite.config.costFunction = new CrossEntropyErrorFunction();
                    break;
            }

            switch (cmbRegularization.SelectedIndex)
            {
                case 0:
                    trainingSuite.config.regularization = TrainingSuite.TrainingConfig.Regularization.None;
                    break;
                case 1:
                    trainingSuite.config.regularization = TrainingSuite.TrainingConfig.Regularization.L1;
                    break;
                case 2:
                default:
                    trainingSuite.config.regularization = TrainingSuite.TrainingConfig.Regularization.L2;
                    break;
            }

            var calculator = GetCalculator(cmbComputeDevice.SelectedIndex);

            var trainingProcess = network.Train(trainingSuite, calculator);

            var dialogWnd = new TrainingDialog(trainingProcess);

            dialogWnd.ShowDialog();
        }

        private static void FillTrainingDataFromFile(ref List<TrainingSuite.TrainingData> trainingData, string trainingFileContent, int networkMaxInput)
        {
            var records = trainingFileContent.Split(";__mrk2__\r\n");

            foreach (var item in records)
            {
                if (item.Length < 4) 
                    continue;
                var sample = item.Split(";__mrk1__;");
                string cleanedText = CleanText(sample[0]);
                int desiredOutputId = int.Parse(sample[1]);

                float[] desiredOutput = new float[] { 0,0,0,0 };
                desiredOutput[desiredOutputId] = 1;

                trainingData.Add(new TrainingSuite.TrainingData(GenerateInputFromCleanedArticle(cleanedText, networkMaxInput), desiredOutput));
            }
        }

        private void Button_Click_1(object sender, RoutedEventArgs e)
        {
            if (network == null)
                return;

            var testFileContent = OpenFileWithDialog("Open news test data", "*.csv|*.csv");

            if (testFileContent == null)
                return;

            var testInputs = testFileContent.Split(";newline_marker\r\n").Where(article => article.Length > 3).Select( article => {
                return CleanText(article);
            }).ToArray();

            //int maxLength = testInputs.Select(s => s.Length).Max();

            var calculator = GetCalculator(cmbComputeDevice.SelectedIndex);

            string fileOut = "";

            foreach (var item in testInputs)
            {
                var output = network.Compute(GenerateInputFromCleanedArticle(item, network.GetInputSize()), calculator);
                var category = EvaluateResult(output);
                fileOut += category.ToString() + "\r\n";
            }

            SaveFileWithDialog("Save results", fileOut, "*.csv|*.csv|*.txt|*.txt");
        }

        private static int EvaluateResult(float[] output)
        {
            int currMaxId = 0;
            float currMaxValue = 0;
            for (int i = 0; i < output.Length; i++)
            {
                if (output[i] > currMaxValue)
                {
                    currMaxValue = output[i];
                    currMaxId = i;
                }
            }
            return currMaxId;
        }

        private static float[] GenerateInputFromCleanedArticle(string cleanedArticle, int maxInput) 
        {
            var numValidCharsDivisor = (float)(validCharacters.Length - 1);
            var ret = new float[maxInput];
            float spacePos = ((float)validCharacters.IndexOf(' ')) / numValidCharsDivisor;
            for (int i = 0; i < maxInput ; i++)
            {
                if (i < cleanedArticle.Length)
                    ret[i] = ((float)validCharacters.IndexOf(cleanedArticle[i])) / numValidCharsDivisor;
                else
                    ret[i] = spacePos;
            }
            return ret;
        }

        private static string validCharacters = " abcdefghijklmnopqrstuvwxyz,;-.?!%#&$()0123456789'\"\n";

        private static string CleanText(string text)
        {
            var ret = "";

            if (text.StartsWith('\"')) { text = text.Substring(1); }
            if (text.EndsWith('\"')) { text = text.Substring(0, text.Length - 1); }

            text = text.ToLower();
            for(int i = 0; i < text.Length; ++i)
            {
                if (validCharacters.Contains(text[i]))
                {
                    ret += text[i];
                }
            }

            return ret;
        }

        private void Button_Click_2(object sender, RoutedEventArgs e)
        {
            if (network == null)
                return;

            //Test news fragment
            var newsFragment = txtNewsFragment.Text;

            var output = network.Compute(GenerateInputFromCleanedArticle(newsFragment, network.GetInputSize()), GetCalculator(cmbComputeDevice.SelectedIndex));

            int resultId = EvaluateResult(output);

            lblNewsFragmentResult.Content = "I think that's a " + ResultIdToString(resultId) + " (" + resultId + ")";
        }

        private static string ResultIdToString(int id)
        {
            switch (id)
            {
                case 0:
                    return "Politics";
                case 1:
                    return "Technology";
                case 2:
                    return "Entertainment";
                case 3:
                    return "Business";
                default:
                    return "unknown";
            }
        }

        private void Button_Click_3(object sender, RoutedEventArgs e)
        {
            RecreateNetwork();
        }

        private void RecreateNetwork()
        {
            try
            {
                network = CreateNetworkFromConfig(txtLayerConf.Text);
                lblActiveNetworkConfig.Content = txtLayerConf.Text;

            }
            catch (Exception)
            {
                network = null;
                lblActiveNetworkConfig.Content = "no-active-network";
            }
        }

        private void Button_Click_4(object sender, RoutedEventArgs e)
        {
            if (network == null)
                return;

            var json = network.ExportToJSON();
            SaveFileWithDialog("Save network to JSON", json, "JSON (*.json)|*.json");
        }

        private void Button_Click_5(object sender, RoutedEventArgs e)
        {
            string json = OpenFileWithDialog("Open network from JSON", "JSON (*.json)|*.json");
            if (json == null)
                return;

            network = Network.CreateNetworkFromJSON(json);
            lblActiveNetworkConfig.Content = string.Join('x', network.GetLayerConfig());
        }
    }
}
