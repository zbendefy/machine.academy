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
            network = CreateNetworkFromConfig(txtLayerConf.Text);

            var deviceList = Macademy.OpenCL.ComputeDevice.GetDevices();
            cmbComputeDevice.Items.Add("CPU Fallback Device");
            foreach (var device in deviceList)
            {
                string item = "[" + device.GetPlatformID() + ":" + device.GetDeviceID() + ", " + device.GetDeviceType().ToString() + "] " + device.GetName().Trim() + " " + (device.GetGlobalMemorySize() / (1024 * 1024)) + "MB";
                cmbComputeDevice.Items.Add(item);
            }
            cmbComputeDevice.SelectedIndex = 0;
        }

        private static string OpenFileWithDialog(string title)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Title = title;
            openFileDialog.Filter = "*.csv|*.csv";
            if (openFileDialog.ShowDialog() == true)
            {
                var content = File.ReadAllText(openFileDialog.FileName);
                return content;
            }
            return null;
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

            var trainingFileContent = OpenFileWithDialog("Open news training data");

            var trainingData = new List<TrainingSuite.TrainingData>();

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

        private void Button_Click_1(object sender, RoutedEventArgs e)
        {
            if (network == null)
                return;

            var testFileContent = OpenFileWithDialog("Open news test data");

            var testInputs = testFileContent.Split(";newline_marker\r\n").Where(article => article.Length > 3).Select( article => {
                var ret = article; 
                if (article.StartsWith('\"')) { ret = ret.Substring(1); }
                if (article.EndsWith('\"')) { ret = ret.Substring(0, ret.Length-1); }
                return CleanText(ret);
            }).ToArray();

            var calculator = GetCalculator(cmbComputeDevice.SelectedIndex);

            foreach (var item in testInputs)
            {
                var result = network.Compute(GenerateInputFromCleanedArticle(item,network.GetLayerConfig()[0]), calculator);

                int currMaxId = 0;
                float currMaxValue = 0;
                for (int i = 0; i < result.Length; i++)
                {
                    if (result[i] > currMaxValue)
                    {
                        currMaxValue = result[i];
                        currMaxId = i;
                    }
                }
            }

            Console.WriteLine("alma");
        }

        private static float[] GenerateInputFromCleanedArticle(string cleanedArticle, int maxInput) 
        {
            var numValidCharsDivisor = (float)(validCharacters.Length - 1);
            var ret = new float[maxInput];
            float spacePos = ((float)validCharacters.IndexOf(' ')) / numValidCharsDivisor;
            for (int i = 0; i < maxInput ; i++)
            {
                if (i >= cleanedArticle.Length)
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



            lblNewsFragmentResult.Content = "not yet implemented";
        }

        private void Button_Click_3(object sender, RoutedEventArgs e)
        {
            

        }
    }
}
