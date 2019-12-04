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
