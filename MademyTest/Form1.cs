using Mademy;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace MademyTest
{
    public partial class Form1 : Form
    {
        Network solver = null;
        MathLib mathLib = null;
        Network.TrainingPromise trainingPromise = null;
        DateTime trainingBegin;
        List<TrainingSuite.TrainingData> trainingData = new List<TrainingSuite.TrainingData>();

        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            List<int> layerConfig = new List<int>();
            layerConfig.Add(5);
            layerConfig.Add(14);
            layerConfig.Add(27);
            layerConfig.Add(5);

            solver = Network.CreateNetworkInitRandom(layerConfig, new SigmoidActivation(), new DefaultWeightInitializer());

            mathLib = new MathLib( null );

            comboBox1.Items.Add("Use CPU calculation");
            comboBox1.SelectedIndex = 0;
            foreach (var device in ComputeDevice.GetDevices())
            {
                string item = "[" + device.GetPlatformID() + ":" + device.GetDeviceID() + ", " + device.GetDeviceType().ToString() + "] " + device.GetName();
                comboBox1.Items.Add(item);
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            List<float> input = new List<float>();
            input.Add((float)numericUpDown1.Value);
            input.Add((float)numericUpDown2.Value);
            input.Add((float)numericUpDown3.Value);
            input.Add((float)numericUpDown4.Value);
            input.Add((float)numericUpDown5.Value);

            var result = solver.Compute(mathLib, input.ToArray());

            label1.Text = ("Result of: " + string.Join(", ", input) + " is: \n" + string.Join("\n", result));
        }

        private void button2_Click(object sender, EventArgs e)
        {
            textBox1.Text = solver.GetTrainingDataJSON();
        }

        private void button3_Click(object sender, EventArgs e)
        {
            solver = Network.LoadTrainingDataFromJSON(textBox1.Text);
        }

        private void button4_Click(object sender, EventArgs e)
        {
            if (trainingData.Count == 0)
            {
                var rnd = new Random();
                for (int i = 0; i < 10000; i++)
                {
                    float[] input = new float[5];
                    float[] output = new float[5];

                    float j = (float)rnd.NextDouble() * 5;
                    int jint = Math.Min(4, (int)Math.Floor(j));
                    input[jint] = 1.0f;
                    output[jint] = 1.0f;

                    trainingData.Add(new TrainingSuite.TrainingData(input, output));
                }
            }

            var trainingSuite = new TrainingSuite(trainingData);
            trainingSuite.config.miniBatchSize = 10;
            trainingSuite.config.epochs = 2;
            trainingSuite.config.shuffleTrainingData = false;

            trainingSuite.config.regularization = TrainingSuite.TrainingConfig.Regularization.None;
            trainingSuite.config.costFunction = new CrossEntropy();

            trainingPromise = solver.Train(mathLib, trainingSuite);

            progressBar1.Value = 0;
            label2.Text = "Training...";
            trainingBegin = DateTime.Now;
            timer1.Start();
        }

        private void comboBox1_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (comboBox1.SelectedIndex == 0)
                mathLib = new MathLib();
            else
                mathLib = new MathLib( ComputeDevice.GetDevices()[comboBox1.SelectedIndex - 1] );
        }

        private void label1_Click(object sender, EventArgs e)
        {

        }

        private void timer1_Tick(object sender, EventArgs e)
        {
            if (trainingPromise != null)
            {
                progressBar1.Value = (int)(trainingPromise.GetTotalProgress() * 100.0f);
                if (trainingPromise.IsReady())
                {
                    var period = DateTime.Now.Subtract(trainingBegin);
                    label2.Text = "Training done in " + period.TotalSeconds+"s";



                    timer1.Stop();
                }
            }
            else
            {
                timer1.Stop();
            }
               
        }
    }
}
