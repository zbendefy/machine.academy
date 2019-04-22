using CLMath;
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

        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            List<int> layerConfig = new List<int>();
            layerConfig.Add(5);
            layerConfig.Add(16);
            layerConfig.Add(20);
            layerConfig.Add(5);
            solver = Network.CreateNetworkInitRandom(layerConfig);

            try
            {
                mathLib = new MathLib( ComputeDevice.GetDevices()[0]);
            }
            catch (Exception)
            {
                mathLib = new MathLib();
                throw;
            }

            foreach (var device in ComputeDevice.GetDevices())
            {
                comboBox1.Items.Add(device);
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
            var config = Network.TrainingConfig.CreateTrainingConfig();
            config.miniBatchSize = 10;
            List<Tuple<float[], float[]>> trainingData = new List<Tuple<float[], float[]>>();

            var rnd = new Random();
            for (int i = 0; i < 10000000; i++)
            {
                float[] input = new float[] { 0,0,0,0,0 };
                float[] output = new float[] { 0,0,0,0,0 };

                float j = (float)rnd.NextDouble() * 5;
                int jint = Math.Min(4, (int)Math.Floor(j));
                input[jint] = 1.0f;
                output[jint] = 1.0f;

                trainingData.Add(new Tuple<float[], float[]>(input, output));
            }


            solver.Train(mathLib, trainingData, config);
        }

        private void comboBox1_SelectedIndexChanged(object sender, EventArgs e)
        {

            mathLib = new MathLib((ComputeDevice)comboBox1.SelectedItem);
        }

        private void label1_Click(object sender, EventArgs e)
        {

        }
    }
}
