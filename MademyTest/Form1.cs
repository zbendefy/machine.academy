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
            layerConfig.Add(2);
            layerConfig.Add(2);
            layerConfig.Add(4);
            solver = Network.CreateNetworkInitRandom(layerConfig);


            foreach (var device in CLMath.ComputeDevice.GetDevices())
            {
                comboBox1.Items.Add(device);
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            List<float> input = new List<float>();
            input.Add(1);
            input.Add(-4);
            input.Add(3);
            input.Add(2);
            input.Add(5);

            if ( mathLib == null)
                mathLib = new MathLib((ComputeDevice)comboBox1.SelectedItem);

            var result = solver.Compute(mathLib, input.ToArray());

            label1.Text = ("Result of: " + string.Join(",", input) + " is: " + string.Join(",", result));
        }

        private void button2_Click(object sender, EventArgs e)
        {
            textBox1.Text = solver.GetTrainingDataJSON();
        }

        private void button3_Click(object sender, EventArgs e)
        {
            solver = Network.LoadTrainingDataFromJSON(textBox1.Text);
        }
    }
}
