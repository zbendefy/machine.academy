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

namespace NumberRecognize
{
    public partial class Form1 : Form
    {
        private MathLib mathLib = null;
        private Network network = null;
        Bitmap bitmap;
        private Network.TrainingPromise trainingPromise = null;
        Timer trainingtimer = new Timer();
        Form2 progressDialog = null;

        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            mathLib = new MathLib(null);

            bitmap = new Bitmap(28, 28, System.Drawing.Imaging.PixelFormat.Format16bppRgb565);
            pictureBox1.Image = bitmap;

            comboBox1.Items.Add("Use CPU calculation");
            comboBox1.SelectedIndex = 0;
            foreach (var device in ComputeDevice.GetDevices())
            {
                string item = "[" + device.GetPlatformID() + ":" + device.GetDeviceID() + ", " + device.GetDeviceType().ToString() + "] " + device.GetName();
                comboBox1.Items.Add(item);
            }

            trainingtimer.Interval = 300;
            trainingtimer.Tick += Trainingtimer_Tick;

            InitRandomNetwork();

        }

        private void Trainingtimer_Tick(object sender, EventArgs e)
        {
            if (progressDialog != null && trainingPromise != null)
            {
                progressDialog.UpdateResult(trainingPromise.GetProgress(), trainingPromise.IsReady(), "Training...");
                if (trainingPromise.IsReady())
                {
                    trainingPromise = null;
                    progressDialog = null;
                    trainingtimer.Stop();
                }
            }
        }

        private void InitRandomNetwork()
        {
            List<int> layerConfig = new List<int>();
            layerConfig.Add(bitmap.Size.Width* bitmap.Size.Height);
            layerConfig.Add(64);
            layerConfig.Add(48);
            layerConfig.Add(10);

            network = Network.CreateNetworkInitRandom(layerConfig);
        }

        private void comboBox1_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (comboBox1.SelectedIndex == 0)
                mathLib = new MathLib();
            else
                mathLib = new MathLib(ComputeDevice.GetDevices()[comboBox1.SelectedIndex - 1]);
        }

        private void button3_Click(object sender, EventArgs e)
        {
            InitRandomNetwork();
        }

        private void button6_Click(object sender, EventArgs e)
        {
            if (network != null)
            {
                saveFileDialog1.Filter = "JSON File|*.json";
                saveFileDialog1.Title = "Save training data";
                if (saveFileDialog1.ShowDialog() == DialogResult.OK)
                {
                    System.IO.File.WriteAllText(saveFileDialog1.FileName, network.GetTrainingDataJSON());
                }
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            openFileDialog1.Filter = "JSON File|*.json";
            openFileDialog1.Title = "Save training data";
            if (openFileDialog1.ShowDialog() == DialogResult.OK)
            {
                try
                {
                    string file = System.IO.File.ReadAllText(openFileDialog1.FileName);
                    var newNetwork = Network.LoadTrainingDataFromJSON(file);
                    network = newNetwork;
                }
                catch (Exception exc)
                {
                    MessageBox.Show("Error when loading network: " + exc.ToString(), "Error",MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            }
        }

        private void button4_Click(object sender, EventArgs e)
        {
            string imgFile = "";
            string labelFile = "";

            openFileDialog1.Filter = "Training data|*.*";
            openFileDialog1.Title = "Open Training images file";
            if (openFileDialog1.ShowDialog() == DialogResult.OK)
            {
                imgFile = openFileDialog1.FileName;
            }
            else
            {
                return;
            }

            openFileDialog1.Title = "Open Training labels file";
            if (openFileDialog1.ShowDialog() == DialogResult.OK)
            {
                labelFile = openFileDialog1.FileName;
            }
            else
            {
                return;
            }

            List<TrainingSuite.TrainingData> trainingData = new List<TrainingSuite.TrainingData>();

            var labelData = System.IO.File.ReadAllBytes(labelFile);
            int labelDataOffset = 8; //first 2x32 bits are not interesting for us.

            var imageData = System.IO.File.ReadAllBytes(imgFile);
            int imageDataOffset = 16; //first 4x32 bits are not interesting for us.

            for (int i = labelDataOffset; i < labelData.Length; i++)
            {
                float[] input = new float[bitmap.Size.Width * bitmap.Size.Height];
                float[] output = new float[10];
                for (int j = 0; j < bitmap.Size.Height; j++)
                {
                    for (int k = 0; k < bitmap.Size.Width; k++)
                    {
                        int offsetInImage = j * bitmap.Size.Width + k;
                        input[offsetInImage] = ((float)imageData[imageDataOffset + i + offsetInImage]) / 255.0f;
                    }
                }
                output[labelData[i]] = 1.0f;
                trainingData.Add(new TrainingSuite.TrainingData(input, output));
            }

            var trainingSuite = new TrainingSuite(trainingData);
            trainingSuite.config.miniBatchSize = 100;
            trainingSuite.config.numThreads = 1;
            trainingSuite.config.learningRate = 0.015f;
            trainingSuite.config.epochs = (int)numericUpDown1.Value;

            trainingPromise = network.Train(mathLib, trainingSuite);
            trainingtimer.Start();

            progressDialog = new Form2();
            progressDialog.ShowDialog();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            for (int i = 0; i < bitmap.Size.Height; i++)
            {
                for (int j = 0; j < bitmap.Size.Width; j++)
                {
                    bitmap.SetPixel(j, i, Color.Black);
                }
            }
            pictureBox1.Refresh();
        }

        private void pictureBox1_MouseDown(object sender, MouseEventArgs e)
        {
        }

        private void pictureBox1_MouseUp(object sender, MouseEventArgs e)
        {
        }

        private void pictureBox1_MouseMove(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Left)
            {
                float relativePosX = (float)e.X / (float)pictureBox1.Width;
                float relativePosY = (float)e.Y / (float)pictureBox1.Height;

                int bitmapX = (int)Math.Max(0,Math.Min( Math.Floor(relativePosX * (float)(bitmap.Size.Width - 1)), bitmap.Size.Width - 1));
                int bitmapY = (int)Math.Max(0, Math.Min(Math.Floor(relativePosY * (float)(bitmap.Size.Height-1)), bitmap.Size.Height- 1));
                bitmap.SetPixel(bitmapX, bitmapY, Color.White);
                pictureBox1.Refresh();
            }
        }

        private void pictureBox1_MouseLeave(object sender, EventArgs e)
        {
        }

        private void button5_Click(object sender, EventArgs e)
        {
            float[] input = new float[bitmap.Size.Width * bitmap.Size.Height];

            for (int i = 0; i < bitmap.Size.Height; i++)
            {
                for (int j = 0; j < bitmap.Size.Width; j++)
                {
                    var color = bitmap.GetPixel(j,i).ToArgb();
                    input[i * bitmap.Size.Width + j] = color == -1 ? 1.0f : 0.0f;
                }
            }

            var output = network.Compute(mathLib, input);

            float largest = 0;
            int resultIdx = 0;
            for (int i = 0; i < output.Length; i++)
            {
                if (output[i] > largest)
                {
                    largest = output[i];
                    resultIdx = i;
                }
            }

            lblResult.Text = "Results:\nI think you drew a " + resultIdx + "\nOutput was:\n";
            lblResult.Text += string.Join("\n ", output);
        }

        private void panel1_Paint(object sender, PaintEventArgs e)
        {

        }
    }
}
