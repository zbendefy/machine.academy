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
            comboRegularization.SelectedIndex = 2;
            comboCostFunction.SelectedIndex = 1;

            mathLib = new MathLib(null);

            bitmap = new Bitmap(28, 28, System.Drawing.Imaging.PixelFormat.Format16bppRgb565);
            ClearBitmap();
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
                progressDialog.UpdateResult(trainingPromise.GetTotalProgress(), trainingPromise.IsReady(), "Training... Epochs done: " + trainingPromise.GetEpochsDone());
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
            layerConfig.Add(bitmap.Size.Width * bitmap.Size.Height);
            layerConfig.Add(512);
            layerConfig.Add(512);
            layerConfig.Add(512);
            layerConfig.Add(10);

            network = Network.CreateNetworkInitRandom(layerConfig, new SigmoidActivation(), new DefaultWeightInitializer());
            network.AttachName("MNIST learning DNN");
            network.AttachDescription("MNIST learning DNN using " + layerConfig.Count + " layers in structure: (" + string.Join(", ", layerConfig) + " ). Creation date: " + DateTime.Now.ToString() );
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

        private void LoadTestDataFromFiles(List<TrainingSuite.TrainingData> trainingData, String labelFileName, String imgFileName)
        {
            var labelData = System.IO.File.ReadAllBytes(labelFileName);
            int labelDataOffset = 8; //first 2x32 bits are not interesting for us.

            var imageData = System.IO.File.ReadAllBytes(imgFileName);
            int imageDataOffset = 16; //first 4x32 bits are not interesting for us.
            int imageSize = bitmap.Size.Width * bitmap.Size.Height;

            for (int i = labelDataOffset; i < labelData.Length; i++)
            {
                int trainingSampleId = i - labelDataOffset;
                int label = labelData[i];
                float[] input = new float[imageSize];
                float[] output = new float[10];
                for (int j = 0; j < bitmap.Size.Height; j++)
                {
                    for (int k = 0; k < bitmap.Size.Width; k++)
                    {
                        int offsetInImage = j * bitmap.Size.Width + k;
                        byte pixelColor = imageData[imageDataOffset + trainingSampleId * imageSize + offsetInImage];
                        input[offsetInImage] = ((float)pixelColor) / 255.0f;
                        //bitmap.SetPixel(k, j, Color.FromArgb(255, 255- pixelColor, 255 - pixelColor, 255 - pixelColor));
                    }
                }
                /*
                pictureBox1.Refresh();
                System.Threading.Thread.Sleep(100);*/
                output[label] = 1.0f;
                trainingData.Add(new TrainingSuite.TrainingData(input, output));
            }
        }

        private void button4_Click(object sender, EventArgs e)
        {
            string imgFile = "";
            string labelFile = "";

            openFileDialog1.Filter = "Image Training data (Image)|*.*";
            openFileDialog1.Title = "Open Training images file";
            if (openFileDialog1.ShowDialog() == DialogResult.OK)
            {
                imgFile = openFileDialog1.FileName;
            }
            else
            {
                return;
            }

            openFileDialog1.Filter = "Training data (Label)|*.*";
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

            LoadTestDataFromFiles(trainingData, labelFile, imgFile);

            var trainingSuite = new TrainingSuite(trainingData);
            trainingSuite.config.miniBatchSize = (int)numMiniBatchSize.Value;
            trainingSuite.config.learningRate = (float)numLearningRate.Value;
            trainingSuite.config.regularizationLambda = (float)numLambda.Value;
            trainingSuite.config.shuffleTrainingData= checkShuffle.Checked;

            if (comboRegularization.SelectedIndex == 0)
                trainingSuite.config.regularization = TrainingSuite.TrainingConfig.Regularization.None;
            else if (comboRegularization.SelectedIndex == 0)
                trainingSuite.config.regularization = TrainingSuite.TrainingConfig.Regularization.L1;
            else if (comboRegularization.SelectedIndex == 0)
                trainingSuite.config.regularization = TrainingSuite.TrainingConfig.Regularization.L2;

            if (comboCostFunction.SelectedIndex == 0)
                trainingSuite.config.costFunction = new MeanSquaredError();
            else if (comboCostFunction.SelectedIndex == 1)
                trainingSuite.config.costFunction = new CrossEntropy();

            trainingSuite.config.epochs = (int)numEpoch.Value;

            trainingPromise = network.Train(mathLib, trainingSuite);
            trainingtimer.Start();

            progressDialog = new Form2(trainingPromise);
            progressDialog.ShowDialog();
        }

        private void ClearBitmap()
        {
            for (int i = 0; i < bitmap.Size.Height; i++)
            {
                for (int j = 0; j < bitmap.Size.Width; j++)
                {
                    bitmap.SetPixel(j, i, Color.White);
                }
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            ClearBitmap();
            pictureBox1.Refresh();
        }

        private void pictureBox1_MouseDown(object sender, MouseEventArgs e)
        {
            paintPixel(e);
        }

        private void pictureBox1_MouseUp(object sender, MouseEventArgs e)
        {
        }

        private void paintPixel(MouseEventArgs e)
        {
            var centerX = (int)Math.Floor(((float)e.X / (float)pictureBox1.Width) * (float)(bitmap.Size.Width - 1));
            var centerY = (int)Math.Floor(((float)e.Y / (float)pictureBox1.Height) * (float)(bitmap.Size.Height- 1));

            Action<int, int, int> applyColor = (x,y,c) => {
                int xClamped = Math.Max(0, Math.Min(x, bitmap.Size.Width - 1));
                int yClamped = Math.Max(0, Math.Min(y, bitmap.Size.Height - 1));
                bitmap.SetPixel(xClamped, yClamped, Color.FromArgb(255,c,c,c) );
            };

            applyColor(centerX, centerY, 0);

            pictureBox1.Refresh();
        }

        private void pictureBox1_MouseMove(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Left)
            {
                paintPixel(e);
            }
        }

        private void pictureBox1_MouseLeave(object sender, EventArgs e)
        {
        }

        private int ClassifyOutput(float[] output)
        {
            float largest = -1;
            int resultIdx = -1;
            for (int i = 0; i < output.Length; i++)
            {
                if (output[i] > largest)
                {
                    largest = output[i];
                    resultIdx = i;
                }
            }
            return resultIdx;
        }

        private void button5_Click(object sender, EventArgs e)
        {
            float[] input = new float[bitmap.Size.Width * bitmap.Size.Height];

            for (int i = 0; i < bitmap.Size.Height; i++)
            {
                for (int j = 0; j < bitmap.Size.Width; j++)
                {
                    var color = bitmap.GetPixel(j,i).GetBrightness();
                    input[i * bitmap.Size.Width + j] = 1.0f - color; //in input 1.0f is black, 0.0f is white
                }
            }

            var output = network.Compute(mathLib, input);

            int resultIdx = ClassifyOutput(output);

            lblResult.Text = "Results:\nI think you drew a " + resultIdx + "\nOutput was:\n";
            lblResult.Text += string.Join("\n ", output);
        }

        private void panel1_Paint(object sender, PaintEventArgs e)
        {

        }

        private void button7_Click(object sender, EventArgs e)
        {
            string imgFile = "";
            string labelFile = "";

            openFileDialog1.Filter = "Test data (Image)|*.*";
            openFileDialog1.Title = "Open Test images file";
            if (openFileDialog1.ShowDialog() == DialogResult.OK)
            {
                imgFile = openFileDialog1.FileName;
            }
            else
            {
                return;
            }

            openFileDialog1.Filter = "Test data (Label)|*.*";
            openFileDialog1.Title = "Open Test labels file";
            if (openFileDialog1.ShowDialog() == DialogResult.OK)
            {
                labelFile = openFileDialog1.FileName;
            }
            else
            {
                return;
            }

            List<TrainingSuite.TrainingData> trainingData = new List<TrainingSuite.TrainingData>();

            LoadTestDataFromFiles(trainingData, labelFile, imgFile);

            int success = 0;
            for (int i = 0; i < trainingData.Count; i++)
            {
                var output = network.Compute(mathLib, trainingData[i].input);

                int resultIdx = ClassifyOutput(output);
                int expectedIdx = ClassifyOutput(trainingData[i].desiredOutput);
                if (resultIdx == expectedIdx)
                    ++success;

            }

            float perc = ((float)success / (float)trainingData.Count) * 100.0f;
            MessageBox.Show("Test completed with " + trainingData.Count + " examples. Successful were: " + success + " (" + perc + "%)", "Test complete", MessageBoxButtons.OK, MessageBoxIcon.Information);
        }

        private void pictureBox1_Click(object sender, EventArgs e)
        {

        }

        private void button8_Click(object sender, EventArgs e)
        {
            string imgFile = "";

            openFileDialog1.Filter = "Image Training data (Image)|*.*";
            openFileDialog1.Title = "Open Training images file";
            if (openFileDialog1.ShowDialog() == DialogResult.OK)
            {
                imgFile = openFileDialog1.FileName;
            }
            else
            {
                return;
            }

            byte[] content = System.IO.File.ReadAllBytes(imgFile);

            int imgid = (int)numericUpDown2.Value;
            int imageDataOffset = 16; //first 4x32 bits are not interesting for us.
            int imageSize = bitmap.Size.Width * bitmap.Size.Height;

            for (int i = 0; i < bitmap.Size.Height; i++)
            {
                for (int j = 0; j < bitmap.Size.Width; j++)
                {
                    int c = 255 - content[imageDataOffset + imageSize * imgid + i * bitmap.Size.Width + j];
                    bitmap.SetPixel(j, i, Color.FromArgb(255, c, c, c));
                }
            }
            pictureBox1.Refresh();
        }
    }
}
