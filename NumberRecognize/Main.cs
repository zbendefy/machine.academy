using Macademy;
using Macademy.OpenCL;
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
    public partial class Main : Form
    {
        private ComputeDevice calculator = null;
        private Network network = null;
        Bitmap bitmap;
        Bitmap bitmapDownscaled;
        private Network.TrainingPromise trainingPromise = null;
        Timer trainingtimer = new Timer();
        TrainingWindow progressDialog = null;
        NetworkConfig layerConfWindow = new NetworkConfig();
        DateTime trainingStart;

        private int targetWidth = 28, targetHeight = 28;
        private int downScaleWidth = 20, downScaleHeight = 20;

        public Main()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            comboRegularization.SelectedIndex = 2;
            comboCostFunction.SelectedIndex = 1;

            calculator = ComputeDeviceFactory.CreateFallbackComputeDevice();

            bitmap = new Bitmap(targetWidth, targetHeight, System.Drawing.Imaging.PixelFormat.Format16bppRgb565);
            bitmapDownscaled = new Bitmap(downScaleWidth, downScaleHeight, System.Drawing.Imaging.PixelFormat.Format16bppRgb565);
            ClearBitmap();
            pictureBox1.Image = bitmap;

            foreach (var device in ComputeDeviceFactory.GetComputeDevices())
            {
                string item = device.GetDeviceAccessType() + " - " + device.GetDeviceName();
                comboBox1.Items.Add(item);
            }
            comboBox1.SelectedIndex = 0;

            trainingtimer.Interval = 300;
            trainingtimer.Tick += Trainingtimer_Tick;

            InitRandomNetwork();

        }

        private void Trainingtimer_Tick(object sender, EventArgs e)
        {
            if (progressDialog != null && trainingPromise != null)
            {
                var timespan = (DateTime.Now - trainingStart);
                string time = new TimeSpan(timespan.Hours, timespan.Minutes, timespan.Seconds).ToString();

                progressDialog.UpdateResult(trainingPromise.GetTotalProgress(), trainingPromise.IsReady(), "Training... Epochs done: " + trainingPromise.GetEpochsDone(), time);
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
            List<int> layerConfig = layerConfWindow.GetLayerConfig();

            network = Network.CreateNetworkInitRandom(layerConfig.ToArray(), new SigmoidActivation(), new DefaultWeightInitializer());
            lblnetcfg.Text = String.Join("x", network.GetLayerConfig());
            network.AttachName("MNIST learning DNN");
            network.AttachDescription("MNIST learning DNN using " + layerConfig.Count + " layers in structure: (" + string.Join(", ", layerConfig) + " ). Creation date: " + DateTime.Now.ToString() );
        }

        private void comboBox1_SelectedIndexChanged(object sender, EventArgs e)
        {
            var devices = ComputeDeviceFactory.GetComputeDevices();
            calculator = ComputeDeviceFactory.CreateComputeDevice(devices[comboBox1.SelectedIndex]);
        }

        private void button3_Click(object sender, EventArgs e)
        {
            if (layerConfWindow.ShowDialog() != DialogResult.OK)
                return;

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
                    System.IO.File.WriteAllText(saveFileDialog1.FileName, network.ExportToJSON());
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
                    var newNetwork = Network.CreateNetworkFromJSON(file);
                    network = newNetwork;

                    lblnetcfg.Text = String.Join("x", network.GetLayerConfig());
                }
                catch (Exception exc)
                {
                    MessageBox.Show("Error when loading network: " + exc.ToString(), "Error",MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            }
        }

        private void LoadTestDataFromFiles(List<TrainingSuite.TrainingData> trainingData, String labelFileName, String imgFileName, Action<int> progressHandler = null, bool diversify = false)
        {
            var labelData = System.IO.File.ReadAllBytes(labelFileName);
            int labelDataOffset = 8; //first 2x32 bits are not interesting for us.

            var imageData = System.IO.File.ReadAllBytes(imgFileName);
            int imageDataOffset = 16; //first 4x32 bits are not interesting for us.
            int imageSize = targetWidth * targetHeight;

            Random rnd = new Random();

            for (int i = labelDataOffset; i < labelData.Length; i++)
            {
                float diversificationFactor = (float)(rnd.NextDouble() * 0.3 + 0.7);
                int trainingSampleId = i - labelDataOffset;
                int label = labelData[i];
                float[] input = new float[imageSize];
                float[] output = new float[10];
                for (int j = 0; j < targetHeight; j++)
                {
                    for (int k = 0; k < targetWidth; k++)
                    {
                        int offsetInImage = j * targetWidth + k;
                        byte pixelColor = imageData[imageDataOffset + trainingSampleId * imageSize + offsetInImage]; //0 is white, 255 is black
                        input[offsetInImage] = ((float)( 255 - pixelColor )) / 255.0f; //1.0 is white, 0.0 is black
                        if (diversify)
                        {
                            input[offsetInImage] *= diversificationFactor;
                        }
                        //bitmap.SetPixel(k, j, Color.FromArgb(255, 255- pixelColor, 255 - pixelColor, 255 - pixelColor));
                    }
                }

                if ( progressHandler != null)
                {
                    if (i % 200 == 0)
                    {
                        progressHandler(((i - labelDataOffset)*100) / (labelData.Length-labelDataOffset)); 
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

            LoadingWindow wnd = new LoadingWindow();
            wnd.Text = "Loading training data";

            List<TrainingSuite.TrainingData> trainingData = new List<TrainingSuite.TrainingData>();

            System.Threading.Thread thread = new System.Threading.Thread(()=> {
                LoadTestDataFromFiles(trainingData, labelFile, imgFile, (x)=> { wnd.SetProgress(x); }, true);
                wnd.Finish();
            });

            thread.Start();

            if (wnd.ShowDialog() != DialogResult.OK)
                return;

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
                trainingSuite.config.costFunction = new MeanSquaredErrorFunction();
            else if (comboCostFunction.SelectedIndex == 1)
                trainingSuite.config.costFunction = new CrossEntropyErrorFunction();

            trainingSuite.config.epochs = (int)numEpoch.Value;

            trainingStart = DateTime.Now;
            trainingPromise = network.Train(trainingSuite, calculator);
            trainingtimer.Start();


            progressDialog = new TrainingWindow(trainingPromise);
            progressDialog.ShowDialog();
        }

        private void ClearBitmap()
        {
            for (int i = 0; i < targetHeight; i++)
            {
                for (int j = 0; j < targetWidth; j++)
                {
                    bitmap.SetPixel(j, i, Color.White);
                }
            }

            for (int i = 0; i < downScaleHeight; i++)
            {
                for (int j = 0; j < downScaleWidth; j++)
                {
                    bitmapDownscaled.SetPixel(j, i, Color.White);
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
            var centerX = (int)Math.Floor(((float)e.X / (float)pictureBox1.Width) * (float)(bitmapDownscaled.Size.Width - 1));
            var centerY = (int)Math.Floor(((float)e.Y / (float)pictureBox1.Height) * (float)(bitmapDownscaled.Size.Height- 1));

            Action<int, int, int> applyColor = (x,y,c) => {
                int xClamped = Math.Max(0, Math.Min(x, bitmapDownscaled.Size.Width - 1));
                int yClamped = Math.Max(0, Math.Min(y, bitmapDownscaled.Size.Height - 1));
                bitmapDownscaled.SetPixel(xClamped, yClamped, Color.FromArgb(255,c,c,c) );
            };

            applyColor(centerX, centerY, 0);

            //upscale
            for (int i = 0; i < targetHeight; i++)
            {
                for (int j = 0; j < targetWidth; j++)
                {
                    float xRatio = (float)j / (float)(targetWidth - 1);
                    float yRatio = (float)i / (float)(targetHeight - 1);

                    float ds_x = xRatio * (float)downScaleWidth;
                    float ds_y = yRatio * (float)downScaleHeight;

                    float xBias = ds_x - (float)Math.Floor(ds_x);
                    float yBias = ds_y - (float)Math.Floor(ds_y);

                    int ds_x_int = (int)ds_x;
                    int ds_y_int = (int)ds_y;

                    bool isAtXBorder = ds_x_int >= downScaleWidth - 1;
                    bool isAtYBorder = ds_y_int >= downScaleHeight - 1;

                    float v = 1;
                    float vx = 1;
                    float vy = 1;
                    float vxy = 1;

                    if (!isAtXBorder && !isAtYBorder)
                    {
                        v = bitmapDownscaled.GetPixel(ds_x_int, ds_y_int).GetBrightness();
                        vx = bitmapDownscaled.GetPixel(ds_x_int + 1, ds_y_int).GetBrightness();
                        vy = bitmapDownscaled.GetPixel(ds_x_int, ds_y_int + 1).GetBrightness();
                        vxy = bitmapDownscaled.GetPixel(ds_x_int + 1, ds_y_int + 1).GetBrightness();
                    }

                    float b1 = vx * xBias + (1.0f - xBias) * v;
                    float b2 = vxy * xBias + (1.0f - xBias) * vy;
                    float b3 = b2 * yBias + (1.0f - yBias) * b1;

                    int c = (int)(b3 * 255.0f);
                    bitmap.SetPixel(j, i, Color.FromArgb(255, c, c, c));
                }
            }

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
            float[] input = new float[targetWidth * targetHeight];

            for (int i = 0; i < targetHeight; i++)
            {
                for (int j = 0; j < targetWidth; j++)
                {
                    var color = bitmap.GetPixel(j,i).GetBrightness();
                    input[i * targetWidth + j] = color; //in input 0.0f is black, 1.0f is white
                }
            }

            var output = network.Compute(input, calculator);

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

            LoadingWindow wnd = new LoadingWindow();
            wnd.Text = "Testing network";
            int success = 0;

            var thread = new System.Threading.Thread(() => {

                wnd.SetText("Opening training file...");
                LoadTestDataFromFiles(trainingData, labelFile, imgFile, (x)=> { wnd.SetProgress(x/10); });

                wnd.SetProgress(10);

                wnd.SetText("Testing...");
                for (int i = 0; i < trainingData.Count; i++)
                {
                    var output = network.Compute(trainingData[i].input, calculator);

                    int resultIdx = ClassifyOutput(output);
                    int expectedIdx = ClassifyOutput(trainingData[i].desiredOutput);
                    if (resultIdx == expectedIdx)
                        ++success;

                    if (i % 200 == 0)
                        wnd.SetProgress(10 + ((i*90) / trainingData.Count));
                }

                wnd.Finish();
            });

            thread.Start();

            if (wnd.ShowDialog() != DialogResult.OK)
                return;

            float perc = ((float)success / (float)trainingData.Count) * 100.0f;

            MessageBox.Show("Test completed with " + trainingData.Count + " examples. Successful were: " + success + " (" + perc + "%)", "Test complete", MessageBoxButtons.OK, MessageBoxIcon.Information);
        }

        private void pictureBox1_Click(object sender, EventArgs e)
        {

        }




        private System.Drawing.Color GetWeightColor(float val, float maxAbsValue)
        {
            val /= maxAbsValue;
            if (val > 0)
            {
                val = Math.Min(val * 255.0f, 255.0f);
                return System.Drawing.Color.FromArgb(255, (int)(val), (int)(val * 0.15f), 0);
            }
            else
            {
                val = Math.Min(val * -255.0f, 255.0f);
                return System.Drawing.Color.FromArgb(255, 0, (int)(val * 0.7f), (int)(val));
            }
        }
        private System.Drawing.Color GetBiasColor(float val, float maxAbsValue)
        {
            val /= maxAbsValue;
            if (val > 0)
            {
                val = Math.Min((val) * 255.0f, 255.0f);
                return System.Drawing.Color.FromArgb(255, (int)(val), (int)(val * 0.8f), (int)(val * 0.1f));
            }
            else
            {
                val = Math.Min(val * -255.0f, 255.0f);
                return System.Drawing.Color.FromArgb(255, (int)(val * 0.2), (int)(val * 0.95f), (int)(val * 0.4f));
            }
        }

        private void LogDebug(string msg)
        {
            System.IO.File.AppendAllText("D:\\nntmp\\log.txt", msg + "\n");
        }

        private void VisualizeNetwork(Network network, string filePath, int epoch, float successRate)
        {
            Font drawFont = new Font("Arial", 13);
            Brush brush = new SolidBrush(Color.White);
            LogDebug("VisualizeNetwork " + DateTime.Now.ToString());
            int padding = 1;
            int textarea = 100;
            if (network != null)
            {
                var structure = network.__GetInternalConfiguration();
                int lyr = 0;
                foreach (var neuronData in structure)
                {
                    var prevLayerNeuronCount = neuronData[0].weights.Length;
                    var isImgLayer = prevLayerNeuronCount == 784;

                    int weightsContentSizeX = isImgLayer ? 28 : 1;
                    int weightsContentSizeY = isImgLayer ? 28 : prevLayerNeuronCount;

                    int neuronWidth = weightsContentSizeX + 1;
                    int neuronHeight = weightsContentSizeY + 1;

                    int bitmapSizeX = Math.Max( isImgLayer ? (8 * (neuronWidth+padding)) : ((neuronData.Count) * (neuronWidth + padding)), 240);
                    int bitmapSizeY = textarea + (isImgLayer ? (8 * (neuronHeight+ padding)) : (neuronHeight + padding));
                    Bitmap bitmap = new Bitmap(bitmapSizeX, bitmapSizeY);

                    float highestAbsoluteWeight = 0;
                    float highestAbsoluteBias = 0;

                    for (int i = 0; i < neuronData.Count; i++)
                    {
                        if (Math.Abs(neuronData[i].bias) > highestAbsoluteBias)
                        {
                            highestAbsoluteBias = Math.Abs(neuronData[i].bias);
                        }

                        for (int j = 0; j < neuronData[i].weights.Length; j++)
                        {
                            if (Math.Abs(neuronData[i].weights[j]) > highestAbsoluteWeight)
                            {
                                highestAbsoluteWeight = Math.Abs(neuronData[i].weights[j]);
                            }
                        }
                    }

                    for (int i = 0; i < neuronData.Count; i++)
                    {
                        int neuronTopLeftCoordX = isImgLayer ? ((i % 8)* (neuronWidth+padding)) : (i * (neuronWidth+padding));
                        int neuronTopLeftCoordY = isImgLayer ? ((i / 8)*(neuronHeight+padding)) : 0;

                        for (int j = 0; j < neuronData[i].weights.Length; j++)
                        {
                            int px = j % weightsContentSizeX;
                            int py = j / weightsContentSizeX;

                            bitmap.SetPixel(neuronTopLeftCoordX + px, neuronTopLeftCoordY + py, GetWeightColor(neuronData[i].weights[j], highestAbsoluteWeight));
                        }

                        var biascolor = GetBiasColor(neuronData[i].bias, highestAbsoluteBias);
                        for (int j = 0; j < weightsContentSizeY; j++)
                        {
                            bitmap.SetPixel(neuronTopLeftCoordX + weightsContentSizeX, neuronTopLeftCoordY + j, biascolor);
                        }
                    }
                    var g = Graphics.FromImage(bitmap);
                    g.DrawString("Epoch: " + epoch + "\nSucess rate: " + successRate.ToString("n2"), drawFont, brush, new PointF(10, bitmap.Height - 90) );
                    bitmap.Save(filePath + lyr + ".bmp", System.Drawing.Imaging.ImageFormat.Bmp);
                    ++lyr;
                }
            }
        }

        private void DrawLegend(string name, int x, int y, Graphics g, int scaleFactor, Font smallFont, Brush fontBrush, float highestAbsoluteBias, float highestAbsoluteWeight)
        {
            g.DrawString(name, smallFont, fontBrush, new PointF(x, y - 20*scaleFactor));

            g.DrawString("Bias: " + highestAbsoluteBias, smallFont, fontBrush, new PointF(x, y));
            g.FillRectangle(new SolidBrush(GetBiasColor(1, 1)), x - 10*scaleFactor, y, 8 * scaleFactor, 8 * scaleFactor);

            g.DrawString("Bias: 0.0", smallFont, fontBrush, new PointF(x, y + 20 * scaleFactor));
            g.FillRectangle(new SolidBrush(Color.Black), x - 10*scaleFactor, y + 20 * scaleFactor, 8 * scaleFactor, 8 * scaleFactor);

            g.DrawString("Bias: " + (-highestAbsoluteBias), smallFont, fontBrush, new PointF(x, y + 40 * scaleFactor));
            g.FillRectangle(new SolidBrush(GetBiasColor(-1, 1)), x - 10*scaleFactor, y + 40 * scaleFactor, 8 * scaleFactor, 8 * scaleFactor);

            g.DrawString("Weight: " + highestAbsoluteWeight, smallFont, fontBrush, new PointF(x, y + 70 * scaleFactor));
            g.FillRectangle(new SolidBrush(GetWeightColor(1, 1)), x - 10*scaleFactor, y + 70 * scaleFactor, 8 * scaleFactor, 8 * scaleFactor);

            g.DrawString("Weight: 0.0", smallFont, fontBrush, new PointF(x, y + 90 * scaleFactor));
            g.FillRectangle(new SolidBrush(Color.Black), x - 10*scaleFactor, y + 90 * scaleFactor, 8 * scaleFactor, 8 * scaleFactor);

            g.DrawString("Weight: " + (-highestAbsoluteWeight), smallFont, fontBrush, new PointF(x, y + 110 * scaleFactor));
            g.FillRectangle(new SolidBrush(GetWeightColor(-1, 1)), x - 10*scaleFactor, y + 110 * scaleFactor, 8 * scaleFactor, 8 * scaleFactor);
        }

        private void VisualizeNetworkSpecific(Network network, string filePath, int epoch, float successRate)
        {
            int scaleFactor = 3;

            Font drawFont = new Font("Arial", 13 * scaleFactor);
            Font smallFont = new Font("Arial", 7 * scaleFactor);
            Brush brush = new SolidBrush(Color.White);



            LogDebug("VisualizeNetwork " + DateTime.Now.ToString());
            int padding = 1 * scaleFactor;
            if (network != null)
            {
                var structure = network.__GetInternalConfiguration();
                int lyr = 0;

                int bitmapSizeX = 640 * scaleFactor;
                int bitmapSizeY = 400 * scaleFactor;
                Bitmap bitmap = new Bitmap(bitmapSizeX, bitmapSizeY);
                var g = Graphics.FromImage(bitmap);
                g.Clear(Color.FromArgb(255,40,40,40));



                foreach (var neuronData in structure)
                {
                    var prevLayerNeuronCount = neuronData[0].weights.Length;
                    var isImgLayer = prevLayerNeuronCount == 784;

                    int weightsContentSizeX_nonscaled = (isImgLayer ? 28 : 8);
                    int weightsContentSizeY_nonscaled = (isImgLayer ? 28 : 8);

                    int weightsContentSizeX = scaleFactor * weightsContentSizeX_nonscaled;
                    int weightsContentSizeY = scaleFactor * weightsContentSizeY_nonscaled;

                    int neuronWidth  =( weightsContentSizeX + 1 * scaleFactor );
                    int neuronHeight =( weightsContentSizeY + 1 * scaleFactor );

                    float highestAbsoluteWeight = 0;
                    float highestAbsoluteBias = 0;

                    for (int i = 0; i < neuronData.Count; i++)
                    {
                        if (Math.Abs(neuronData[i].bias) > highestAbsoluteBias)
                        {
                            highestAbsoluteBias = Math.Abs(neuronData[i].bias);
                        }

                        for (int j = 0; j < neuronData[i].weights.Length; j++)
                        {
                            if (Math.Abs(neuronData[i].weights[j]) > highestAbsoluteWeight)
                            {
                                highestAbsoluteWeight = Math.Abs(neuronData[i].weights[j]);
                            }
                        }
                    }

                    int layerScaleFactor = isImgLayer ? 1 : 3;

                    for (int i = 0; i < neuronData.Count; i++)
                    {
                        int neuronTopLeftCoordX = isImgLayer ? (scaleFactor*30 + (i % 8) * (neuronWidth + (padding+ 1 * scaleFactor))) : (350 * scaleFactor);
                        int neuronTopLeftCoordY = scaleFactor * 50 + (isImgLayer ? ((i / 8) * (neuronHeight + (padding+1*scaleFactor))) : (i * (neuronHeight* layerScaleFactor + (4*scaleFactor ))));

                        for (int j = 0; j < neuronData[i].weights.Length; j++)
                        {
                            int px = (j % weightsContentSizeX_nonscaled) * (scaleFactor * layerScaleFactor);
                            int py = (j / weightsContentSizeX_nonscaled) * (scaleFactor * layerScaleFactor);
                            int border = isImgLayer ? 0 : 2;
                            var wcolor = GetWeightColor(neuronData[i].weights[j], highestAbsoluteWeight);
                            for (int scx = 0; scx < scaleFactor * layerScaleFactor - border; scx++)
                            {
                                for (int scy = 0; scy < scaleFactor * layerScaleFactor - border; scy++)
                                {
                                    bitmap.SetPixel(neuronTopLeftCoordX + px + scx, neuronTopLeftCoordY + py + scy, wcolor);
                                }
                            }
                        }

                        var biascolor = GetBiasColor(neuronData[i].bias, highestAbsoluteBias);
                        for (int j = 0; j < weightsContentSizeY * layerScaleFactor; j++)
                        {
                            for (int scx = 0; scx < scaleFactor * (isImgLayer ? 1 : 2); scx++)
                            {
                                for (int scy = 0; scy < scaleFactor; scy++)
                                {
                                    bitmap.SetPixel(neuronTopLeftCoordX + weightsContentSizeX* layerScaleFactor + scx, neuronTopLeftCoordY + j + scy, biascolor);
                                }
                            }
                        }
                    }

                    DrawLegend( isImgLayer?"Hidden Layer":"Output Layer", bitmap.Width - 100*scaleFactor, 50 * scaleFactor+ lyr*200 * scaleFactor, g, scaleFactor, smallFont, fontBrush:brush, highestAbsoluteBias, highestAbsoluteWeight);
                    ++lyr;
                }
                g.DrawString("Hidden Layer", drawFont, brush, new PointF(60 * scaleFactor, 15 * scaleFactor));
                g.DrawString("Output layer", drawFont, brush, new PointF(300 * scaleFactor, 15 * scaleFactor));
                for (int i = 0; i < 10; i++)
                {
                    g.DrawString(i.ToString(), drawFont, brush, new PointF(375 * scaleFactor, ((60 + (i * 31)) * scaleFactor)));
                }
                g.DrawString("Epoch: " + epoch + "\nSucess rate: " + successRate.ToString("n2") + "%", drawFont, brush, new PointF(10 * scaleFactor, bitmap.Height - 90 * scaleFactor));


                bitmap.Save(filePath + lyr + ".bmp", System.Drawing.Imaging.ImageFormat.Bmp);
            }
        }


        private void Button9_Click(object sender, EventArgs e)
        {
            System.IO.File.WriteAllText("D:\\nntmp\\log.txt", "BEGIN\n");

            string imgFile = "";
            string labelFile = "";
            string testImgFile = "";
            string testLabelFile = "";

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

            openFileDialog1.Filter = "Verification Image Training data (Image)|*.*";
            openFileDialog1.Title = "Open Verification images file";
            if (openFileDialog1.ShowDialog() == DialogResult.OK)
            {
                testImgFile = openFileDialog1.FileName;
            }
            else
            {
                return;
            }

            openFileDialog1.Filter = "Verification Training data (Label)|*.*";
            openFileDialog1.Title = "Open Verification labels file";
            if (openFileDialog1.ShowDialog() == DialogResult.OK)
            {
                testLabelFile = openFileDialog1.FileName;
            }
            else
            {
                return;
            }

            LoadingWindow wnd = new LoadingWindow();
            wnd.Text = "Loading training data";

            List<TrainingSuite.TrainingData> trainingData = new List<TrainingSuite.TrainingData>();
            List<TrainingSuite.TrainingData> testData = new List<TrainingSuite.TrainingData>();

            System.Threading.Thread thread = new System.Threading.Thread(() => {
                LoadTestDataFromFiles(trainingData, labelFile, imgFile, (x) => { wnd.SetProgress(x); }, true);
                LoadTestDataFromFiles(testData, testLabelFile, testImgFile, (x) => { wnd.SetProgress(x); }, false);
                wnd.Finish();
            });

            thread.Start();

            if (wnd.ShowDialog() != DialogResult.OK)
                return;


            int success = 0;
            for (int i = 0; i < testData.Count; i++)
            {
                var output = network.Compute(testData[i].input, calculator);

                int resultIdx = ClassifyOutput(output);
                int expectedIdx = ClassifyOutput(testData[i].desiredOutput);
                if (resultIdx == expectedIdx)
                    ++success;
            }
            network.AttachDescription("Network (" + string.Join(",", network.GetLayerConfig()) + ") epoch: 0 (initial)   Test success rate: [" + success + " of " + testData.Count + "]");
            System.IO.File.WriteAllText("D:\\nntmp\\network_000000.json", network.ExportToJSON());
            VisualizeNetworkSpecific(network, "D:\\nntmp\\network_000000_vis_", 0, ((float)success / testData.Count)*100.0f);

            var trainingSuite = new TrainingSuite(trainingData);
            trainingSuite.config.miniBatchSize = (int)numMiniBatchSize.Value;
            trainingSuite.config.learningRate = (float)numLearningRate.Value;
            trainingSuite.config.regularizationLambda = (float)numLambda.Value;
            trainingSuite.config.shuffleTrainingData = true;

            if (comboRegularization.SelectedIndex == 0)
                trainingSuite.config.regularization = TrainingSuite.TrainingConfig.Regularization.None;
            else if (comboRegularization.SelectedIndex == 0)
                trainingSuite.config.regularization = TrainingSuite.TrainingConfig.Regularization.L1;
            else if (comboRegularization.SelectedIndex == 0)
                trainingSuite.config.regularization = TrainingSuite.TrainingConfig.Regularization.L2;

            if (comboCostFunction.SelectedIndex == 0)
                trainingSuite.config.costFunction = new MeanSquaredErrorFunction();
            else if (comboCostFunction.SelectedIndex == 1)
                trainingSuite.config.costFunction = new CrossEntropyErrorFunction();
            trainingSuite.config.epochs = 1;

            LogDebug("Initial network saved " + DateTime.Now.ToString());

            for (int epoch = 0; epoch < (int)numEpoch.Value; epoch++)
            {
                LogDebug("Starting epoch #" + (epoch+1) +"  "+DateTime.Now.ToString());

                trainingPromise = network.Train(trainingSuite, calculator);
                trainingPromise.Await();
                LogDebug("  Training finished  " + DateTime.Now.ToString());

                success = 0;
                for (int i = 0; i < testData.Count; i++)
                {
                    var output = network.Compute(testData[i].input, calculator);

                    int resultIdx = ClassifyOutput(output);
                    int expectedIdx = ClassifyOutput(testData[i].desiredOutput);
                    if (resultIdx == expectedIdx)
                        ++success;
                }
                LogDebug("  Verification finished Success rate: [" + success + " of " + testData.Count + "]  " + DateTime.Now.ToString());

                network.AttachDescription( "Network (" + string.Join(",", network.GetLayerConfig()) +  ") epoch: " + (epoch + 1) + "    Test success rate: [" + success + " of " + testData.Count + "]");
                System.IO.File.WriteAllText("D:\\nntmp\\network_" + (epoch+1).ToString().PadLeft(6, '0') + ".json", network.ExportToJSON());
                VisualizeNetworkSpecific(network, "D:\\nntmp\\network_" + (epoch + 1).ToString().PadLeft(6, '0') + "_vis_", epoch+1, ((float)success / testData.Count)*100.0f);
                LogDebug("  Saving finished  " + DateTime.Now.ToString());
            }
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
            int imageSize = targetWidth * targetHeight;

            for (int i = 0; i < targetHeight; i++)
            {
                for (int j = 0; j < targetWidth; j++)
                {
                    int c = 255 - content[imageDataOffset + imageSize * imgid + i * targetWidth + j];
                    bitmap.SetPixel(j, i, Color.FromArgb(255, c, c, c));
                }
            }
            pictureBox1.Refresh();
        }

        private void Main_FormClosing(object sender, FormClosingEventArgs e)
        {
        }
    }
}
