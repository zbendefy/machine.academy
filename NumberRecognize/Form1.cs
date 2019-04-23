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

        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            mathLib = new MathLib(null);

            comboBox1.Items.Add("Use CPU calculation");
            comboBox1.SelectedIndex = 0;
            foreach (var device in ComputeDevice.GetDevices())
            {
                string item = "[" + device.GetPlatformID() + ":" + device.GetDeviceID() + ", " + device.GetDeviceType().ToString() + "] " + device.GetName();
                comboBox1.Items.Add(item);
            }

            InitRandomNetwork();

            bitmap = new Bitmap(28, 28,System.Drawing.Imaging.PixelFormat.Format16bppRgb565);
            pictureBox1.Image = bitmap;
        }

        private void InitRandomNetwork()
        {
            List<int> layerConfig = new List<int>();
            layerConfig.Add(5);
            layerConfig.Add(8);
            layerConfig.Add(8);
            layerConfig.Add(5);

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
    }
}
