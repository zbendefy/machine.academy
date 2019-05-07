using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using static Mademy.Network;

namespace NumberRecognize
{
    public partial class TrainingWindow : Form
    {
        TrainingPromise trainingPromise;
        public TrainingWindow(TrainingPromise _trainingPromise)
        {
            InitializeComponent();
            trainingPromise = _trainingPromise;
        }

        public void UpdateResult(float percentage, bool isFinished, string text)
        {
            if (isFinished)
            {
                DialogResult = DialogResult.OK;
                this.Close();
            }
            label1.Text = text;
            progressBar1.Value = (int)(percentage*100);
        }
        private void Form2_Load(object sender, EventArgs e)
        {

        }

        private void button1_Click(object sender, EventArgs e)
        {
            trainingPromise.StopAtNextEpoch();
            lblStop.Text = "Stopping after current epoch ends";
        }
    }
}
