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
    public partial class Form2 : Form
    {
        public Form2()
        {
            InitializeComponent();
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
    }
}
