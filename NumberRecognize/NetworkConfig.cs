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

    //TODO: Clean up this UI setup code, its not easily maintainable

    public partial class NetworkConfig : Form
    {
        List<int> layerconfig = new List<int>();
        int prevControls = 0;
        int paddingTop = 10;
        int elementHeight = 30;

        NumericUpDown outputLayerNum = null;
        Label outputLayerLbl= null;

        public NetworkConfig()
        {
            InitializeComponent();
            layerconfig.Add(28 * 28);
            layerconfig.Add(10);

            {
                Label lbl = new Label();
                lbl.Text = "Input layer";
                lbl.Top = paddingTop;
                panel1.Controls.Add(lbl);

                NumericUpDown num = new NumericUpDown();
                num.Minimum = 784;
                num.Maximum = 784;
                num.Value = 784;
                num.Top = paddingTop;
                num.ReadOnly = true;
                panel1.Controls.Add(num);
                num.Width = 80;
                num.Left = 200;
            }

            {
                Label lbl = new Label();
                lbl.Text = "Output layer";
                lbl.Top = paddingTop + 3;
                panel1.Controls.Add(lbl);
                outputLayerLbl = lbl;

                NumericUpDown num = new NumericUpDown();
                num.Top = paddingTop;
                num.ReadOnly = true;
                num.Value = 10;
                panel1.Controls.Add(num);
                outputLayerNum = num;
                num.Top = elementHeight * 2;
                num.Width = 80;
                num.Left = 200;

            }

            numericUpDown1.Value = 1;
        }

        public List<int> GetLayerConfig() { return layerconfig; }

        private void numericUpDown1_ValueChanged(object sender, EventArgs e)
        {
            List<Control> itemsToRemove = new List<Control>();

            foreach (var item in panel1.Controls)
            {
                if (((Control)item).Tag != null && (int)((Control)item).Tag - 1 > numericUpDown1.Value)
                    itemsToRemove.Add((Control)item);
            }

            foreach (var item in itemsToRemove)
            {
                panel1.Controls.Remove(item);
            }

            if ( prevControls > (int)numericUpDown1.Value)
                layerconfig.RemoveRange((int)numericUpDown1.Value, prevControls - (int)numericUpDown1.Value );

            for (int i = prevControls; i < numericUpDown1.Value; i++)
            {
                Label lbl = new Label();
                lbl.Text = "Hidden layer #" + (i+1);
                lbl.Tag = i + 1;
                panel1.Controls.Add(lbl);
                lbl.Top = (i + 1) * elementHeight + paddingTop + 3;
                lbl.Height = elementHeight;

                NumericUpDown num = new NumericUpDown();
                num.Minimum= 1;
                num.Maximum= 65536;
                num.Value = 64;
                num.ValueChanged += Num_ValueChanged;
                num.Tag = i+1;
                panel1.Controls.Add(num);
                num.Top = (i+1) * elementHeight + paddingTop;
                num.Width = 80;
                num.Left = 200;

                layerconfig.Insert(layerconfig.Count-1, (int)num.Value);
            }

            outputLayerLbl.Top = paddingTop + ((int)numericUpDown1.Value+1) * elementHeight + 3; 
            outputLayerNum.Top = paddingTop + ((int)numericUpDown1.Value + 1) * elementHeight;

            prevControls = (int)numericUpDown1.Value;
        }

        private void Num_ValueChanged(object sender, EventArgs e)
        {
            NumericUpDown num = (NumericUpDown)sender;

            if (num == null)
                return;
            layerconfig[(int)num.Tag] = (int)num.Value;
        }

        private void Form1_Load(object sender, EventArgs e)
        {
        }

        private void button1_Click(object sender, EventArgs e)
        {
            this.DialogResult = DialogResult.OK;
            Close();
        }
    }
}
