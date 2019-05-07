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
    public partial class LoadingWindow : Form
    {
        int progressBarValue = 0;
        string txt = null;
        bool finish = false;
        object syncobj = new object();

        public LoadingWindow()
        {
            InitializeComponent();
            timer1.Start();
        }

        public void SetProgress(int progress)
        {
            lock (syncobj)
            {
                progressBarValue = progress;
            }
        }

        public void Finish()
        {
            this.DialogResult = DialogResult.OK;
            finish = true;
        }

        public void SetText(string text)
        {
            txt = text;
        }

        private void timer1_Tick(object sender, EventArgs e)
        {
            lock (syncobj)
            {
                if (finish)
                    this.Close();

                if (txt != null)
                {
                    label1.Text = txt;
                    txt = null;
                }

                progressBar1.Value = progressBarValue;
            }
        }

        private void LoadingWindow_Load(object sender, EventArgs e)
        {

        }
    }
}
