using Macademy;
using System;
using System.Collections.Generic;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using System.Windows.Threading;
using static Macademy.Network;

namespace NewsAnalyzer
{
    /// <summary>
    /// Interaction logic for TrainingDialog.xaml
    /// </summary>
    public partial class TrainingDialog : Window
    {
        private Network.TrainingPromise trainingPromise = null;
        private static int updateInterval = 500;
        private long ticks = 0;
        public TrainingDialog(TrainingPromise _promise)
        {
            InitializeComponent();
            trainingPromise = _promise;

            DispatcherTimer timer = new DispatcherTimer();
            timer.Interval = TimeSpan.FromMilliseconds(updateInterval);
            timer.Tick += timer_Tick;
            timer.Start();
        }

        void timer_Tick(object sender, EventArgs e)
        {
            if (trainingPromise.IsReady())
                this.Close();
            ++ticks;
            long elapsedTotalSeconds = (ticks * (long)updateInterval) / 1000L;
            int dispMinutes = (int)(elapsedTotalSeconds / 60L);
            int dispSeconds = (int)(elapsedTotalSeconds % 60L);

            int percentageDone = (int)(trainingPromise.GetTotalProgress() * 100.0f);

            progressBar.Value = trainingPromise.GetTotalProgress() * 100.0f;

            lblDisp.Content = "Epochs done: " + trainingPromise.GetEpochsDone() + " (" + percentageDone + "%)   Time elapsed: " + dispMinutes + ":" + dispSeconds;
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            trainingPromise.StopAtNextEpoch();
        }
    }
}
