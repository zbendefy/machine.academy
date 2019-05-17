namespace NumberRecognize
{
    partial class Main
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.pictureBox1 = new System.Windows.Forms.PictureBox();
            this.button1 = new System.Windows.Forms.Button();
            this.button2 = new System.Windows.Forms.Button();
            this.button3 = new System.Windows.Forms.Button();
            this.button4 = new System.Windows.Forms.Button();
            this.button5 = new System.Windows.Forms.Button();
            this.lblResult = new System.Windows.Forms.Label();
            this.label1 = new System.Windows.Forms.Label();
            this.comboBox1 = new System.Windows.Forms.ComboBox();
            this.button6 = new System.Windows.Forms.Button();
            this.openFileDialog1 = new System.Windows.Forms.OpenFileDialog();
            this.saveFileDialog1 = new System.Windows.Forms.SaveFileDialog();
            this.numEpoch = new System.Windows.Forms.NumericUpDown();
            this.button7 = new System.Windows.Forms.Button();
            this.button8 = new System.Windows.Forms.Button();
            this.numericUpDown2 = new System.Windows.Forms.NumericUpDown();
            this.panel1 = new System.Windows.Forms.Panel();
            this.lblnetcfg = new System.Windows.Forms.Label();
            this.label10 = new System.Windows.Forms.Label();
            this.label9 = new System.Windows.Forms.Label();
            this.comboCostFunction = new System.Windows.Forms.ComboBox();
            this.label8 = new System.Windows.Forms.Label();
            this.checkShuffle = new System.Windows.Forms.CheckBox();
            this.label7 = new System.Windows.Forms.Label();
            this.comboRegularization = new System.Windows.Forms.ComboBox();
            this.numLambda = new System.Windows.Forms.NumericUpDown();
            this.label6 = new System.Windows.Forms.Label();
            this.numLearningRate = new System.Windows.Forms.NumericUpDown();
            this.label5 = new System.Windows.Forms.Label();
            this.numMiniBatchSize = new System.Windows.Forms.NumericUpDown();
            this.label4 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numEpoch)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDown2)).BeginInit();
            this.panel1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.numLambda)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLearningRate)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numMiniBatchSize)).BeginInit();
            this.SuspendLayout();
            // 
            // pictureBox1
            // 
            this.pictureBox1.BackColor = System.Drawing.SystemColors.Control;
            this.pictureBox1.Location = new System.Drawing.Point(12, 12);
            this.pictureBox1.Name = "pictureBox1";
            this.pictureBox1.Size = new System.Drawing.Size(280, 280);
            this.pictureBox1.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.pictureBox1.TabIndex = 0;
            this.pictureBox1.TabStop = false;
            this.pictureBox1.Click += new System.EventHandler(this.pictureBox1_Click);
            this.pictureBox1.MouseDown += new System.Windows.Forms.MouseEventHandler(this.pictureBox1_MouseDown);
            this.pictureBox1.MouseLeave += new System.EventHandler(this.pictureBox1_MouseLeave);
            this.pictureBox1.MouseMove += new System.Windows.Forms.MouseEventHandler(this.pictureBox1_MouseMove);
            this.pictureBox1.MouseUp += new System.Windows.Forms.MouseEventHandler(this.pictureBox1_MouseUp);
            // 
            // button1
            // 
            this.button1.Location = new System.Drawing.Point(12, 298);
            this.button1.Name = "button1";
            this.button1.Size = new System.Drawing.Size(54, 23);
            this.button1.TabIndex = 1;
            this.button1.Text = "Clear";
            this.button1.UseVisualStyleBackColor = true;
            this.button1.Click += new System.EventHandler(this.button1_Click);
            // 
            // button2
            // 
            this.button2.Location = new System.Drawing.Point(305, 268);
            this.button2.Name = "button2";
            this.button2.Size = new System.Drawing.Size(217, 23);
            this.button2.TabIndex = 2;
            this.button2.Text = "Load network from JSON";
            this.button2.UseVisualStyleBackColor = true;
            this.button2.Click += new System.EventHandler(this.button2_Click);
            // 
            // button3
            // 
            this.button3.Location = new System.Drawing.Point(305, 297);
            this.button3.Name = "button3";
            this.button3.Size = new System.Drawing.Size(217, 23);
            this.button3.TabIndex = 3;
            this.button3.Text = "Init new random network";
            this.button3.UseVisualStyleBackColor = true;
            this.button3.Click += new System.EventHandler(this.button3_Click);
            // 
            // button4
            // 
            this.button4.Location = new System.Drawing.Point(3, 283);
            this.button4.Name = "button4";
            this.button4.Size = new System.Drawing.Size(274, 23);
            this.button4.TabIndex = 4;
            this.button4.Text = "Train network";
            this.button4.UseVisualStyleBackColor = true;
            this.button4.Click += new System.EventHandler(this.button4_Click);
            // 
            // button5
            // 
            this.button5.Location = new System.Drawing.Point(207, 297);
            this.button5.Name = "button5";
            this.button5.Size = new System.Drawing.Size(85, 23);
            this.button5.TabIndex = 5;
            this.button5.Text = "Recognize!";
            this.button5.UseVisualStyleBackColor = true;
            this.button5.Click += new System.EventHandler(this.button5_Click);
            // 
            // lblResult
            // 
            this.lblResult.AutoSize = true;
            this.lblResult.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblResult.Location = new System.Drawing.Point(302, 12);
            this.lblResult.Name = "lblResult";
            this.lblResult.Size = new System.Drawing.Size(56, 16);
            this.lblResult.TabIndex = 6;
            this.lblResult.Text = "Results:";
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(302, 239);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(87, 13);
            this.label1.TabIndex = 7;
            this.label1.Text = "Compute device:";
            // 
            // comboBox1
            // 
            this.comboBox1.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboBox1.FormattingEnabled = true;
            this.comboBox1.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.comboBox1.Location = new System.Drawing.Point(395, 236);
            this.comboBox1.Name = "comboBox1";
            this.comboBox1.Size = new System.Drawing.Size(336, 21);
            this.comboBox1.TabIndex = 8;
            this.comboBox1.SelectedIndexChanged += new System.EventHandler(this.comboBox1_SelectedIndexChanged);
            // 
            // button6
            // 
            this.button6.Location = new System.Drawing.Point(528, 268);
            this.button6.Name = "button6";
            this.button6.Size = new System.Drawing.Size(203, 23);
            this.button6.TabIndex = 9;
            this.button6.Text = "Save network to JSON";
            this.button6.UseVisualStyleBackColor = true;
            this.button6.Click += new System.EventHandler(this.button6_Click);
            // 
            // openFileDialog1
            // 
            this.openFileDialog1.FileName = "openFileDialog1";
            // 
            // numEpoch
            // 
            this.numEpoch.Location = new System.Drawing.Point(151, 40);
            this.numEpoch.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.numEpoch.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numEpoch.Name = "numEpoch";
            this.numEpoch.Size = new System.Drawing.Size(115, 20);
            this.numEpoch.TabIndex = 10;
            this.numEpoch.Value = new decimal(new int[] {
            10,
            0,
            0,
            0});
            // 
            // button7
            // 
            this.button7.Location = new System.Drawing.Point(528, 297);
            this.button7.Name = "button7";
            this.button7.Size = new System.Drawing.Size(203, 23);
            this.button7.TabIndex = 11;
            this.button7.Text = "Test network";
            this.button7.UseVisualStyleBackColor = true;
            this.button7.Click += new System.EventHandler(this.button7_Click);
            // 
            // button8
            // 
            this.button8.Location = new System.Drawing.Point(72, 298);
            this.button8.Name = "button8";
            this.button8.Size = new System.Drawing.Size(75, 23);
            this.button8.TabIndex = 12;
            this.button8.Text = "Load";
            this.button8.UseVisualStyleBackColor = true;
            this.button8.Click += new System.EventHandler(this.button8_Click);
            // 
            // numericUpDown2
            // 
            this.numericUpDown2.Location = new System.Drawing.Point(153, 300);
            this.numericUpDown2.Maximum = new decimal(new int[] {
            60000,
            0,
            0,
            0});
            this.numericUpDown2.Name = "numericUpDown2";
            this.numericUpDown2.Size = new System.Drawing.Size(48, 20);
            this.numericUpDown2.TabIndex = 13;
            this.numericUpDown2.Value = new decimal(new int[] {
            10,
            0,
            0,
            0});
            // 
            // panel1
            // 
            this.panel1.Controls.Add(this.lblnetcfg);
            this.panel1.Controls.Add(this.label10);
            this.panel1.Controls.Add(this.label9);
            this.panel1.Controls.Add(this.comboCostFunction);
            this.panel1.Controls.Add(this.label8);
            this.panel1.Controls.Add(this.checkShuffle);
            this.panel1.Controls.Add(this.label7);
            this.panel1.Controls.Add(this.comboRegularization);
            this.panel1.Controls.Add(this.numLambda);
            this.panel1.Controls.Add(this.label6);
            this.panel1.Controls.Add(this.numLearningRate);
            this.panel1.Controls.Add(this.label5);
            this.panel1.Controls.Add(this.numMiniBatchSize);
            this.panel1.Controls.Add(this.label4);
            this.panel1.Controls.Add(this.label3);
            this.panel1.Controls.Add(this.label2);
            this.panel1.Controls.Add(this.numEpoch);
            this.panel1.Controls.Add(this.button4);
            this.panel1.Location = new System.Drawing.Point(737, 12);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(280, 309);
            this.panel1.TabIndex = 14;
            // 
            // lblnetcfg
            // 
            this.lblnetcfg.AutoSize = true;
            this.lblnetcfg.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(238)));
            this.lblnetcfg.Location = new System.Drawing.Point(3, 256);
            this.lblnetcfg.Name = "lblnetcfg";
            this.lblnetcfg.Size = new System.Drawing.Size(0, 13);
            this.lblnetcfg.TabIndex = 30;
            // 
            // label10
            // 
            this.label10.AutoSize = true;
            this.label10.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(238)));
            this.label10.Location = new System.Drawing.Point(9, 243);
            this.label10.Name = "label10";
            this.label10.Size = new System.Drawing.Size(82, 13);
            this.label10.TabIndex = 29;
            this.label10.Text = "Network config:";
            // 
            // label9
            // 
            this.label9.AutoSize = true;
            this.label9.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(238)));
            this.label9.Location = new System.Drawing.Point(12, 194);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(69, 13);
            this.label9.TabIndex = 28;
            this.label9.Text = "Cost function";
            // 
            // comboCostFunction
            // 
            this.comboCostFunction.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboCostFunction.FormattingEnabled = true;
            this.comboCostFunction.Items.AddRange(new object[] {
            "Mean Squared",
            "Cross Entropy"});
            this.comboCostFunction.Location = new System.Drawing.Point(151, 191);
            this.comboCostFunction.Name = "comboCostFunction";
            this.comboCostFunction.Size = new System.Drawing.Size(115, 21);
            this.comboCostFunction.TabIndex = 27;
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(238)));
            this.label8.Location = new System.Drawing.Point(12, 171);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(88, 13);
            this.label8.TabIndex = 26;
            this.label8.Text = "Shuffle minibatch";
            // 
            // checkShuffle
            // 
            this.checkShuffle.AutoSize = true;
            this.checkShuffle.Checked = true;
            this.checkShuffle.CheckState = System.Windows.Forms.CheckState.Checked;
            this.checkShuffle.Location = new System.Drawing.Point(151, 171);
            this.checkShuffle.Name = "checkShuffle";
            this.checkShuffle.Size = new System.Drawing.Size(15, 14);
            this.checkShuffle.TabIndex = 25;
            this.checkShuffle.UseVisualStyleBackColor = true;
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(238)));
            this.label7.Location = new System.Drawing.Point(12, 121);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(74, 13);
            this.label7.TabIndex = 24;
            this.label7.Text = "Regularization";
            // 
            // comboRegularization
            // 
            this.comboRegularization.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboRegularization.FormattingEnabled = true;
            this.comboRegularization.Items.AddRange(new object[] {
            "None",
            "L1",
            "L2"});
            this.comboRegularization.Location = new System.Drawing.Point(151, 118);
            this.comboRegularization.Name = "comboRegularization";
            this.comboRegularization.Size = new System.Drawing.Size(115, 21);
            this.comboRegularization.TabIndex = 23;
            // 
            // numLambda
            // 
            this.numLambda.DecimalPlaces = 6;
            this.numLambda.Increment = new decimal(new int[] {
            1,
            0,
            0,
            131072});
            this.numLambda.Location = new System.Drawing.Point(151, 145);
            this.numLambda.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.numLambda.Name = "numLambda";
            this.numLambda.Size = new System.Drawing.Size(115, 20);
            this.numLambda.TabIndex = 22;
            this.numLambda.Value = new decimal(new int[] {
            1,
            0,
            0,
            0});
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(238)));
            this.label6.Location = new System.Drawing.Point(12, 147);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(111, 13);
            this.label6.TabIndex = 21;
            this.label6.Text = "Regularization lambda";
            // 
            // numLearningRate
            // 
            this.numLearningRate.DecimalPlaces = 6;
            this.numLearningRate.Increment = new decimal(new int[] {
            1,
            0,
            0,
            131072});
            this.numLearningRate.Location = new System.Drawing.Point(151, 92);
            this.numLearningRate.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.numLearningRate.Name = "numLearningRate";
            this.numLearningRate.Size = new System.Drawing.Size(115, 20);
            this.numLearningRate.TabIndex = 20;
            this.numLearningRate.Value = new decimal(new int[] {
            25,
            0,
            0,
            131072});
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(238)));
            this.label5.Location = new System.Drawing.Point(12, 94);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(69, 13);
            this.label5.TabIndex = 19;
            this.label5.Text = "Learning rate";
            // 
            // numMiniBatchSize
            // 
            this.numMiniBatchSize.Location = new System.Drawing.Point(151, 66);
            this.numMiniBatchSize.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.numMiniBatchSize.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numMiniBatchSize.Name = "numMiniBatchSize";
            this.numMiniBatchSize.Size = new System.Drawing.Size(115, 20);
            this.numMiniBatchSize.TabIndex = 18;
            this.numMiniBatchSize.Value = new decimal(new int[] {
            250,
            0,
            0,
            0});
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(238)));
            this.label4.Location = new System.Drawing.Point(12, 68);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(77, 13);
            this.label4.TabIndex = 17;
            this.label4.Text = "Mini batch size";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(238)));
            this.label3.Location = new System.Drawing.Point(12, 42);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(43, 13);
            this.label3.TabIndex = 16;
            this.label3.Text = "Epochs";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label2.Location = new System.Drawing.Point(12, 11);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(129, 16);
            this.label2.TabIndex = 15;
            this.label2.Text = "Training parameters";
            // 
            // Main
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1029, 325);
            this.Controls.Add(this.panel1);
            this.Controls.Add(this.numericUpDown2);
            this.Controls.Add(this.button8);
            this.Controls.Add(this.button7);
            this.Controls.Add(this.button6);
            this.Controls.Add(this.comboBox1);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.lblResult);
            this.Controls.Add(this.button5);
            this.Controls.Add(this.button3);
            this.Controls.Add(this.button2);
            this.Controls.Add(this.button1);
            this.Controls.Add(this.pictureBox1);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
            this.MinimizeBox = false;
            this.Name = "Main";
            this.Text = "MNIST Number recognize";
            this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.Main_FormClosing);
            this.Load += new System.EventHandler(this.Form1_Load);
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numEpoch)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDown2)).EndInit();
            this.panel1.ResumeLayout(false);
            this.panel1.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.numLambda)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLearningRate)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numMiniBatchSize)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.PictureBox pictureBox1;
        private System.Windows.Forms.Button button1;
        private System.Windows.Forms.Button button2;
        private System.Windows.Forms.Button button3;
        private System.Windows.Forms.Button button4;
        private System.Windows.Forms.Button button5;
        private System.Windows.Forms.Label lblResult;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.ComboBox comboBox1;
        private System.Windows.Forms.Button button6;
        private System.Windows.Forms.OpenFileDialog openFileDialog1;
        private System.Windows.Forms.SaveFileDialog saveFileDialog1;
        private System.Windows.Forms.NumericUpDown numEpoch;
        private System.Windows.Forms.Button button7;
        private System.Windows.Forms.Button button8;
        private System.Windows.Forms.NumericUpDown numericUpDown2;
        private System.Windows.Forms.Panel panel1;
        private System.Windows.Forms.NumericUpDown numMiniBatchSize;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.NumericUpDown numLearningRate;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.ComboBox comboRegularization;
        private System.Windows.Forms.NumericUpDown numLambda;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.CheckBox checkShuffle;
        private System.Windows.Forms.Label label9;
        private System.Windows.Forms.ComboBox comboCostFunction;
        private System.Windows.Forms.Label lblnetcfg;
        private System.Windows.Forms.Label label10;
    }
}

