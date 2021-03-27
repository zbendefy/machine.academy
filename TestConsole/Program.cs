using Macademy;
using Macademy.OpenCL;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;

namespace TestConsole
{
    class Program
    {
        static Network reference_network = null;
        static Network target_network = null;
        static Random random = new Random();
        
        static void Main(string[] args)
        {
            Console.WriteLine(" ### macademy test console ");
            ComputeDevice selectedDevice = ComputeDeviceFactory.CreateFallbackComputeDevice();

            Generate();

            while (true)
            {
                Console.WriteLine("");
                Console.Write("> ");
                string rawCommand = Console.ReadLine().Trim();
                var commands = rawCommand.Split(' ');
                if (commands.Length == 0)
                    continue;

                var nextCommand = commands[0];

                try
                {
                    if (nextCommand == "exit")
                    {
                        break;
                    }
                    else if (nextCommand == "help")
                    {
                        Console.WriteLine("General");
                        Console.WriteLine(" help          - Displays this help message");
                        Console.WriteLine(" exit          - Exits the app");
                        Console.WriteLine("");
                        Console.WriteLine("Device selection");
                        Console.WriteLine(" devices       - Displays available devices");
                        Console.WriteLine(" select (n)    - Selectes the devices with the given id");
                        Console.WriteLine(" info          - Displays information about the selected device");
                        Console.WriteLine(" quicktest     - Performs a quick test on the selected device");
                        Console.WriteLine(" benchmark [n] - Performs a benchmark on the selected device, on the given difficulty (1-10, default: 2)");
                        Console.WriteLine("");
                        Console.WriteLine("Model training and testing");
                        Console.WriteLine(" generate      - Generates a new network");
                        Console.WriteLine(" train         - Trains the network");
                        Console.WriteLine(" eval (i)      - Evaluates the output of the network to the given input");
                        Console.WriteLine(" export [f]    - Exports the current network to the filename provided (default: 'output.json')");
                        Console.WriteLine(" import [f]    - Imports the network from the filename provided (default: 'output.json')");
                    }
                    else if (nextCommand == "devices")
                    {
                        var devices = ComputeDeviceFactory.GetComputeDevices();
                        System.Console.WriteLine(String.Format("Found a total of {0} devices!", devices.Count));
                        int i = 0;
                        foreach (var dev in devices)
                        {
                            Console.WriteLine(String.Format((i++).ToString() + ": [{0}] {1}", dev.GetDeviceAccessType(), dev.GetDeviceName()));
                        }
                    }
                    else if (nextCommand.StartsWith("select"))
                    {
                        if (commands.Length >= 2)
                        {
                            var devices = ComputeDeviceFactory.GetComputeDevices();

                            int selectedDeviceId = 0;
                            if (int.TryParse(commands[1], out selectedDeviceId))
                            {
                                if (selectedDeviceId < 0 || selectedDeviceId >= devices.Count)
                                {
                                    Console.WriteLine("No such device: " + selectedDeviceId);
                                    continue;
                                }

                                selectedDevice = ComputeDeviceFactory.CreateComputeDevice(devices[selectedDeviceId]);
                                Console.WriteLine("Selected device: " + selectedDeviceId + ": " + selectedDevice.GetName());
                            }
                            else
                            {
                                Console.WriteLine("Invalid device id given!");
                            }
                        }
                        else
                        {
                            Console.WriteLine("No device id given!");
                        }
                    }
                    else if (nextCommand == "quicktest")
                    {
                        Console.WriteLine("Testing on device: " + selectedDevice.GetName());
                        TestDevice(selectedDevice);
                    }
                    else if (nextCommand == "train")
                    {
                        int epochs = 1;

                        if (commands.Length >= 2)
                        {
                            int.TryParse(commands[1], out epochs);
                        }

                        Train(selectedDevice, epochs);
                    }
                    else if (nextCommand == "eval")
                    {
                        if (commands.Length < 2)
                        {
                            Console.WriteLine("No input given!");
                            continue;
                        }

                        List<float> input_values = new List<float>();
                        for (int i = 1; i < commands.Length; ++i)
                        {
                            float input = 0;
                            if (!float.TryParse(commands[i], out input))
                            {
                                Console.WriteLine("Invalid input given at index {0}", (i-1));
                            }

                            input_values.Add(input);
                        }

                        float[] result = Eval(input_values.ToArray(), selectedDevice);

                        Console.WriteLine("Network output: [" + string.Join(", ", result) + "]");
                    }
                    else if (nextCommand == "generate")
                    {
                        Generate();
                        Console.WriteLine("A new network has been generated!");
                    }
                    else if (nextCommand == "benchmark")
                    {
                        int level = 2;

                        if (commands.Length >= 2)
                        {
                            try
                            {
                                level = Math.Max(Math.Min(10, Int32.Parse(commands[1])), 1);
                            }
                            catch (System.Exception)
                            {
                            }
                        }

                        BenchmarkDevice(selectedDevice, level);
                    }
                    else if (nextCommand == "export")
                    {
                        string filename = "output.json";

                        if (commands.Length >= 2)
                        {
                            filename = commands[1];
                        }

                        try
                        {
                            System.IO.File.WriteAllText(filename, target_network.ExportToJSON());
                            Console.WriteLine("Model exported to: '{0}'", filename);
                        }
                        catch (Exception exc)
                        {
                            Console.WriteLine("Error: " + exc.Message);
                        }
                    }
                    else if (nextCommand == "import")
                    {
                        string filename = "output.json";

                        if (commands.Length >= 2)
                        {
                            filename = commands[1];
                        }

                        try
                        {
                            string json_network = System.IO.File.ReadAllText(filename);
                            target_network = Network.CreateNetworkFromJSON(json_network);
                            Console.WriteLine("Model imported from: '{0}'", filename);
                        }
                        catch (Exception exc)
                        {
                            Console.WriteLine("Error: " + exc.Message);
                        }
                    }
                    else if (nextCommand == "info")
                    {
                        if (selectedDevice != null)
                        {
                            Console.WriteLine("Device Name: " + selectedDevice.GetName());
                            Console.WriteLine("Device Access: " + selectedDevice.GetDeviceAccessMode());
                            Console.WriteLine("Core count: " + selectedDevice.GetDeviceCoreCount());
                            Console.WriteLine("Memory: " + selectedDevice.GetDeviceMemorySize());
                        }
                        else
                        {
                            Console.WriteLine("CPU Fallback device is selected!");
                        }
                    }
                }
                catch (System.Exception exc)
                {
                    Console.WriteLine("An error occured when running the command! " + exc.ToString());
                }
            }
        }

        private static float[] Eval(float[] input, ComputeDevice device)
        {
            var result = target_network.Compute(input, device);
            var transformed_result = result[0] * 2.0f - 1.0f;
            var expected_result = (float)Math.Sin((input[0] - 0.5f) * 4.0f);

            Console.WriteLine("Output is:           " + transformed_result);
            Console.WriteLine("Expected result is:  " + expected_result);
            Console.WriteLine("Calculated error:    " + (expected_result - transformed_result));

            return result;
        }

        private static void Generate()
        {
            target_network = Network.CreateNetworkInitRandom(new int[]{1,128,64,1}, new SigmoidActivation());
        }

        private static void TestDevice(ComputeDevice selectedDevice)
        {
            int[] referenceLayerConf = new int[] { 3, 7, 5, 4 };

            if(reference_network == null)
            {
                reference_network = Network.CreateNetworkInitRandom(referenceLayerConf, new SigmoidActivation());
            }

            int errorCount = 0;
            CheckResults(3, reference_network.GetLayerConfig()[0], () => { ++errorCount; } );
            CheckResults(7, reference_network.GetLayerConfig()[1], () => { ++errorCount; } );
            CheckResults(5, reference_network.GetLayerConfig()[2], () => { ++errorCount; } );
            CheckResults(4, reference_network.GetLayerConfig()[3], () => { ++errorCount; } );

            float[] result = reference_network.Compute(new float[] { 0.2f, 0.4f, 0.5f }, selectedDevice);
            CheckResults(referenceLayerConf[referenceLayerConf.Length - 1], result.Length, () => { ++errorCount; });
            Console.WriteLine("Test finished with " + errorCount + " error(s)!");
            Console.WriteLine("Result: " + String.Join(", ", result));
        }

        private static void Train(ComputeDevice selectedDevice, int epochs)
        {
            List<TrainingSuite.TrainingData> trainingData = new List<TrainingSuite.TrainingData>();
            for (int i = 0; i < 10000; i++)
            {
                float[] input = new float[1];
                float[] desiredOutput = new float[1];

                float rnd = (float)random.NextDouble();
                input[0] = rnd;
                desiredOutput[0] = (float)Math.Sin((rnd - 0.5f) * 4.0f) * 0.5f + 0.5f;

                trainingData.Add(new TrainingSuite.TrainingData(input, desiredOutput));
            }

            TrainingSuite suite = new TrainingSuite(trainingData);
            suite.config.epochs = epochs;
            suite.config.shuffleTrainingData = true;
            suite.config.miniBatchSize = 100;

            suite.config.costFunction = new CrossEntropyErrorFunction();
            suite.config.regularization = TrainingConfig.Regularization.L2;
            suite.config.regularizationLambda = 0.01f;
            suite.config.learningRate = 0.01f;

            Console.WriteLine("Running training for {0} epochs!",epochs);
            Stopwatch sw = Stopwatch.StartNew();

            int progress = 0;

            var promise = target_network.Train(suite, ComputeDeviceFactory.CreateFallbackComputeDevice());

            Console.WriteLine("____________________");

            while (!promise.IsReady())
            {
                int progress_rounded = (int)(promise.GetTotalProgress() * 20);
                if (progress_rounded > progress)
                {
                    ++progress;
                    Console.Write("#");
                }
                Thread.Sleep(50);
            }

            sw.Stop();
            Console.WriteLine("#");
            Console.WriteLine("Training finished! Elapsed={0}ms",sw.Elapsed.TotalMilliseconds);
        }

        private static void BenchmarkDevice(ComputeDevice selectedDevice, int level)
        {
            int difficulty = (int)Math.Pow(2, 5+level);
            Console.WriteLine("Preparing data...");
            int[] referenceLayerConf = new int[] { 3, difficulty, difficulty, difficulty, difficulty, difficulty, difficulty, 16 };
            var reference_benchmark_network = Network.CreateNetworkInitRandom(referenceLayerConf, new SigmoidActivation());
            
            Console.WriteLine("Starting benchmark on " + selectedDevice.GetName() + ", difficulty=" + level + " (" + difficulty + ")" );
            Stopwatch sw = Stopwatch.StartNew();
            float[] result = reference_benchmark_network.Compute(new float[] { 0.2f, 0.4f, 0.5f }, selectedDevice);
            sw.Stop();

            Console.WriteLine("Elapsed={0}ms",sw.Elapsed.TotalMilliseconds);
        }

        private static void CheckResults(int expected, int actual, Action onError )
        {
            if (expected != actual)
                onError();
        }

        private static void CheckResults(float[] expected, float[] actual, Action onError)
        {
            if (expected.Length != actual.Length)
            {
                onError();
                return;
            }

            float errMargin = 0.001f;

            for (int i = 0; i < expected.Length; i++)
            {
                if (Math.Abs(expected[i] - actual[i]) > errMargin)
                    onError();
            }
        }
    }
}
