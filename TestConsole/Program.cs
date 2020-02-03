using Macademy;
using Macademy.OpenCL;
using System;

namespace TestConsole
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine(" ### macademy test console ");
            ComputeDevice selectedDevice = ComputeDeviceUtil.CreateFallbackComputeDevice();

            while (true)
            {
                Console.WriteLine("");
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
                        Console.WriteLine("help       - Displays this help message");
                        Console.WriteLine("devices    - Displays available devices");
                        Console.WriteLine("select (n) - Selectes the devices with the given id");
                        Console.WriteLine("info       - Displays information about the selected device");
                        Console.WriteLine("test       - Performs a quick test on the selected device");
                        Console.WriteLine("exit       - Exits the app");
                    }
                    else if (nextCommand == "devices")
                    {
                        var devices = ComputeDeviceUtil.GetComputeDevices();
                        System.Console.WriteLine(String.Format("Found a total of {0} devices!", devices.Count));
                        int i = 0;
                        foreach (var dev in devices)
                        {
                            Console.WriteLine(String.Format((i++).ToString() + ": [{0}] {1}", dev.GetDeviceAccessType(), dev.GetDeviceName() ));
                        }
                    }
                    else if (nextCommand.StartsWith("select"))
                    {
                        if (commands.Length >= 2)
                        {
                            var devices = ComputeDeviceUtil.GetComputeDevices();

                            int selectedDeviceId = 0;
                            if (int.TryParse(commands[1], out selectedDeviceId))
                            {
                                if (selectedDeviceId < 0 || selectedDeviceId >= devices.Count)
                                {
                                    Console.WriteLine("No such device: " + selectedDeviceId);
                                    continue;
                                }

                                selectedDevice = ComputeDeviceUtil.CreateComputeDevice( devices[selectedDeviceId] );
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
                    else if (nextCommand == "test")
                    {
                        Console.WriteLine("Testing on device: " + selectedDevice.GetName() );
                        TestDevice(selectedDevice);
                    }
                    else if (nextCommand == "info")
                    {
                        if (selectedDevice != null)
                        {
                            Console.WriteLine("Device Name: " + selectedDevice.GetName());
                            Console.WriteLine("Device Access: " + selectedDevice.GetDeviceAccessMode());
                            Console.WriteLine("Core count: " + selectedDevice.GetDeviceCoreCount());
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

        private static void TestDevice(ComputeDevice selectedDevice)
        {
            int[] referenceLayerConf = new int[] { 3, 7, 5, 4 };

            var network = Network.CreateNetworkInitRandom(referenceLayerConf, new SigmoidActivation());

            int errorCount = 0;
            CheckResults(3, network.GetLayerConfig()[0], () => { ++errorCount; } );
            CheckResults(7, network.GetLayerConfig()[1], () => { ++errorCount; } );
            CheckResults(5, network.GetLayerConfig()[2], () => { ++errorCount; } );
            CheckResults(4, network.GetLayerConfig()[3], () => { ++errorCount; } );

            float[] result = network.Compute(new float[] { 0.2f, 0.4f, 0.5f }, selectedDevice);
            CheckResults(referenceLayerConf[referenceLayerConf.Length - 1], result.Length, () => { ++errorCount; });
            Console.WriteLine("Test finished with " + errorCount + " error(s)!");
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
