using System;

namespace Macademy.OpenCL
{
    internal class CLSourceProvider
    {
        public static string ReadSourceFile()
        {
            return Properties.Resources.OpenCLKernels;
        }
    }
}
