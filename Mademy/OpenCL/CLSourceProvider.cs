using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace Macademy.OpenCL
{
    internal class CLSourceProvider
    {
        public static string ReadSourceFile()
        {
            return Properties.Resources.CLKernel;
        }
    }
}
