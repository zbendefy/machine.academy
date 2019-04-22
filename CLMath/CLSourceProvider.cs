using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace CLMath
{
    class CLSourceProvider
    {
        public static string ReadSourceFile()
        {
            return CLMath.Properties.Resources.CLKernel;
        }
    }
}
