using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Mademy
{
    public abstract class IErrorFunction
    {
    }

    public class MeanSquaredError : IErrorFunction
    {
    }

    public class CrossEntropy : IErrorFunction
    {
    }
}
