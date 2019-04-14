using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Mademy
{
    public class Solver
    {
        Network network;

        public Solver(Network network)
        {
            this.network = network;
        }

        public List<float> Solve(List<float> input)
        {
            return network.Compute(input);
        }
    }
}
