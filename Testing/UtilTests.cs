using Macademy;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace UtilTests
{
    [TestClass]
    public class UtilTests
    {
        [TestMethod]
        public void TestListShuffle()
        {
            int shuffledLists = 0;

            for (int i = 0; i < 1000; i++)
            {
                List<int> testData = new List<int>();

                testData.Add(0);
                testData.Add(1);
                testData.Add(2);
                testData.Add(3);
                testData.Add(4);
                testData.Add(5);
                testData.Add(6);
                testData.Add(7);
                testData.Add(8);
                testData.Add(9);

                Utils.ShuffleList(ref testData);

                Assert.AreEqual(10, testData.Count);

                int shuffledElements = 0;

                for (int j = 0; j < testData.Count; j++)
                {
                    if (testData[j] != j)
                        ++shuffledElements;
                }

                if (shuffledElements > 3)
                    ++shuffledLists;
            }

            if (shuffledLists <= 950)
                Assert.Fail("Expected to shuffle lists at least 9 out of 10");
        }

        [TestMethod]
        public void TestSign()
        {
            Assert.AreEqual(-1.0f, Utils.Sign(-1000.0f));
            Assert.AreEqual(-1.0f, Utils.Sign(-5.0f));
            Assert.AreEqual(-1.0f, Utils.Sign(-1.0f));
            Assert.AreEqual(-1.0f, Utils.Sign(-0.3f));

            Assert.AreEqual(0.0f, Utils.Sign(0.0f));

            Assert.AreEqual(1.0f, Utils.Sign(0.4f));
            Assert.AreEqual(1.0f, Utils.Sign(1.0f));
            Assert.AreEqual(1.0f, Utils.Sign(10.0f));
            Assert.AreEqual(1.0f, Utils.Sign(1000.0f));
        }
    }
}
