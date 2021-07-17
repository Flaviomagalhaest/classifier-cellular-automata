import unittest
import pso

class initPopulationTest(unittest.TestCase):
   def test_initializePopulation(self):
      pop = pso.initPopulation(10, {})
      self.assertTrue(len(pop) == 10)


if __name__ == "__main__":
   unittest.main()