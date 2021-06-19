import unittest
import cca

class returnNeighboringClassifiersTest(unittest.TestCase):
   def setUp(self):
      self.matrixTest = [
         ["AA", "AB", "AC", "AD", "AE", "AF"],
         ["BA", "BB", "BC", "BD", "BE", "BF"],
         ["CA", "CB", "CC", "CD", "CE", "CF"],
         ["DA", "DB", "DC", "DD", "DE", "DF"],
         ["EA", "EB", "EC", "ED", "EE", "EF"],
         ["FA", "FB", "FC", "FD", "FE", "FF"]
      ]

      self.neighbors = [
         {'name': 'vizinho1', 'score': 0.814, 'predict': [0, 0, 0]},
         {'name': 'vizinho2', 'score': 0.846, 'predict': [0, 0, 0]},
         {'name': 'vizinho3', 'score': 0.82, 'predict': [1, 1, 1]},
         {'name': 'vizinho4', 'score': 0.82, 'predict': [1, 1, 1]}
      ]

   def test_retorna_AB_BA_BB_DISTANCE1(self):
      neighbors = cca.returnNeighboringClassifiers(6, 6, 0, 0, 1, self.matrixTest)
      self.assertListEqual(["AB","BA","BB"], neighbors)

   def test_retorna_AA_AB_AC_BA_BC_CA_CB_CC_DISTANCE1(self):
      neighbors = cca.returnNeighboringClassifiers(6, 6, 1, 1, 1, self.matrixTest)
      self.assertListEqual(["AA","AB","AC","BA","BC","CA","CB","CC"], neighbors)

   def test_retorna_BC_BD_BE_CC_CE_DC_DD_DE_DISTANCE1(self):
      neighbors = cca.returnNeighboringClassifiers(6, 6, 2, 3, 1, self.matrixTest)
      self.assertListEqual(["BC","BD","BE","CC","CE","DC","DD","DE"], neighbors)

   def test_retorna_DD_DE_DF_ED_EF_FD_FE_FF_DISTANCE1(self):
      neighbors = cca.returnNeighboringClassifiers(6, 6, 4, 4, 1, self.matrixTest)
      self.assertListEqual(["DD","DE","DF","ED","EF","FD","FE","FF"], neighbors)

   def test_retorna_AB_AC_BA_BB_BC_CA_CB_CC_DISTANCE2(self):
      neighbors = cca.returnNeighboringClassifiers(6, 6, 0, 0, 2, self.matrixTest)
      self.assertListEqual(["AB","AC","BA","BB","BC","CA","CB","CC"], neighbors)

   def test_retorna_AA_AB_AC_AD_BA_BC_BD_CA_CB_CC_CD_DA_DB_DC_DD_DISTANCE2(self):
      neighbors = cca.returnNeighboringClassifiers(6, 6, 1, 1, 2, self.matrixTest)
      self.assertListEqual(["AA","AB","AC","AD","BA","BC","BD","CA","CB","CC","CD","DA","DB","DC","DD"], neighbors)

   def test_retorna_ABACADAEAF_BBBCBDBEBF_CBCCCECF_DBDCDDDEDF_EBECEDEEEF_DISTANCE2(self):
      neighbors = cca.returnNeighboringClassifiers(6, 6, 2, 3, 2, self.matrixTest)
      self.assertListEqual(["AB","AC","AD","AE","AF","BB","BC","BD","BE","BF","CB","CC","CE","CF","DB","DC","DD","DE","DF","EB","EC","ED","EE","EF"], neighbors)

   def test_retorna_CCCDCECF_DCDDDEDF_ECEDEF_FCFDFEFF_DISTANCE2(self):
      neighbors = cca.returnNeighboringClassifiers(6, 6, 4, 4, 2, self.matrixTest)
      self.assertListEqual(["CC","CD","CE","CF","DC","DD","DE","DF","EC","ED","EF","FC","FD","FE","FF"], neighbors)

   def test_retorna_AAABAC_BABC_CACBCC_DISTANCE1(self):
      neighbors = cca.returnNeighboringClassifiers(4, 4, 1, 1, 1, self.matrixTest)
      self.assertListEqual(["AA","AB","AC","BA","BC","CA","CB","CC"], neighbors)

   def test_retorna_AAABACAD_BABCBD_CACBCCCD_DADBDCDDDISTANCE1(self):
      neighbors = cca.returnNeighboringClassifiers(4, 4, 1, 1, 2, self.matrixTest)
      self.assertListEqual(["AA","AB","AC","AD","BA","BC","BD","CA","CB","CC","CD","DA","DB","DC","DD"], neighbors)


if __name__ == "__main__":
   unittest.main()