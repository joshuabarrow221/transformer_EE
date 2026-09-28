"""Focused checks for the scientific failure modes of this workflow."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from topology import decode


def encode(p,plus=0,minus=0,zero=0):
    return f'3{p:02d}00{plus:02d}00{minus:02d}00{zero:02d}'


class TopologyTests(unittest.TestCase):
    def test_charge_sum_and_revised_overflow(self):
        self.assertEqual(decode(encode(2,1,1,0))[1:],(2,2,'2p2pi'))
        self.assertEqual(decode(encode(0,zero=1))[3],'0p1pi')
        for code in [encode(3),encode(1,1,1,1),encode(0,plus=3),encode(10)]:
            self.assertEqual(decode(code)[3],'NpNpi')
        for code in [encode(0),encode(0,zero=2)]:self.assertEqual(decode(code)[3],'Other')

    def test_precision_and_corruption(self):
        self.assertEqual(decode('301000000000001.0'),decode('3.01000000000001e14'))
        for value in ['nan','301000000000001.5','3010000000000','301100000000001',str(int(np.float32(301000000000001)))]:
            with self.assertRaises(ValueError):decode(value)
        # A rounded string can accidentally look like another valid topology.
        # No decoder can recover information that was already discarded.
        self.assertNotEqual(str(np.float32(301000000000001)), '301000000000001.0')

    def test_sampling_and_sv_decimal_alignment(self):
        from plot_tsne import sample_csv
        with tempfile.TemporaryDirectory() as tmp:
            paths=[]
            for i,t in enumerate(['Nu_Energy','Nu_Mom_X','Nu_Mom_Y','Nu_Mom_Z']):
                p=Path(tmp)/f'{i}.csv';paths.append(str(p))
                pd.DataFrame({'true_Topology':[encode(j%3,zero=1)+('.0' if i%2 else '') for j in range(120)],
                              'pred_'+t:np.arange(120)/10}).to_csv(p,index=False)
            frame,audit=sample_csv({'paths':paths},50,42)
            again,_=sample_csv({'paths':paths},50,42)
            self.assertEqual(len(frame),50);self.assertEqual(audit['total_rows'],120)
            self.assertTrue(np.array_equal(frame.source_row,again.source_row))
            self.assertGreater(frame.source_row.max(),50)
            first,_=sample_csv({'paths':paths},50,42,policy='first')
            np.testing.assert_array_equal(first.source_row,np.arange(50))
            for t in ['Nu_Energy','Nu_Mom_X','Nu_Mom_Y','Nu_Mom_Z']:
                np.testing.assert_allclose(frame['pred_'+t],frame.source_row/10)
            damaged=pd.read_csv(paths[-1],dtype={'true_Topology':str})
            damaged.loc[0,'true_Topology']=encode(7);damaged.to_csv(paths[-1],index=False)
            with self.assertRaises(ValueError):sample_csv({'paths':paths},50,42)

    def test_first_valid_skips_nonfinite_without_replacement(self):
        from plot_tsne import sample_csv, TARGETS
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'events.csv'
            values={'true_Topology':[encode(1)]*70}
            for target in TARGETS:values['pred_'+target]=np.arange(70,dtype=float)
            values['pred_Nu_Energy'][[0,12]]=np.nan
            pd.DataFrame(values).to_csv(path,index=False)
            frame,audit=sample_csv({'path':str(path)},50,42,policy='first')
            np.testing.assert_array_equal(frame.source_row,np.delete(np.arange(52),[0,12]))
            self.assertEqual(audit['invalid_rows'],2)
            with self.assertRaises(ValueError):sample_csv({'path':str(path)},69,42,policy='first')


if __name__=='__main__':unittest.main()
