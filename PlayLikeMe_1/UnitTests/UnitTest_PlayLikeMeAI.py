import sys
sys.path.append('..')
sys.path.append('../../TexasPydEm_Main/TexasPydEm')
# autopep8: off
import unittest as UT
from PlayLikeMe_1 import PlayLikeMeAI as AI
# autopep8: on


class UnitTest_PlayLikeMeAI(UT.TestCase):

    def test_makeIndexCounterArray_dec(self):
        expect = [3, 2, 1, 0, 6, 5, 4, -1, -1, -1]
        length = len(expect)
        slots = length - expect.count(-1)
        refIdx = expect.index(0)
        actual = AI.makeIndexCounterArray(refIdx, slots, length)
        self.assertEqual(len(expect), len(actual), 'array length is incorrect')
        for idx in range(len(expect)):
            self.assertEqual(expect[idx], actual[idx],
                             f'different value at index {idx}')

    def test_makeIndexCounterArray_inc(self):
        expect = [6, 7, 0, 1, 2, 3, 4, 5, -1, -1]
        length = len(expect)
        slots = length - expect.count(-1)
        refIdx = expect.index(0)
        actual = AI.makeIndexCounterArray(refIdx, slots, length, True)
        self.assertEqual(len(expect), len(actual), 'incorrect array length')
        for idx in range(len(expect)):
            self.assertEqual(expect[idx], actual[idx],
                             f'different value at index {idx}')
