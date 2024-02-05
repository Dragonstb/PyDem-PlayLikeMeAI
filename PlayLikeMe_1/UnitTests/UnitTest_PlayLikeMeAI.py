import sys
sys.path.append('..')
sys.path.append('../../TexasPydEm_Main/TexasPydEm')
# autopep8: off
import unittest as UT
from PlayLikeMe_1 import PlayLikeMeAI as AI
from PlayLikeMe_1.PlayLikeMeAI import PlayLikeMeAI as AIP
from Player import Player
import numpy as np
# autopep8: on

# execute from console by cd-ing to directory PlayLikeMe_1 and enter
# python3 -m unittest UnitTests/UnitTest_PlayLikeMeAI.py


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

    def test_notifyElimination_observeAheadEliminated(self):
        names = ['Alice', 'Bob', 'Clara', 'Dylan', 'Eve', 'Freddy']
        players = [Player(name) for name in names]
        numPlr = len(players)
        delIdx = 2
        delPlayer = players[delIdx]

        idolIdx = 1
        idol = players[idolIdx]
        ai = AIP('Zoe', idol=idol)
        ai.setPlayers(players)
        # set some well defined values
        ai._playerInput[:] = 0
        ai._playerInput[:numPlr] += np.arange(numPlr).reshape(numPlr, 1) + 1
        # assert beforehand
        self.assertEqual(
            1, np.min(ai._playerInput[0]), "too low values in row 0")
        self.assertEqual(
            1, np.max(ai._playerInput[0]), "too high values in row 0")
        self.assertEqual(
            2, np.min(ai._playerInput[1]), "too low values in row 1")
        self.assertEqual(
            2, np.max(ai._playerInput[1]), "too high values in row 1")
        self.assertEqual(
            3, np.min(ai._playerInput[2]), "too low values in row 2")
        self.assertEqual(
            3, np.max(ai._playerInput[2]), "too high values in row 2")
        self.assertEqual(
            4, np.min(ai._playerInput[3]), "too low values in row 3")
        self.assertEqual(
            4, np.max(ai._playerInput[3]), "too high values in row 3")
        self.assertEqual(
            5, np.min(ai._playerInput[4]), "too low values in row 4")
        self.assertEqual(
            5, np.max(ai._playerInput[4]), "too high values in row 4")
        self.assertEqual(
            6, np.min(ai._playerInput[5]), "too low values in row 5")
        self.assertEqual(
            6, np.max(ai._playerInput[5]), "too high values in row 5")

        self.assertEqual(numPlr, len(ai._players),
                         "unexpected number of players listed")
        self.assertIn(delPlayer, ai._players, "elimnated player still list")
        # delete and notify, in that order, like in the game
        del players[delIdx]
        ai.notifyElimination(delPlayer)
        # assert ai's values
        # player inputs without column _SEATS_AHEAD_OFFSET, as this column is set within the method
        indices = np.arange(ai._playerInput.shape[1])[np.arange(
            ai._playerInput.shape[1]) != AI._SEATS_AHEAD_OFFSET]
        self.assertEqual(
            1, np.min(ai._playerInput[0, indices]), "too low values in row 0")
        self.assertEqual(
            1, np.max(ai._playerInput[0, indices]), "too high values in row 0")
        self.assertEqual(
            2, np.min(ai._playerInput[1, indices]), "too low values in row 1")
        self.assertEqual(
            2, np.max(ai._playerInput[1, indices]), "too high values in row 1")
        self.assertEqual(
            4, np.min(ai._playerInput[2, indices]), "too low values in row 2")
        self.assertEqual(
            4, np.max(ai._playerInput[2, indices]), "too high values in row 2")
        self.assertEqual(
            5, np.min(ai._playerInput[3, indices]), "too low values in row 3")
        self.assertEqual(
            5, np.max(ai._playerInput[3, indices]), "too high values in row 3")
        self.assertEqual(
            6, np.min(ai._playerInput[4, indices]), "too low values in row 4")
        self.assertEqual(
            6, np.max(ai._playerInput[4, indices]), "too high values in row 4")
        self.assertEqual(
            0, np.min(ai._playerInput[5, indices]), "too low values in row 5")
        self.assertEqual(
            0, np.max(ai._playerInput[5, indices]), "too high values in row 5")

        self.assertEqual(numPlr-1, len(ai._players),
                         "unexpected number of players listed")
        self.assertNotIn(delPlayer, ai._players, "elimnated player still list")

    def test_notifyElimination_observeBehindEliminated(self):
        names = ['Alice', 'Bob', 'Clara', 'Dylan', 'Eve', 'Freddy']
        players = [Player(name) for name in names]
        numPlr = len(players)
        delIdx = 2
        delPlayer = players[delIdx]

        idolIdx = 4
        idol = players[idolIdx]
        ai = AIP('Zoe', idol=idol)
        ai.setPlayers(players)
        # set some well defined values
        ai._playerInput[:] = 0
        ai._playerInput[:numPlr] += np.arange(numPlr).reshape(numPlr, 1) + 1
        # assert beforehand
        self.assertEqual(
            1, np.min(ai._playerInput[0]), "too low values in row 0")
        self.assertEqual(
            1, np.max(ai._playerInput[0]), "too high values in row 0")
        self.assertEqual(
            2, np.min(ai._playerInput[1]), "too low values in row 1")
        self.assertEqual(
            2, np.max(ai._playerInput[1]), "too high values in row 1")
        self.assertEqual(
            3, np.min(ai._playerInput[2]), "too low values in row 2")
        self.assertEqual(
            3, np.max(ai._playerInput[2]), "too high values in row 2")
        self.assertEqual(
            4, np.min(ai._playerInput[3]), "too low values in row 3")
        self.assertEqual(
            4, np.max(ai._playerInput[3]), "too high values in row 3")
        self.assertEqual(
            5, np.min(ai._playerInput[4]), "too low values in row 4")
        self.assertEqual(
            5, np.max(ai._playerInput[4]), "too high values in row 4")
        self.assertEqual(
            6, np.min(ai._playerInput[5]), "too low values in row 5")
        self.assertEqual(
            6, np.max(ai._playerInput[5]), "too high values in row 5")

        self.assertEqual(numPlr, len(ai._players),
                         "unexpected number of players listed")
        self.assertIn(delPlayer, ai._players, "elimnated player still list")
        # delete and notify, in that order, like in the game
        del players[delIdx]
        ai.notifyElimination(delPlayer)
        # assert ai's values
        # player inputs without column _SEATS_AHEAD_OFFSET, as this column is set within the method
        indices = np.arange(ai._playerInput.shape[1])[np.arange(
            ai._playerInput.shape[1]) != AI._SEATS_AHEAD_OFFSET]
        self.assertEqual(
            1, np.min(ai._playerInput[0, indices]), "too low values in row 0")
        self.assertEqual(
            1, np.max(ai._playerInput[0, indices]), "too high values in row 0")
        self.assertEqual(
            2, np.min(ai._playerInput[1, indices]), "too low values in row 1")
        self.assertEqual(
            2, np.max(ai._playerInput[1, indices]), "too high values in row 1")
        self.assertEqual(
            4, np.min(ai._playerInput[2, indices]), "too low values in row 2")
        self.assertEqual(
            4, np.max(ai._playerInput[2, indices]), "too high values in row 2")
        self.assertEqual(
            5, np.min(ai._playerInput[3, indices]), "too low values in row 3")
        self.assertEqual(
            5, np.max(ai._playerInput[3, indices]), "too high values in row 3")
        self.assertEqual(
            6, np.min(ai._playerInput[4, indices]), "too low values in row 4")
        self.assertEqual(
            6, np.max(ai._playerInput[4, indices]), "too high values in row 4")
        self.assertEqual(
            0, np.min(ai._playerInput[5, indices]), "too low values in row 5")
        self.assertEqual(
            0, np.max(ai._playerInput[5, indices]), "too high values in row 5")

        self.assertEqual(numPlr-1, len(ai._players),
                         "unexpected number of players listed")
        self.assertNotIn(delPlayer, ai._players, "elimnated player still list")

        self.assertEqual(idolIdx-1, ai._myIndex, "Idol at unexpected index")

    def test_revealAllCards_twoCards(self):
        names = ['Alice', 'Bob', 'Clara', 'Dylan', 'Eve', 'Freddy']
        players = [Player(name) for name in names]
        sampleIdx = 3
        sample = players[sampleIdx]
        # set my idol
        idolIdx = 4
        idol = players[idolIdx]
        ai = AIP('Zoe', idol=idol)
        ai.setPlayers(players)
        # set some well defined values
        ai._playerInput[:] = 0
        # hand out pocket cards to sample
        sample.pockets = [21, 33]

        ai.revealAllCards(sample)
        hots = np.sum(ai._playerInput[:len(
            players), AI._POCKET_BEGIN:AI._POCKET_END], axis=1)
        expect = np.zeros(len(players))
        # two cards, each contributing unity for the suit and unity for the value
        expect[sampleIdx] = 4
        for idx in range(len(players)):
            self.assertEqual(
                expect[idx], hots[idx], "Wrong number of one hot neurons being unity at index "+str(idx))

    def test_revealAllCards_oneCard(self):
        names = ['Alice', 'Bob', 'Clara', 'Dylan', 'Eve', 'Freddy']
        players = [Player(name) for name in names]
        sampleIdx = 3
        sample = players[sampleIdx]
        # set my idol
        idolIdx = 4
        idol = players[idolIdx]
        ai = AIP('Zoe', idol=idol)
        ai.setPlayers(players)
        # set some well defined values
        ai._playerInput[:] = 0
        # hand out pocket cards to sample
        sample.pockets = [9]

        ai.revealAllCards(sample)
        hots = np.sum(ai._playerInput[:len(
            players), AI._POCKET_BEGIN:AI._POCKET_END], axis=1)
        expect = np.zeros(len(players))
        # one card contributing unity for the suit and unity for the value
        expect[sampleIdx] = 2
        for idx in range(len(players)):
            self.assertEqual(
                expect[idx], hots[idx], "Wrong number of one hot neurons being unity at index "+str(idx))
