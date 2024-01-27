from Player import Player
from tensorflow.keras import Model as TFM
import Model
import numpy as np
# from numpy.typing import ArrayLike
import numpy.typing as npt
from typing import List
import CardUtils as CA

# array offsets of player specific values
# TODO: deck agnostic
_POCKET_BEGIN = 0
_POCKET_SUIT_OFFSETS = [k*17 + _POCKET_BEGIN for k in range(2)]
_POCKET_VALUE_OFFSETS = [off+4 for off in _POCKET_SUIT_OFFSETS]
_POCKET_END = _POCKET_VALUE_OFFSETS[len(_POCKET_VALUE_OFFSETS)-1] + 13
_STACK_OFFSET = _POCKET_END
_BET_OFFSET = _STACK_OFFSET + 1
_SEATS_AHEAD_OFFSET = _BET_OFFSET + 1
_POSITION_OFFSET = _SEATS_AHEAD_OFFSET + 1
_ACTIVE_OFFSET = _POSITION_OFFSET + 1
_SLOT_USED_OFFSET = _ACTIVE_OFFSET + 1

# array offset of community values
_POT_SIZE_OFFSET = 0
_CUR_BET_OFFSET = _POT_SIZE_OFFSET + 1
_SMALL_BLIND_OFFSET = _CUR_BET_OFFSET + 1
_BIG_BLIND_OFFSET = _SMALL_BLIND_OFFSET + 1
_COM_SUIT_OFFSETS = [k*17+_BIG_BLIND_OFFSET+1 for k in range(5)]
_COM_VALUE_OFFSETS = [off+4 for off in _COM_SUIT_OFFSETS]

# number of community inputs
_NUM_COM_INPUTS = _COM_VALUE_OFFSETS[len(_COM_VALUE_OFFSETS)-1] + 12

_MAX_PLAYERS = 10

# _______________ utility functions _______________


def makeIndexCounterArray(refIdx: int, slots: int, length: int, inc: bool = False) -> List:
    """
    Computes an array with decreasing (increasing) integers from 'slots' through 0.
    The 0 becomes placed at the index 'refIdx'. If this is not the last index,
    the next index will have the value 'slots' (or 1, respectively).
    Afterwards, the list is extended with -1 until it has a length og 'length'.
    refIdx:
    Index that will have the value of 0.
    inc (default: False):
    Increase rather than decrease the values with increasing indices.
    """
    factor = 1
    if inc:
        factor = -1
    list = [((refIdx-idx) * factor) % slots for idx in range(slots)]
    if slots < length:
        list += [-1 for _ in range(length-slots)]
    return list

# _______________ class definition _______________


class PlayLikeMeAI(Player):

    _players: List[Player]     # list of players
    _myIndex: int              # my idol's / my own index in _players
    _idol: Player              # the player observed and trained from
    _model: TFM                # the tensorflow model backing the AI
    # input values that result from the players
    _playerInput: npt.NDArray[np.float_]
    # input values that result from the community
    _comInput: npt.NDArray[np.float_]

    def __init__(self, name, idol: Player = None, model: TFM = None, printModel: bool = False):
        super().__init__(name)
        self._idol = idol
        if model is not None:
            self._model = model
        else:
            self._model = Model.setupNewModel(printModel)

        self._playerInput = np.zeros((_MAX_PLAYERS, 40), dtype=np.float_)
        self._comInput = np.zeros(_NUM_COM_INPUTS, dtype=np.float_)

    # _______________ utility functions _______________

    def setPlayersBalance(self, player) -> None:
        """
        Puts the player's bet and stack values into the input array.

        player:
        Player of interest. Can be any player.
        """
        self._playerInput[self._players.index(
            player), _BET_OFFSET] = player.bet
        self._playerInput[self._players.index(
            player), _STACK_OFFSET] = player.stack

    # _______________ start a game _______________

    def setPlayers(self, players):
        self._players = players
        super().setPlayers(players)
        if self._idol is None:
            self._myIndex = players.index(self)
        else:
            self._myIndex = players.index(self._idol)

        ahead = makeIndexCounterArray(
            self._myIndex, len(players), _MAX_PLAYERS)
        self._playerInput[:, _SEATS_AHEAD_OFFSET] = ahead
        self._playerInput[0:len(players), _SLOT_USED_OFFSET] = 1

        stacks = [pl.stack for pl in players]
        self._playerInput[0: len(players), _STACK_OFFSET] = stacks

    # def announceFirstDealer(self, player):
    #     pass

    # _______________ tell user agent what is going on _______________

    def notifyBeginOfHand(self, dealer):
        # positions relative to dealer
        dealIdx = self._players.index(dealer)
        positions = makeIndexCounterArray(
            dealIdx, len(self._players), _MAX_PLAYERS, True)
        self._playerInput[:, _POSITION_OFFSET] = positions
        # clear pocket cards
        self._playerInput[:, _POCKET_BEGIN:_POCKET_END] = 0
        # set all players as active
        self._playerInput[0:len(self._players), _ACTIVE_OFFSET] = 1
        # clear community values, values of sb and bb are set again every hand, so we can zero them, too.
        self._comInput[:] = 0

    def notifyCommunityCards(self, cards: List[int]):
        for idx in range(len(cards)):
            card = cards[idx]
            suit = CA.getCardSuit(card)
            value = CA.getCardValue(card)
            self._comInput[_COM_SUIT_OFFSETS[idx]+suit] = 1
            self._comInput[_COM_VALUE_OFFSETS[idx]+value] = 1

    def notifyCardDealing(self, player):
        if (self._idol is not None and self._idol == player) or player == self:
            cards = self.getPocketCards()
            for idx in range(len(cards)):
                card = cards[idx]
                suit = CA.getCardSuit(card)
                value = CA.getCardValue(card)
                self._pocketInput[self._my_index,
                                  _POCKET_SUIT_OFFSETS[idx]+suit] = 1
                self._pocketInput[self._my_index,
                                  _POCKET_VALUE_OFFSETS[idx]+value] = 1

    def notifySmallBlind(self, player):
        self._comInput[_SMALL_BLIND_OFFSET] = player.bet

    def notifyBigBlind(self, player):
        self._comInput[_BIG_BLIND_OFFSET] = player.bet

    def notifyFolding(self, player):
        self._playerInput[self._players.index(player), _ACTIVE_OFFSET] = 0

    def notifyCheck(self, player):
        self.setPlayersBalance(player)

    def notifyRaise(self, player):
        self.setPlayersBalance(player)

    def notifyCall(self, player):
        self.setPlayersBalance(player)

    def notifyLastPenny(self, player):
        self.setPlayersBalance(player)

    def notifyAllIn(self, player):
        self.setPlayersBalance(player)

    # def notifyPotWin(self, player, win: int):
    #     pass

    # def notifyShowdown(self):
    #     pass

    # def notifyElimination(self, player):
    #     pass

    # def notifyEndOfHand(self):
    #     pass

    # def revealAllCards(self, player):
    #     pass

    # _______________ other _______________

    # def debug(self):
    #     print(self.name)
    #     [cd.debug('  ') for pl, cd in self.comps.items()]
