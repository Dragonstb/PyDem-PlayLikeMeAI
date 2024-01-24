import ImportMain
from Player import Player
from tensorflow.keras import Model as TFM
import Model
import numpy as np

# TODO: deck agnostic
_SUIT_1_OFFSET = 0
_VALUE_1_OFFSET = _SUIT_1_OFFSET + 4
_SUIT_2_OFFST = _VALUE_1_OFFSET + 13
_VALUE_2_OFFSET = _SUIT_2_OFFST + 4
_STACK_OFFSET = _VALUE_2_OFFSET + 13
_BET_OFFSET = _STACK_OFFSET + 1
_SEATS_AHEAD_OFFSET = _BET_OFFSET + 1
_POSITION_OFFSET = _SEATS_AHEAD_OFFSET + 1
_ACTIVE_OFFSET = _POSITION_OFFSET + 1
_SLOT_USED_OFFSET = _ACTIVE_OFFSET + 1

_MAX_PLAYERS = 10


class PlayLikeMeAI(Player):

    _idol: Player           # the player observed and trained from
    _model: TFM             # the tensorflow model backing the AI
    _playerInput: np.array  # input values that results from the players

    def __init__(self, name, idol: Player = None, model: TFM = None, printModel: bool = False):
        super().__init__(name)
        self._idol = idol
        if model is not None:
            self._model = model
        else:
            self._model = Model.setupNewModel(printModel)

        self._playerInput = np.zeros((_MAX_PLAYERS, 40))

    # _______________ start a game _______________

    def setPlayers(self, players):
        super().setPlayers(players)
        if self._idol is None:
            myIndex = players.index(self)
        else:
            myIndex = players.index(self._idol)

        ahead = [(myIndex-idx) % len(players)
                 for idx in range(len(players))]
        if len(players) < _MAX_PLAYERS:
            ahead += [-1 for _ in range(_MAX_PLAYERS-len(players))]
        self._playerInput[:, _SEATS_AHEAD_OFFSET] = ahead

    # def announceFirstDealer(self, player):
    #     pass

    # _______________ tell user agent what is going on _______________

    # def notifyBeginOfHand(self, dealer):
    #     pass

    # def notifyCardDealing(self, player):
    #     pass

    # def notifySmallBlind(self, player):
    #     pass

    # def notifyBigBlind(self, player):
    #     pass

    # def notifyFolding(seld, player):
    #     pass

    # def notifyCheck(self, player):
    #     pass

    # def notifyRaise(self, player):
    #     pass

    # def notifyCall(self, player):
    #     pass

    # def notifyLastPenny(self, player):
    #     pass

    # def notifyAllIn(self, player):
    #     pass

    # def notifyPotWin(self, player, win: int):
    #     pass

    # def notifyCommunityCards(self, cards: List[int]):
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

    # def isHuman(self):
    #     """
    #     Does this instance represent a human player or any kind of bot?

    #     returns:
    #     True for humans and aliens and super-intelligent, poker-playing cats and alike, and False for bots.
    #     """
    #     return False

    # def debug(self):
    #     print(self.name)
    #     [cd.debug('  ') for pl, cd in self.comps.items()]
