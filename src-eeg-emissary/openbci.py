from brainflow import BrainFlowInputParams, BoardShim, BoardIds, BrainFlowPresets


class CytonDaisy:
    def __init__(self, serial_port="COM17"):
        self.params = BrainFlowInputParams()
        self.params.serial_port = serial_port
        self.board_preset = BrainFlowPresets.DEFAULT_PRESET
        self.board: BoardShim = BoardShim(BoardIds.CYTON_DAISY_BOARD, self.params)

        self.chs = self.board.get_exg_channels(BoardIds.CYTON_DAISY_BOARD)
        self.hz = self.board.get_sampling_rate(BoardIds.CYTON_DAISY_BOARD)

    def __enter__(self):
        self.board.prepare_session()
        self.board.start_stream()
        return self

    def get_data(self):
        return self.board.get_board_data(num_samples=None, preset=self.board_preset)

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.board.stop_stream()
        self.board.release_session()
