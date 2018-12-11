#!/usr/bin/python3
import math
import numpy as np
import tensorflow as tf

import hive_lib

UI_LINES_PER_ROW = 4
UI_CHARS_PER_COLUMN = 9

def boards_pprint(boards):
    for board_idx in range(boards.shape[0]):
        print("\n=============================================================")
        print("Board {}\n".format(board_idx))
        for line in _board_lines(boards[board_idx]):
            print("{}".format(line))


def _board_lines(board):
    """Yields lines of the printed board."""
    lines = []
    to_line = math.floor(board.shape[0] + 1 + 0.5) * UI_LINES_PER_ROW
    yield _board_line(board, -1, first_line=True)
    for line_y in range(to_line):
        yield _board_line(board, line_y, first_line=False)

def _board_line(board, line_y, first_line):
    """Yields one line of the printed board: relative to self.min_y."""
    strips = []
    for x in range(board.shape[1]+1):
        adj_line_y = line_y
        if x % 2 == 1:
            adj_line_y = line_y - math.floor(UI_LINES_PER_ROW / 2)
        y = math.floor(adj_line_y / UI_LINES_PER_ROW)
        if y >= 0 and y < board.shape[0] and x >= 0 and x < board.shape[1]:
            value = board[y, x, 0]
        else:
            value = 0.0
        sub_y = adj_line_y % UI_LINES_PER_ROW
        strips += [_board_strip(value, x, y, sub_y, is_final=(x == board.shape[1]))]
    return ''.join(strips)


def _board_strip(value, x, y, sub_y, is_final):
    """Yields a strip related to the given x column of a line of the board."""
    if sub_y == 0:
        if is_final:
            return ' /'
        return ' /' + (UI_CHARS_PER_COLUMN - 2) * ' '
    elif sub_y == 1:
        if is_final:
            return '/'
        coord = "%d,%d" % (x, y)
        return ('/ {:^' + str(UI_CHARS_PER_COLUMN - 2) + '}').format(coord)
    elif sub_y == 2:
        if is_final:
            return '\\'
        return ('\\ {:^' + str(UI_CHARS_PER_COLUMN - 2) + '.3f}'
                ).format(value)
    else:
        if is_final:
            return ' \\'
        else:
            return ' \\' + (UI_CHARS_PER_COLUMN - 2) * '_'


class HiveLibTest(tf.test.TestCase):

    def test_sigmoid_to_max(self):
        with self.test_session() as sess:
            x = tf.constant([9.8, -12., 15., -3.2])
            y = hive_lib.sigmoid_to_max(x)
            y = sess.run(y)
            self.assertAllClose(9.8, y[0], msg="At the limit of linear, value shouldn't have changed.")
            self.assertAllClose(-3.2, y[3], msg="Negative numbers within limit of linear, shouldn't change either.")
            self.assertLess(-10, y[1], msg="Value should be limited by -10.")
            self.assertGreater(10, y[2], msg="Value should be limited by 10.")
            self.assertLess(-y[1], y[2], msg="Sigmoid should be monotonic.")


    def test_hexagonal_filters(self):
        # Test a given set of filter values.
        with tf.Graph().as_default():
            center = tf.reshape(tf.constant([0.001, 1., 0.001], dtype=tf.float32),
                                [3, 1, 1, 1])
            left = tf.reshape(tf.constant([0.002, 0.002], dtype=tf.float32),
                              [2, 1, 1, 1])
            right = tf.reshape(tf.constant([0.003, 0.003], dtype=tf.float32),
                               [2, 1, 1, 1])
            hex_filters = hive_lib.hexagonal_filters(1, 1, tf.float32, (center, left, right))
            hex_filters = [tf.reshape(f, [3, 3]) for f in hex_filters]
            init = tf.initializers.global_variables()
            with self.test_session() as sess:
                sess.run(init)
                hex_filters = sess.run(hex_filters)
        want = np.array([
            [ 0.002, 0.001, 0.003],
            [ 0.002, 1.000, 0.003],
            [ 0.000, 0.001, 0.000],
        ], dtype=np.float32)
        self.assertAllClose(hex_filters[0], want, msg="Even (for x%2==0) filter not initialized as expected.")
        want = np.array([
            [ 0.000, 0.001, 0.000],
            [ 0.002, 1.000, 0.003],
            [ 0.002, 0.001, 0.003],
        ], dtype=np.float32)
        self.assertAllClose(hex_filters[1], want, msg="Odd (for x%2==0) filter not initialized as expected.")

        # Test with random values, that the two filters get the same values.
        with tf.Graph().as_default():
            hex_filters = hive_lib.hexagonal_filters(1, 1, tf.float32,
                                                   tf.initializers.truncated_normal(0.0, 0.1))
            hex_filters = [tf.reshape(f, [3, 3]) for f in hex_filters]
            init = tf.initializers.global_variables()
            with tf.Session() as sess:
                sess.run(init)
                hex_filters = sess.run(hex_filters)

        center_even = hex_filters[0][:,1]
        center_odd = hex_filters[1][:,1]
        self.assertAllClose(center_even, center_odd,
                            msg="Even and Odd filter values should be the same, just the position is shifted.")

        left_even = hex_filters[0][0:1,0]
        left_odd = hex_filters[1][1:2,0]
        self.assertAllClose(left_even, left_odd,
                            msg="Even and Odd filter values should be the same, just the position is shifted.")
        right_even = hex_filters[0][0:1,2]
        right_odd = hex_filters[1][1:2,2]
        self.assertAllClose(right_even, right_odd,
                            msg="Even and Odd filter values should be the same, just the position is shifted.")


    def test_hexagonal_conv2d(self):
        BATCH_SIZE=2
        HEIGHT=5
        WIDTH=5
        DEPTH=1
        examples = np.arange(BATCH_SIZE * HEIGHT * WIDTH * DEPTH, dtype=np.float32)
        examples = examples.reshape([BATCH_SIZE, HEIGHT, WIDTH, DEPTH])
        with tf.Graph().as_default() as graph:
            hex = tf.constant(examples)
            center = tf.reshape(tf.constant([0.001, 1., 0.001], dtype=tf.float32),
                                [3, 1, 1, 1])
            left = tf.reshape(tf.constant([0.001, 0.001], dtype=tf.float32),
                              [2, 1, 1, 1])
            right = tf.reshape(tf.constant([0.001, 0.001], dtype=tf.float32),
                               [2, 1, 1, 1])
            hex_conv = hive_lib.hexagonal_conv2d(hex, 1, filter_initializer=[center, left, right])
            init = tf.initializers.global_variables()
            with tf.Session() as sess:
                sess.run(init)
                got = hex_conv.eval()
        want = np.array([
            [
                [[  0.006], [  1.020], [  2.011], [  3.030], [  4.012]],
                [[  5.017], [  6.046], [  7.032], [  8.058], [  9.029]],
                [[ 10.037], [ 11.076], [ 12.062], [ 13.088], [ 14.049]],
                [[ 15.057], [ 16.106], [ 17.092], [ 18.118], [ 19.069]],
                [[ 20.052], [ 21.058], [ 22.095], [ 23.064], [ 24.060]],
            ],
            [
                [[ 25.056], [ 26.145], [ 27.086], [ 28.155], [ 29.062]],
                [[ 30.117], [ 31.196], [ 32.182], [ 33.208], [ 34.129]],
                [[ 35.137], [ 36.226], [ 37.212], [ 38.238], [ 39.149]],
                [[ 40.157], [ 41.256], [ 42.242], [ 43.268], [ 44.169]],
                [[ 45.127], [ 46.133], [ 47.220], [ 48.139], [ 49.135]],
            ],
        ])

        if not np.allclose(got, want):
            print("hexagonal_conv2d returned: ")
            boards_pprint(got)
            print("But test wants:")
            boards_pprint(want)
            self.assertAllClose(got, want)


if __name__ == '__main__':
    tf.test.main()