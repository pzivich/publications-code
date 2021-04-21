import pytest

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import networkx as nx

from amonhen import NetworkTMLE


@pytest.fixture
def sm_network():
    """Loads a small network for short test runs and checks of data set creations"""
    G = nx.Graph()
    G.add_nodes_from([(1, {'W': 1, 'A': 1, 'Y': 1, 'C': 1}),
                      (2, {'W': 0, 'A': 0, 'Y': 0, 'C': -1}),
                      (3, {'W': 0, 'A': 1, 'Y': 0, 'C': 5}),
                      (4, {'W': 0, 'A': 0, 'Y': 1, 'C': 0}),
                      (5, {'W': 1, 'A': 0, 'Y': 0, 'C': 0}),
                      (6, {'W': 1, 'A': 0, 'Y': 1, 'C': 0}),
                      (7, {'W': 0, 'A': 1, 'Y': 0, 'C': 10}),
                      (8, {'W': 0, 'A': 0, 'Y': 0, 'C': -5}),
                      (9, {'W': 1, 'A': 1, 'Y': 0, 'C': -5})])

    G.add_edges_from([(1, 2), (1, 3), (1, 9),
                      (2, 3), (2, 6),
                      (3, 4),
                      (4, 7),
                      (5, 7), (5, 9)
                      ])
    return G


@pytest.fixture
def r_network():
    """Loads network from the R library tmlenet for comparison"""
    df = pd.read_csv("tests/cross-comparison/tmlenet_r_data.csv")
    df['IDs'] = df['IDs'].str[1:].astype(int)
    df['NETID_split'] = df['Net_str'].str.split()

    G = nx.DiGraph()
    G.add_nodes_from(df['IDs'])

    for i, c in zip(df['IDs'], df['NETID_split']):
        if type(c) is list:
            for j in c:
                G.add_edge(i, int(j[1:]))

    # Adding attributes
    for node in G.nodes():
        G.nodes[node]['W'] = np.int(df.loc[df['IDs'] == node, 'W1'])
        G.nodes[node]['A'] = np.int(df.loc[df['IDs'] == node, 'A'])
        G.nodes[node]['Y'] = np.int(df.loc[df['IDs'] == node, 'Y'])

    return G


class TestNetworkTMLE:

    def test_error_node_ids(self):
        G = nx.Graph()
        G.add_nodes_from([(1, {'A': 1, 'Y': 1}), (2, {'A': 0, 'Y': 1}), ("N", {'A': 1, 'Y': 0}), (4, {'A': 0, 'Y': 0})])
        with pytest.raises(ValueError):
            NetworkTMLE(network=G, exposure='A', outcome='Y')

    def test_error_self_loops(self):
        G = nx.Graph()
        G.add_nodes_from([(1, {'A': 1, 'Y': 1}), (2, {'A': 0, 'Y': 1}), (3, {'A': 1, 'Y': 0}), (4, {'A': 0, 'Y': 0})])
        G.add_edges_from([(1, 1), (1, 2), (3, 4)])
        with pytest.raises(ValueError):
            NetworkTMLE(network=G, exposure='A', outcome='Y')

    def test_error_nonbinary_a(self):
        G = nx.Graph()
        G.add_nodes_from([(1, {'A': 2, 'Y': 1}), (2, {'A': 5, 'Y': 1}), (3, {'A': 1, 'Y': 0}), (4, {'A': 0, 'Y': 0})])
        with pytest.raises(ValueError):
            NetworkTMLE(network=G, exposure='A', outcome='Y')

    def test_error_degree_restrictions(self, r_network):
        with pytest.raises(ValueError):
            NetworkTMLE(network=r_network, exposure='A', outcome='Y', degree_restrict=2)
        with pytest.raises(ValueError):
            NetworkTMLE(network=r_network, exposure='A', outcome='Y', degree_restrict=[0, 1, 2])
        with pytest.raises(ValueError):
            NetworkTMLE(network=r_network, exposure='A', outcome='Y', degree_restrict=[2, 0])

    def test_error_fit_gimodel(self, r_network):
        tmle = NetworkTMLE(network=r_network, exposure='A', outcome='Y')
        # tmle.exposure_model('W')
        tmle.exposure_map_model('W', distribution=None)
        tmle.outcome_model('A + W')
        with pytest.raises(ValueError):
            tmle.fit(p=0.0, samples=10)

    def test_error_fit_gsmodel(self, r_network):
        tmle = NetworkTMLE(network=r_network, exposure='A', outcome='Y')
        tmle.exposure_model('W')
        # tmle.exposure_map_model('W', distribution=None)
        tmle.outcome_model('A + W')
        with pytest.raises(ValueError):
            tmle.fit(p=0.0, samples=10)

    def test_error_gs_distributions(self, r_network):
        tmle = NetworkTMLE(network=r_network, exposure='A', outcome='Y')

        with pytest.raises(ValueError):
            tmle.exposure_map_model('W', measure='mean', distribution=None)

        with pytest.raises(ValueError):
            tmle.exposure_map_model('W', measure='mean', distribution='multinomial')

    def test_error_fit_qmodel(self, r_network):
        tmle = NetworkTMLE(network=r_network, exposure='A', outcome='Y')
        tmle.exposure_model('W')
        tmle.exposure_map_model('W', distribution=None)
        # tmle.outcome_model('A + W')
        with pytest.raises(ValueError):
            tmle.fit(p=0.0, samples=10)

    def test_error_p_bound(self, r_network):
        tmle = NetworkTMLE(network=r_network, exposure='A', outcome='Y')
        tmle.exposure_model('W')
        tmle.exposure_map_model('W', distribution=None)
        tmle.outcome_model('A + W')
        # For single 'p'
        with pytest.raises(ValueError):
            tmle.fit(p=1.5, samples=10)

        # For multiple 'p'
        with pytest.raises(ValueError):
            tmle.fit(p=[0.1, 1.5, 0.1,
                        0.1, 0.1, 0.1,
                        0.1, 0.1, 0.1], samples=100)

    def test_error_p_type(self, r_network):
        tmle = NetworkTMLE(network=r_network, exposure='A', outcome='Y')
        tmle.exposure_model('W')
        tmle.exposure_map_model('W', distribution=None)
        tmle.outcome_model('A + W')
        with pytest.raises(ValueError):
            tmle.fit(p=5, samples=10)

    def test_error_summary(self, r_network):
        tmle = NetworkTMLE(network=r_network, exposure='A', outcome='Y')
        tmle.exposure_model('W')
        tmle.exposure_map_model('W', distribution=None)
        tmle.outcome_model('A + W')
        with pytest.raises(ValueError):
            tmle.summary()

    def test_df_creation(self, sm_network):
        columns = ["_original_id_", "W", "A", "Y", "A_sum", "A_mean", "W_sum", "W_mean", "degree"]
        expected = pd.DataFrame([[1, 1, 1, 1, 2, 2/3, 1, 1/3, 3],
                                 [2, 0, 0, 0, 2, 2/3, 2, 2/3, 3],
                                 [3, 0, 1, 0, 1, 1/3, 1, 1/3, 3],
                                 [4, 0, 0, 1, 2, 1,   0, 0,   2],
                                 [5, 1, 0, 0, 2, 1,   1, 1/2, 2],
                                 [6, 1, 0, 1, 0, 0,   0, 0,   1],
                                 [7, 0, 1, 0, 0, 0,   1, 1/2, 2],
                                 [8, 0, 0, 0, 0, 0,   0, 0,   0],
                                 [9, 1, 1, 0, 1, 1/2, 2, 1,   2]],
                                columns=columns,
                                index=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        tmle = NetworkTMLE(network=sm_network, exposure='A', outcome='Y')
        created = tmle.df

        # Checking that expected is the same as the created
        assert tmle._continuous_outcome is False
        pdt.assert_frame_equal(expected,
                               created[columns],
                               check_dtype=False)

    def test_df_creation_restricted(self, sm_network):
        expected = pd.DataFrame([[1, 1, 1, 2, 2/3, 1, 1/3, 3],
                                 [0, 0, 0, 2, 2/3, 2, 2/3, 3],
                                 [0, 1, 0, 1, 1/3, 1, 1/3, 3],
                                 [0, 0, 1, 2, 1,   0, 0,   2],
                                 [1, 0, 0, 2, 1,   1, 1/2, 2],
                                 [1, 0, 1, 0, 0,   0, 0,   1],
                                 [0, 1, 0, 0, 0,   1, 1/2, 2],
                                 [0, 0, 0, 0, 0,   0, 0,   0],
                                 [1, 1, 0, 1, 1/2, 2, 1,   2]],
                                columns=["W", "A", "Y", "A_sum", "A_mean", "W_sum", "W_mean", "degree"],
                                index=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        expected_r = pd.DataFrame([[0, 0, 1, 2, 1,   0, 0,   2],
                                   [1, 0, 0, 2, 1,   1, 1/2, 2],
                                   [1, 0, 1, 0, 0,   0, 0,   1],
                                   [0, 1, 0, 0, 0,   1, 1/2, 2],
                                   [0, 0, 0, 0, 0,   0, 0,   0],
                                   [1, 1, 0, 1, 1/2, 2, 1,   2]],
                                  columns=["W", "A", "Y", "A_sum", "A_mean", "W_sum", "W_mean", "degree"],
                                  index=[3, 4, 5, 6, 7, 8])
        tmle = NetworkTMLE(network=sm_network, exposure='A', outcome='Y', degree_restrict=[0, 2])
        created = tmle.df
        created_r = tmle.df_restricted

        # Checking that expected is the same as the created
        pdt.assert_frame_equal(expected,
                               created[["W", "A", "Y", "A_sum", "A_mean", "W_sum", "W_mean", "degree"]],
                               check_dtype=False)

        pdt.assert_frame_equal(expected_r,
                               created_r[["W", "A", "Y", "A_sum", "A_mean", "W_sum", "W_mean", "degree"]],
                               check_dtype=False)

    def test_restricted_number(self, sm_network):
        tmle = NetworkTMLE(network=sm_network, exposure='A', outcome='Y', degree_restrict=[0, 2])
        n_created = tmle.df.shape[0]
        n_created_r = tmle.df_restricted.shape[0]
        assert 6 == n_created_r
        assert 3 == n_created - n_created_r

        tmle = NetworkTMLE(network=sm_network, exposure='A', outcome='Y', degree_restrict=[1, 3])
        n_created = tmle.df.shape[0]
        n_created_r = tmle.df_restricted.shape[0]
        assert 8 == n_created_r
        assert 1 == n_created - n_created_r

    def test_continuous_processing(self):
        G = nx.Graph()
        y_list = [1, -1, 5, 0, 0, 0, 10, -5]
        G.add_nodes_from([(1, {'A': 0, 'Y': y_list[0]}), (2, {'A': 1, 'Y': y_list[1]}),
                          (3, {'A': 1, 'Y': y_list[2]}), (4, {'A': 0, 'Y': y_list[3]}),
                          (5, {'A': 1, 'Y': y_list[4]}), (6, {'A': 1, 'Y': y_list[5]}),
                          (7, {'A': 0, 'Y': y_list[6]}), (8, {'A': 0, 'Y': y_list[7]})])

        tmle = NetworkTMLE(network=G, exposure='A', outcome='Y', continuous_bound=0.0001)

        # Checking all flagged parts are correct
        assert tmle._continuous_outcome is True
        assert tmle._continuous_min == -5.0001
        assert tmle._continuous_max == 10.0001
        assert tmle._cb == 0.0001

        # Checking that TMLE bounding works as intended
        maximum = 10.0001
        minimum = -5.0001
        y_bound = (np.array(y_list) - minimum) / (maximum - minimum)

        pdt.assert_series_equal(pd.Series(y_bound, index=[0, 1, 2, 3, 4, 5, 6, 7]),
                                tmle.df['Y'],
                                check_dtype=False, check_names=False)

    def test_df_creation_continuous(self, sm_network):
        expected = pd.DataFrame([[1, 1, 2, 1, 3],
                                 [0, 0, 2, 2, 3],
                                 [0, 1, 1, 1, 3],
                                 [0, 0, 2, 0, 2],
                                 [1, 0, 2, 1, 2],
                                 [1, 0, 0, 0, 1],
                                 [0, 1, 0, 1, 2],
                                 [0, 0, 0, 0, 0],
                                 [1, 1, 1, 2, 2]],
                                columns=["W", "A", "A_sum", "W_sum", "degree"],
                                index=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        expected["C"] = [4.00001333e-01, 2.66669778e-01, 6.66664444e-01, 3.33335556e-01, 3.33335556e-01,
                         3.33335556e-01, 9.99993333e-01, 6.66657778e-06, 6.66657778e-06]

        tmle = NetworkTMLE(network=sm_network, exposure='A', outcome='C', continuous_bound=0.0001)
        created = tmle.df

        # Checking that expected is the same as the created
        assert tmle._continuous_outcome is True
        pdt.assert_frame_equal(expected[["W", "A", "C", "A_sum", "W_sum", "degree"]],
                               created[["W", "A", "C", "A_sum", "W_sum", "degree"]],
                               check_dtype=False)

    def test_no_consecutive_ids(self):
        G = nx.Graph()
        G.add_nodes_from([(1, {'W': 1, 'A': 1, 'Y': 1}), (2, {'W': 0, 'A': 0, 'Y': 0}),
                          (3, {'W': 0, 'A': 1, 'Y': 0}), (4, {'W': 0, 'A': 0, 'Y': 1}),
                          (5, {'W': 1, 'A': 0, 'Y': 0}), (7, {'W': 1, 'A': 0, 'Y': 1}),
                          (9, {'W': 0, 'A': 1, 'Y': 0}), (11, {'W': 0, 'A': 0, 'Y': 0}),
                          (12, {'W': 1, 'A': 1, 'Y': 0})])
        G.add_edges_from([(1, 2), (1, 3), (1, 12), (2, 3), (2, 7),
                          (3, 4), (4, 9), (5, 9), (5, 12)])

        expected = pd.DataFrame([[1, 1, 1, 1, 2, 2 / 3, 1, 1 / 3, 3],
                                 [2, 0, 0, 0, 2, 2/3, 2, 2/3, 3],
                                 [3, 0, 1, 0, 1, 1 / 3, 1, 1 / 3, 3],
                                 [4, 0, 0, 1, 2, 1, 0, 0, 2],
                                 [5, 1, 0, 0, 2, 1, 1, 1 / 2, 2],
                                 [7, 1, 0, 1, 0, 0, 0, 0, 1],
                                 [8, 0, 1, 0, 0, 0, 1, 1 / 2, 2],
                                 [11, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [12, 1, 1, 0, 1, 1 / 2, 2, 1, 2]
                                 ],
                                columns=["_original_id_", "W", "A", "Y", "A_sum",
                                         "A_mean", "W_sum", "W_mean", "degree"],
                                index=[0, 1, 2, 3, 4, 5, 6, 7, 8])

        tmle = NetworkTMLE(network=G, exposure='A', outcome='Y')
        created = tmle.df.sort_values(by='_original_id_').reset_index()

        pdt.assert_frame_equal(expected[["W", "A", "Y", "A_sum", "A_mean", "W_sum", "W_mean", "degree"]],
                               created[["W", "A", "Y", "A_sum", "A_mean", "W_sum", "W_mean", "degree"]],
                               check_dtype=False)

    def test_df_creation_nonparametric(self, sm_network):
        columns = ["_original_id_", "A", "A_map1", "A_map2", "A_map3"]
        expected = pd.DataFrame([[1, 1, 0, 1, 1],
                                 [2, 0, 1, 1, 0],
                                 [3, 1, 1, 0, 0],
                                 [4, 0, 1, 1, 0],
                                 [5, 0, 1, 1, 0],
                                 [6, 0, 0, 0, 0],
                                 [7, 1, 0, 0, 0],
                                 [8, 0, 0, 0, 0],
                                 [9, 1, 1, 0, 0]],
                                columns=columns,
                                index=[0, 1, 2, 3, 4, 5, 6, 7, 8])

        tmle = NetworkTMLE(network=sm_network, exposure='A', outcome='Y')
        created = tmle.df.sort_values(by='_original_id_').reset_index()
        # Checking that expected is the same as the created
        pdt.assert_frame_equal(expected[columns], created[columns], check_dtype=False)

    def test_summary_measures_creation(self, sm_network):
        columns = ["_original_id_", "A_sum", "A_mean", "A_var", "W_sum", "W_mean", "W_var"]
        neighbors_w = {1: np.array([0, 0, 1]), 2: np.array([0, 1, 1]), 3: np.array([0, 0, 1]), 4: np.array([0, 0]),
                       5: np.array([0, 1]), 6: np.array([0]), 7: np.array([0, 1]), 9: np.array([1, 1])}
        neighbors_a = {1: np.array([0, 1, 1]), 2: np.array([0, 1, 1]), 3: np.array([0, 0, 1]), 4: np.array([1, 1]),
                       5: np.array([1, 1]), 6: np.array([0]), 7: np.array([0, 0]), 9: np.array([0, 1])}

        expected = pd.DataFrame([[1, np.sum(neighbors_a[1]), np.mean(neighbors_a[1]), np.var(neighbors_a[1]),
                                  np.sum(neighbors_w[1]), np.mean(neighbors_w[1]), np.var(neighbors_w[1])],
                                 [2, np.sum(neighbors_a[2]), np.mean(neighbors_a[2]), np.var(neighbors_a[2]),
                                  np.sum(neighbors_w[2]), np.mean(neighbors_w[2]), np.var(neighbors_w[2])],
                                 [3, np.sum(neighbors_a[3]), np.mean(neighbors_a[3]), np.var(neighbors_a[3]),
                                  np.sum(neighbors_w[3]), np.mean(neighbors_w[3]), np.var(neighbors_w[3])],
                                 [4, np.sum(neighbors_a[4]), np.mean(neighbors_a[4]), np.var(neighbors_a[4]),
                                  np.sum(neighbors_w[4]), np.mean(neighbors_w[4]), np.var(neighbors_w[4])],
                                 [5, np.sum(neighbors_a[5]), np.mean(neighbors_a[5]), np.var(neighbors_a[5]),
                                  np.sum(neighbors_w[5]), np.mean(neighbors_w[5]), np.var(neighbors_w[5])],
                                 [6, np.sum(neighbors_a[6]), np.mean(neighbors_a[6]), np.var(neighbors_a[6]),
                                  np.sum(neighbors_w[6]), np.mean(neighbors_w[6]), np.var(neighbors_w[6])],
                                 [7, np.sum(neighbors_a[7]), np.mean(neighbors_a[7]), np.var(neighbors_a[7]),
                                  np.sum(neighbors_w[7]), np.mean(neighbors_w[7]), np.var(neighbors_w[7])],
                                 [8, 0, 0, 0, 0, 0, 0],  # Isolates are = 0
                                 [9, np.sum(neighbors_a[9]), np.mean(neighbors_a[9]), np.var(neighbors_a[9]),
                                  np.sum(neighbors_w[9]), np.mean(neighbors_w[9]), np.var(neighbors_w[9])]],
                                columns=columns,
                                index=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        tmle = NetworkTMLE(network=sm_network, exposure='A', outcome='Y')
        created = tmle.df

        # Checking that expected is the same as the created
        assert tmle._continuous_outcome is False
        pdt.assert_frame_equal(expected,
                               created[columns],
                               check_dtype=False)

    def test_distance_measures_creation(self, sm_network):
        columns = ["_original_id_", "A_mean_dist", "A_var_dist", "W_mean_dist", "W_var_dist"]
        neighbors_w = {1: np.array([-1, -1, 0]), 2: np.array([0, 1, 1]), 3: np.array([0, 0, 1]), 4: np.array([0, 0]),
                       5: np.array([-1, 0]), 6: np.array([-1]), 7: np.array([0, 1]), 9: np.array([0, 0])}
        neighbors_a = {1: np.array([-1, 0, 0]), 2: np.array([0, 1, 1]), 3: np.array([-1, -1, 0]), 4: np.array([1, 1]),
                       5: np.array([1, 1]), 6: np.array([0]), 7: np.array([-1, -1]), 9: np.array([-1, 0])}

        expected = pd.DataFrame([[1, np.mean(neighbors_a[1]), np.var(neighbors_a[1]),
                                  np.mean(neighbors_w[1]), np.var(neighbors_w[1])],
                                 [2, np.mean(neighbors_a[2]), np.var(neighbors_a[2]),
                                  np.mean(neighbors_w[2]), np.var(neighbors_w[2])],
                                 [3, np.mean(neighbors_a[3]), np.var(neighbors_a[3]),
                                  np.mean(neighbors_w[3]), np.var(neighbors_w[3])],
                                 [4, np.mean(neighbors_a[4]), np.var(neighbors_a[4]),
                                  np.mean(neighbors_w[4]), np.var(neighbors_w[4])],
                                 [5, np.mean(neighbors_a[5]), np.var(neighbors_a[5]),
                                  np.mean(neighbors_w[5]), np.var(neighbors_w[5])],
                                 [6, np.mean(neighbors_a[6]), np.var(neighbors_a[6]),
                                  np.mean(neighbors_w[6]), np.var(neighbors_w[6])],
                                 [7, np.mean(neighbors_a[7]), np.var(neighbors_a[7]),
                                  np.mean(neighbors_w[7]), np.var(neighbors_w[7])],
                                 [8, 0, 0, 0, 0],  # Isolates are = 0
                                 [9, np.mean(neighbors_a[9]), np.var(neighbors_a[9]),
                                  np.mean(neighbors_w[9]), np.var(neighbors_w[9])]],
                                columns=columns,
                                index=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        tmle = NetworkTMLE(network=sm_network, exposure='A', outcome='Y')
        created = tmle.df

        # Checking that expected is the same as the created
        assert tmle._continuous_outcome is False
        pdt.assert_frame_equal(expected,
                               created[columns],
                               check_dtype=False)

    def test_threshold_create(self, sm_network):
        expected = pd.DataFrame([[1, 1, 2, 1, 3],
                                 [0, 0, 2, 2, 3],
                                 [0, 1, 1, 1, 3],
                                 [0, 0, 2, 0, 2],
                                 [1, 0, 2, 1, 2],
                                 [1, 0, 0, 0, 1],
                                 [0, 1, 0, 1, 2],
                                 [0, 0, 0, 0, 0],
                                 [1, 1, 1, 2, 2]],
                                columns=["W", "A", "A_sum", "W_sum", "degree"],
                                index=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        expected["A_t2"] = np.where(expected['A_sum'] > 2, 1, 0)
        expected["W_tp50"] = np.where(expected['W_sum']/expected['degree'] > 0.5, 1, 0)

        tmle = NetworkTMLE(network=sm_network, exposure='A', outcome='Y')
        tmle.define_threshold(variable='A', threshold=2, definition='sum')
        tmle.define_threshold(variable='W', threshold=0.5, definition='mean')
        created = tmle.df_restricted
        # Checking that expected is the same as the created

        columns = ["A", "A_t2", "W_tp50"]
        pdt.assert_frame_equal(expected[columns], created[columns], check_dtype=False)

    def test_check_denominator_est(self, r_network):
        tmle = NetworkTMLE(network=r_network, exposure='A', outcome='Y')
        tmle.exposure_model('W + W_sum')
        tmle.exposure_map_model('A + W + W_sum', distribution=None)
        tmle.outcome_model('A + W + A_sum + W_sum')

        assert tmle._denominator_estimated_ is False
        tmle.fit(p=0.4, samples=5, seed=20110129)
        assert tmle._denominator_estimated_ is True

    def test_marginal_vector_length_stoch(self, r_network):
        tmle = NetworkTMLE(network=r_network, exposure='A', outcome='Y')
        tmle.exposure_model('W + W_sum')
        tmle.exposure_map_model('A + W + W_sum', distribution=None)
        tmle.outcome_model('A + W + A_sum + W_sum')
        tmle.fit(p=0.4, samples=10, seed=20110129)
        assert len(tmle.marginals_vector) == 10

    def test_qmodel_params1(self, sm_network):
        # Comparing to SAS logit model
        sas_params = [-1.5109, -0.9583, 0.3694, 1.5332]
        sas_preds = [0.45083299, 0.31601450, 0.10911248, 0.31601450, 0.68157767,
                     0.50558116, 0.07804636, 0.18081217, 0.36200818]

        tmle = NetworkTMLE(network=sm_network, exposure='A', outcome='Y')
        tmle.outcome_model('A + A_sum + W')
        est_params = tmle._outcome_model.params
        est_preds = tmle._Qinit_

        npt.assert_allclose(sas_params, est_params, atol=1e-4)
        npt.assert_allclose(sas_preds, est_preds)

    def test_qmodel_params2(self, sm_network):
        # Comparing to SAS linear regression
        sas_params = [0.3598, 0.2806, -0.0187, -0.2100]
        sas_preds = [0.3929295, 0.3223863, 0.6216814, 0.3223863, 0.1123546, 0.1497950, 0.6404016, 0.3598267, 0.4116497]

        tmle = NetworkTMLE(network=sm_network, exposure='A', outcome='C')
        tmle.outcome_model('A + A_sum + W')
        est_params = tmle._outcome_model.params
        est_preds = tmle._Qinit_

        npt.assert_allclose(sas_params, est_params, atol=1e-4)
        npt.assert_allclose(sas_preds, est_preds, atol=1e-6)

    def test_qmodel_params3(self, sm_network):
        # Comparing to SAS linear regression
        sas_params = [-1.1691, 0.7686, -0.0028, -0.6053]
        sas_preds = [0.3637460, 0.3089358, 0.6681595, 0.3089358, 0.1686493, 0.1695826, 0.6700057, 0.3106454, 0.3647510]

        tmle = NetworkTMLE(network=sm_network, exposure='A', outcome='C')
        tmle.outcome_model('A + A_sum + W', distribution='poisson')
        est_params = tmle._outcome_model.params
        est_preds = tmle._Qinit_

        npt.assert_allclose(sas_params, est_params, atol=1e-4)
        npt.assert_allclose(sas_preds, est_preds, atol=1e-6)

    def test_qmodel_params4(self, sm_network):
        # Comparing to SAS logit model
        sas_params = [-0.9628, 0.3087]
        sas_preds = [0.4144844, 0.4144844, 0.2763229, 0.2763229, 0.2763229, 0.3420625]

        tmle = NetworkTMLE(network=sm_network, exposure='A', outcome='Y', degree_restrict=[0, 2])  # Restricted
        tmle.outcome_model('A_sum')
        est_params = tmle._outcome_model.params
        est_preds = tmle._Qinit_
        print(est_preds)

        npt.assert_allclose(sas_params, est_params, atol=1e-4)
        npt.assert_allclose(sas_preds, est_preds)

    def test_qmodel_params5(self, sm_network):
        # Comparing to SAS linear regression
        sas_params = [0.3718, 0.2436, -0.0128, -0.2179]
        sas_preds = [0.3461641, 0.1282299, 0.1538692, 0.6153769, 0.3718034, 0.3846231]

        tmle = NetworkTMLE(network=sm_network, exposure='A', outcome='C', degree_restrict=[0, 2])  # Restricted
        tmle.outcome_model('A + A_sum + W')
        est_params = tmle._outcome_model.params
        est_preds = tmle._Qinit_

        npt.assert_allclose(sas_params, est_params, atol=1e-4)
        npt.assert_allclose(sas_preds, est_preds, atol=1e-6)

    def test_qmodel_params6(self, sm_network):
        # Comparing to SAS linear regression
        sas_params = [-1.2612, 0.6328, -0.1340]
        sas_preds = [0.2167149, 0.2167149, 0.2833184, 0.5334486, 0.2833184, 0.4665515]

        tmle = NetworkTMLE(network=sm_network, exposure='A', outcome='C', degree_restrict=[0, 2])  # Restricted
        tmle.outcome_model('A + A_sum', distribution='poisson')
        est_params = tmle._outcome_model.params
        est_preds = tmle._Qinit_

        npt.assert_allclose(sas_params, est_params, atol=1e-4)
        npt.assert_allclose(sas_preds, est_preds, atol=1e-6)

    def test_gmodel_params(self, r_network):
        # Nonparametric g-model
        tmle = NetworkTMLE(network=r_network, exposure='A', outcome='Y')
        tmle.exposure_model('W + W_sum')
        tmle.exposure_map_model('A + W + W_sum', distribution=None)
        tmle.outcome_model('A + A_sum + W + W_sum')
        tmle.fit(p=0.5, samples=1)

        sas_gi_param = [-1.2043, 1.4001, 0.6412]
        est_params = tmle._treatment_models[0].params
        npt.assert_allclose(est_params, sas_gi_param, atol=1e-4)

        sas_gs1_param = [-1.8720, 0.1960, 0.0815, 1.6364]
        est_params = tmle._treatment_models[1].params
        npt.assert_allclose(est_params, sas_gs1_param, atol=1e-4)

        sas_gs2_param = [-2.7908, -0.2574, -0.1038, 1.7206, -0.2127]
        est_params = tmle._treatment_models[2].params
        npt.assert_allclose(est_params, sas_gs2_param, atol=1e-4)

        # Poisson gs-model
        tmle.exposure_model('W + W_sum')
        tmle.exposure_map_model('A + W + W_sum', distribution='poisson', measure='sum')
        tmle.fit(p=0.5, samples=1)

        sas_gi_param = [-1.2043, 1.4001, 0.6412]
        est_params = tmle._treatment_models[0].params
        npt.assert_allclose(est_params, sas_gi_param, atol=1e-4)

        sas_gs_param = [-1.5670, 0.0150, 0.0201, 1.0277]
        est_params = tmle._treatment_models[1].params
        npt.assert_allclose(est_params, sas_gs_param, atol=1e-4)

        # Linear gs-model
        tmle.exposure_model('W + W_sum')
        tmle.exposure_map_model('A + W + W_sum', distribution='normal', measure='sum')
        tmle.fit(p=0.5, samples=1)

        sas_gi_param = [-1.2043, 1.4001, 0.6412]
        est_params = tmle._treatment_models[0].params
        npt.assert_allclose(est_params, sas_gi_param, atol=1e-4)

        sas_gs_param = [0.18303, 0.01088, 0.00271, 0.53839]
        est_params = tmle._treatment_models[1].params
        npt.assert_allclose(est_params, sas_gs_param, atol=1e-4)

        tmle.exposure_model('W + W_sum')
        tmle.exposure_map_model('A + W + W_sum', distribution='normal', measure='mean')
        tmle.fit(p=0.5, samples=1)
        sas_gs_param = [0.12776, 0.02769, 0.01849, 0.31258]
        est_params = tmle._treatment_models[1].params
        npt.assert_allclose(est_params, sas_gs_param, atol=1e-4)

    def test_gs_distribution_measure_checks(self, sm_network):
        tmle = NetworkTMLE(network=sm_network, exposure='A', outcome='Y')

        # Non-parametric with incorrect measures
        with pytest.raises(ValueError):
            tmle.exposure_map_model('A + A_sum', measure='sum', distribution=None)
        with pytest.raises(ValueError):
            tmle.exposure_map_model('A + A_sum', measure='mean', distribution=None)

        # Multinomial with incorrect
        with pytest.raises(ValueError):
            tmle.exposure_map_model('A + A_sum', measure='mean', distribution='Multinomial')

    def test_procedure_vs_sas(self, r_network):
        sas_params = [-2.3922, 0.8113, 1.0667, 1.5355, 1.2313]

        tmle = NetworkTMLE(network=r_network, exposure='A', outcome='Y')

        # Checking Q-model results
        tmle.outcome_model('A + W + A_sum + W_sum', distribution='poisson')
        est_params = tmle._outcome_model.params
        npt.assert_allclose(sas_params, est_params, atol=1e-4)

        # Checking g-model denominator
        est_gi_param = [-1.2043, 1.4001, 0.6412]
        est_gs1_param = [-1.8720, 0.1960, 0.0815, 1.6364]
        est_gs2_param = [-2.7908, -0.2574, -0.1038, 1.7206, -0.2127]
        tmle.exposure_model('W + W_sum')
        tmle.exposure_map_model('A + W + W_sum', distribution=None)
        tmle.fit(p=0.5, samples=10)
        est_params = tmle._treatment_models[0].params
        npt.assert_allclose(est_params, est_gi_param, atol=1e-4)
        est_params = tmle._treatment_models[1].params
        npt.assert_allclose(est_params, est_gs1_param, atol=1e-4)
        est_params = tmle._treatment_models[2].params
        npt.assert_allclose(est_params, est_gs2_param, atol=1e-4)

        # Check overall versus deterministic treatments

    # TODO check weighted network data generation
