import pytest

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import networkx as nx

from amonhen.utils import (exp_map, fast_exp_map, exp_map_individual, network_to_df, probability_to_odds,
                           odds_to_probability, bounding, tmle_unit_bounds, tmle_unit_unbound, check_conditional)


class TestExposureMapping:

    def test_exp_map_graph1(self):
        G = nx.star_graph(4)
        a = [1, 0, 0, 0, 0]
        for node in G.nodes():
            G.nodes[node]['A'] = a[node]

        npt.assert_equal([0, 1, 1, 1, 1],
                         exp_map(G, 'A'))

    def test_exp_map_graph2(self):
        G = nx.star_graph(4)
        a = [1, 1, 0, 0, 0]
        for node in G.nodes():
            G.nodes[node]['A'] = a[node]

        npt.assert_equal([1, 1, 1, 1, 1],
                         exp_map(G, 'A'))

    def test_exp_map_graph3(self):
        G = nx.complete_graph(5)
        a = [1, 1, 1, 0, 0]
        for node in G.nodes():
            G.nodes[node]['A'] = a[node]

        npt.assert_equal([2, 2, 2, 3, 3],
                         exp_map(G, 'A'))

    def test_fast_exp_map_graph1(self):
        G = nx.star_graph(4)
        a = [1, 0, 0, 0, 0]
        for node in G.nodes():
            G.nodes[node]['A'] = a[node]

        npt.assert_equal(fast_exp_map(nx.adjacency_matrix(G, weight=None), np.array(a), measure='sum'),
                         exp_map(G, 'A'))

    def test_fast_exp_map_graph2(self):
        G = nx.star_graph(4)
        a = [1, 1, 0, 0, 0]
        for node in G.nodes():
            G.nodes[node]['A'] = a[node]

        npt.assert_equal(fast_exp_map(nx.adjacency_matrix(G, weight=None), np.array(a), measure='sum'),
                         exp_map(G, 'A'))

    def test_fast_exp_map_graph3(self):
        G = nx.complete_graph(5)
        a = [1, 1, 1, 0, 0]
        for node in G.nodes():
            G.nodes[node]['A'] = a[node]

        npt.assert_equal(fast_exp_map(nx.adjacency_matrix(G, weight=None), np.array(a), measure='sum'),
                         exp_map(G, 'A'))

    def test_exp_map_directed(self):
        G = nx.DiGraph()
        G.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4)])
        a = [1, 0, 1, 1, 1]
        for node in G.nodes():
            G.nodes[node]['A'] = a[node]

        npt.assert_equal([3, 0, 0, 0, 0],
                         exp_map(G, 'A'))

    def test_fast_exp_map_directed(self):
        G = nx.DiGraph()
        G.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4)])
        a = [1, 0, 1, 1, 1]
        for node in G.nodes():
            G.nodes[node]['A'] = a[node]

        npt.assert_equal(fast_exp_map(nx.adjacency_matrix(G, weight=None), np.array(a), measure='sum'),
                         exp_map(G, 'A'))

    def test_exp_map_individual1(self):
        G = nx.Graph()
        G.add_edges_from([(1, 2), (1, 3), (1, 4), (2, 3)])
        a = [1, 0, 1, 0]
        for node in G.nodes():
            G.nodes[node]['A'] = a[node-1]

        expected = pd.DataFrame()
        expected['A_map1'] = [0, 1, 1, 1]
        expected['A_map2'] = [1, 1, 0, np.nan]
        expected['A_map3'] = [0, np.nan, np.nan, np.nan]

        pdt.assert_frame_equal(expected, exp_map_individual(network=G, measure='A', max_degree=3))

    def test_exp_map_individual2(self):
        G = nx.Graph()
        G.add_edges_from([(1, 2), (1, 3), (1, 4), (2, 3)])
        a = [1, 0, 1, 0]
        for node in G.nodes():
            G.nodes[node]['A'] = a[node-1]

        expected = pd.DataFrame()
        expected['A_map1'] = [0, 1, 1, 1]
        expected['A_map2'] = [1, 1, 0, np.nan]

        pdt.assert_frame_equal(expected, exp_map_individual(network=G, measure='A', max_degree=2))

    def test_exp_map_individual3(self):
        G = nx.Graph()
        G.add_edges_from([(1, 2), (1, 3), (1, 4), (2, 3)])
        G.add_node(5)
        a = [1, 0, 1, 0, 0]
        for node in G.nodes():
            G.nodes[node]['A'] = a[node-1]

        expected = pd.DataFrame()
        expected['A_map1'] = [0, 1, 1, 1, np.nan]
        expected['A_map2'] = [1, 1, 0, np.nan, np.nan]
        expected['A_map3'] = [0, np.nan, np.nan, np.nan, np.nan]

        pdt.assert_frame_equal(expected, exp_map_individual(network=G, measure='A', max_degree=3))

    def test_exp_map_individual_directed(self):
        G = nx.DiGraph()
        G.add_edges_from([(1, 2), (1, 3), (1, 4), (2, 3)])
        a = [1, 0, 1, 0, 0]
        for node in G.nodes():
            G.nodes[node]['A'] = a[node-1]

        expected = pd.DataFrame()
        expected['A_map1'] = [0, 1, np.nan, np.nan]
        expected['A_map2'] = [1, np.nan, np.nan, np.nan]
        expected['A_map3'] = [0, np.nan, np.nan, np.nan]

        pdt.assert_frame_equal(expected, exp_map_individual(network=G, measure='A', max_degree=3))


class TestNetworkDataFrame:

    def test_graph1(self):
        G = nx.complete_graph(5)
        a = [1, 1, 1, 0, 0]
        b = [5, 6, 1, 2, 1]
        for node in G.nodes():
            G.nodes[node]['A'] = a[node]
            G.nodes[node]['B'] = b[node]

        df_test = pd.DataFrame()
        df_test['A'] = a
        df_test['B'] = b
        pdt.assert_frame_equal(df_test,
                               network_to_df(G),
                               check_like=True)

    def test_graph2(self):
        G = nx.star_graph(4)
        a = ['1', '1', '1', '0', 'a']
        b = [5.1, 6.3, 1.1, 2.8, 1.2]
        for node in G.nodes():
            G.nodes[node]['A'] = a[node]
            G.nodes[node]['B'] = b[node]

        df_test = pd.DataFrame()
        df_test['A'] = a
        df_test['B'] = b
        pdt.assert_frame_equal(df_test,
                               network_to_df(G),
                               check_like=True)


class TestProbabilityOdds:

    def test_conversions(self):
        p = np.array([0.7, 0.5, 0.1, 0.01, 0.99])
        npt.assert_allclose(p,
                            odds_to_probability(probability_to_odds(p)),
                            rtol=1e-12)

        o = np.array([2, 1, 0.1, 9, 12, 0.3, ])
        npt.assert_allclose(o,
                            probability_to_odds(odds_to_probability(o)),
                            rtol=1e-12)


class TestBounding:

    def test_error_negative_bound(self):
        with pytest.raises(ValueError):
            bounding(np.array([0.1, 0.5, 1.3]), bound=-3)

    def test_error_string(self):
        with pytest.raises(ValueError):
            bounding(np.array([0.1, 0.5, 1.3]), bound='three')

    def test_error_order(self):
        with pytest.raises(ValueError):
            bounding(np.array([0.1, 0.5, 1.3]), bound=[5, 0.1])

    def test_bound_above1(self):
        v = bounding(np.array([0.2, 1.1, 2, 5, 10]), bound=3)
        npt.assert_allclose([1/3, 1.1, 2, 3, 3],
                            v,
                            atol=1e-5)

    def test_bound_below1(self):
        v = bounding(np.array([0.1, 0.2, 0.5, 1.0, 40]), bound=0.3)
        npt.assert_allclose([0.3, 0.3, 0.5, 1.0, 1/0.3],
                            v,
                            atol=1e-5)


class TestTMLEBounding:

    def test_outcome_bound(self):
        y = np.array([-3, 4, 7, 1, 0, -10, 10])
        yb = tmle_unit_bounds(y=y, mini=np.min(y), maxi=np.max(y))
        npt.assert_allclose([3.500e-01, 7.000e-01, 8.500e-01, 5.500e-01, 5.000e-01, 0.0, 1.0],
                            yb,
                            atol=1e-5)

    def test_outcome_unbound(self):
        yb = np.array([3.500e-01, 7.000e-01, 8.500e-01, 5.500e-01, 5.000e-01, 0.0, 1.0])
        y = tmle_unit_unbound(ystar=yb, mini=-10, maxi=10)
        npt.assert_allclose([-3, 4, 7, 1, 0, -10, 10],
                            y,
                            atol=1e-5)

    def test_bound_unbound(self):
        y = [-4, 2, 41, 6, 1, 20, -3, -2, -8]
        y_max = np.max(y)
        y_min = np.min(y)

        yb = tmle_unit_bounds(y=y, mini=y_min, maxi=y_max)
        yu = tmle_unit_unbound(ystar=yb, mini=y_min, maxi=y_max)

        # Checking get back to original Y
        npt.assert_allclose(y, yu, atol=1e-5)

    def test_unbound_bound(self):
        y = np.array([0.05, 0.15, 0.9999, 0.24, 0.34, 0, 0.01, 0.54, 1])
        y_max = 20
        y_min = -8

        yu = tmle_unit_unbound(ystar=y, mini=y_min, maxi=y_max)
        yb = tmle_unit_bounds(y=yu, mini=y_min, maxi=y_max)

        # Checking get back to original Y
        npt.assert_allclose(y, yb, atol=1e-5)


class TestConditionalCheck:

    @pytest.fixture
    def data(self):
        df = pd.DataFrame()
        df['A'] = [0]*5 + [1]*10
        df['B'] = [0]*3 + [1]*7 + [0]*5
        return df

    def test_no_problems(self, data):
        with pytest.warns(None) as record:
            check_conditional(df=data, conditional=["g['A']==1", "g['A']==0"])

        # Makes sure no warnings are generated
        assert not record

    def test_conditional_warn(self, data):
        with pytest.warns(UserWarning):
            check_conditional(df=data, conditional=["g['A']==1", "g['B']==1"])
