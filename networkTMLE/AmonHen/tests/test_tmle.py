import pytest
import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
from sklearn.linear_model import LogisticRegression

from zepid import load_sample_data, spline
from zepid.causal.doublyrobust import TMLE

from amonhen import StochasticTMLE


class TestStochasticTMLE:

    @pytest.fixture
    def df(self):
        df = load_sample_data(False)
        df[['cd4_rs1', 'cd4_rs2']] = spline(df, 'cd40', n_knots=3, term=2, restricted=True)
        df[['age_rs1', 'age_rs2']] = spline(df, 'age0', n_knots=3, term=2, restricted=True)
        return df.drop(columns=['cd4_wk45']).dropna()

    @pytest.fixture
    def cf(self):
        df = load_sample_data(False)
        df[['cd4_rs1', 'cd4_rs2']] = spline(df, 'cd40', n_knots=3, term=2, restricted=True)
        df[['age_rs1', 'age_rs2']] = spline(df, 'age0', n_knots=3, term=2, restricted=True)
        return df.drop(columns=['dead']).dropna()

    @pytest.fixture
    def simple_df(self):
        expected = pd.DataFrame([[1, 1, 1, 1, 1],
                                 [0, 0, 0, -1, 2],
                                 [0, 1, 0, 5, 1],
                                 [0, 0, 1, 0, 0],
                                 [1, 0, 0, 0, 1],
                                 [1, 0, 1, 0, 0],
                                 [0, 1, 0, 10, 1],
                                 [0, 0, 0, -5, 0],
                                 [1, 1, 0, -5, 2]],
                                columns=["W", "A", "Y", "C", "S"],
                                index=[1, 2, 3, 4, 5, 6, 7, 8, 9])
        return expected

    def test_error_continuous_exp(self, df):
        with pytest.raises(ValueError):
            StochasticTMLE(df=df, exposure='cd40', outcome='dead')

    def test_error_fit(self, df):
        stmle = StochasticTMLE(df=df, exposure='art', outcome='dead')
        with pytest.raises(ValueError):
            stmle.fit(p=0.5)

        stmle = StochasticTMLE(df=df, exposure='art', outcome='dead')
        stmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
        with pytest.raises(ValueError):
            stmle.fit(p=0.5)

        stmle = StochasticTMLE(df=df, exposure='art', outcome='dead')
        stmle.outcome_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
        with pytest.raises(ValueError):
            stmle.fit(p=0.5)

    def test_error_p_oob(self, df):
        stmle = StochasticTMLE(df=df, exposure='art', outcome='dead')
        stmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
        stmle.outcome_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
        with pytest.raises(ValueError):
            stmle.fit(p=1.1)

        with pytest.raises(ValueError):
            stmle.fit(p=-0.1)

    def test_error_p_cond_len(self, df):
        stmle = StochasticTMLE(df=df, exposure='art', outcome='dead')
        stmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
        stmle.outcome_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
        with pytest.raises(ValueError):
            stmle.fit(p=[0.1], conditional=["df['male']==1", "df['male']==0"])

        with pytest.raises(ValueError):
            stmle.fit(p=[0.1, 0.3], conditional=["df['male']==1"])

    def test_error_summary(self, df):
        stmle = StochasticTMLE(df=df, exposure='art', outcome='dead')
        stmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
        stmle.outcome_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
        with pytest.raises(ValueError):
            stmle.summary()

    def test_warn_missing_data(self):
        df = pd.DataFrame()
        df['A'] = [1, 1, 0, 0, np.nan]
        df['Y'] = [np.nan, 0, 1, 0, 1]
        with pytest.warns(UserWarning):
            StochasticTMLE(df=df, exposure='A', outcome='Y')

    def test_drop_missing_data(self):
        df = pd.DataFrame()
        df['A'] = [1, 1, 0, 0, np.nan]
        df['Y'] = [np.nan, 0, 1, 0, 1]
        stmle = StochasticTMLE(df=df, exposure='A', outcome='Y')
        assert stmle.df.shape[0] == 3

    def test_continuous_processing(self):
        a_list = [0, 1, 1, 0, 1, 1, 0, 0]
        y_list = [1, -1, 5, 0, 0, 0, 10, -5]
        df = pd.DataFrame()
        df['A'] = a_list
        df['Y'] = y_list

        stmle = StochasticTMLE(df=df, exposure='A', outcome='Y', continuous_bound=0.0001)

        # Checking all flagged parts are correct
        assert stmle._continuous_outcome is True
        assert stmle._continuous_min == -5
        assert stmle._continuous_max == 10
        assert stmle._cb == 0.0001

        # Checking that TMLE bounding works as intended
        y_bound = [2 / 5, 4 / 15, 2 / 3, 1 / 3, 1 / 3, 1 / 3, 0.9999, 0.0001]
        pdt.assert_series_equal(pd.Series(y_bound),
                                stmle.df['Y'],
                                check_dtype=False, check_names=False)

    def test_marginal_vector_length_stoch(self, df):
        stmle = StochasticTMLE(df=df, exposure='art', outcome='dead')
        stmle.exposure_model('male')
        stmle.outcome_model('art + male + age0')
        stmle.fit(p=0.4, samples=7)
        assert len(stmle.marginals_vector) == 7

    def test_qmodel_params(self, simple_df):
        # Comparing to SAS logit model
        sas_params = [-1.0699, -0.9525, 1.5462]
        sas_preds = [0.3831332, 0.2554221, 0.1168668, 0.2554221, 0.6168668, 0.6168668, 0.1168668, 0.2554221, 0.3831332]

        stmle = StochasticTMLE(df=simple_df, exposure='A', outcome='Y')
        stmle.outcome_model('A + W')
        est_params = stmle._outcome_model.params
        est_preds = stmle._Qinit_

        npt.assert_allclose(sas_params, est_params, atol=1e-4)
        npt.assert_allclose(sas_preds, est_preds, atol=1e-6)

    def test_qmodel_params2(self, simple_df):
        # Comparing to SAS linear model
        sas_params = [0.3876, 0.3409, -0.2030, -0.0883]
        sas_preds = [0.437265, 0.210957, 0.6402345, 0.3876202, 0.0963188, 0.1846502, 0.6402345, 0.38762016, 0.34893314]

        stmle = StochasticTMLE(df=simple_df, exposure='A', outcome='C')
        stmle.outcome_model('A + W + S', continuous_distribution='normal')
        est_params = stmle._outcome_model.params
        est_preds = stmle._Qinit_

        npt.assert_allclose(sas_params, est_params, atol=1e-4)
        npt.assert_allclose(sas_preds, est_preds, atol=1e-6)

    def test_qmodel_params3(self, simple_df):
        # Comparing to SAS Poisson model
        sas_params = [-1.0478, 0.9371, -0.5321, -0.2733]
        sas_preds = [0.4000579, 0.2030253, 0.6811115, 0.3507092, 0.1567304, 0.20599265, 0.6811115, 0.3507092, 0.3043857]

        stmle = StochasticTMLE(df=simple_df, exposure='A', outcome='C')
        stmle.outcome_model('A + W + S', continuous_distribution='Poisson')
        est_params = stmle._outcome_model.params
        est_preds = stmle._Qinit_

        npt.assert_allclose(sas_params, est_params, atol=1e-4)
        npt.assert_allclose(sas_preds, est_preds, atol=1e-6)

    def test_gmodel_params(self, simple_df):
        # Comparing to SAS Poisson model
        sas_preds = [2.0, 1.6666666667, 2.5, 1.6666666667, 2, 2, 2.5, 1.6666666667, 2]

        stmle = StochasticTMLE(df=simple_df, exposure='A', outcome='C')
        stmle.exposure_model('W')
        est_preds = 1 / stmle._denominator_

        npt.assert_allclose(sas_preds, est_preds, atol=1e-6)

    def test_compare_tmle_binary(self, df):
        stmle = StochasticTMLE(df, exposure='art', outcome='dead')
        stmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
        stmle.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
        stmle.fit(p=1.0, samples=1)
        all_treat = stmle.marginal_outcome
        stmle.fit(p=0.0, samples=1)
        non_treat = stmle.marginal_outcome

        tmle = TMLE(df, exposure='art', outcome='dead')
        tmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0', print_results=False)
        tmle.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False)
        tmle.fit()
        expected = tmle.risk_difference

        npt.assert_allclose(expected, all_treat - non_treat, atol=1e-4)

    def test_compare_tmle_continuous(self, cf):
        cf['cd4_wk45'] = np.log(cf['cd4_wk45'])
        stmle = StochasticTMLE(cf, exposure='art', outcome='cd4_wk45')
        stmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
        stmle.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
        stmle.fit(p=1.0, samples=1)
        all_treat = stmle.marginal_outcome
        stmle.fit(p=0.0, samples=1)
        non_treat = stmle.marginal_outcome

        tmle = TMLE(cf, exposure='art', outcome='cd4_wk45')
        tmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0', print_results=False)
        tmle.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False)
        tmle.fit()
        expected = tmle.average_treatment_effect

        npt.assert_allclose(expected, all_treat - non_treat, atol=1e-3)

    def test_gmodel_bound(self, simple_df):
        # Comparing to SAS Poisson model
        sas_preds = [2, 1/0.55, 1/0.45, 1/0.55, 2, 2, 1/0.45, 1/0.55, 2]

        stmle = StochasticTMLE(df=simple_df, exposure='A', outcome='C')
        stmle.exposure_model('W', bound=[0.45, 0.55])
        est_preds = 1 / stmle._denominator_

        npt.assert_allclose(sas_preds, est_preds, atol=1e-6)

    def test_calculate_epsilon1(self, df):
        stmle = StochasticTMLE(df, exposure='art', outcome='dead')
        stmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
        stmle.outcome_model('art + male + age0 + age_rs1 + age_rs2 + dvl0  + cd40 + cd4_rs1 + cd4_rs2')

        stmle.fit(p=0.15, samples=1)
        npt.assert_allclose(-0.0157043107, stmle.epsilon, atol=1e-6)

        stmle.fit(p=0.4, samples=1)
        npt.assert_allclose(-0.0381559025, stmle.epsilon, atol=1e-6)

    def test_calculate_epsilon2(self, cf):
        stmle = StochasticTMLE(cf, exposure='art', outcome='cd4_wk45')
        stmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
        stmle.outcome_model('art + male + age0 + age_rs1 + age_rs2 + dvl0  + cd40 + cd4_rs1 + cd4_rs2')

        stmle.fit(p=0.15, samples=1)
        npt.assert_allclose(-0.0059476590, stmle.epsilon, atol=1e-6)

        stmle.fit(p=0.4, samples=1)
        npt.assert_allclose(-0.0154923643, stmle.epsilon, atol=1e-6)

    def test_machine_learning_runs(self, df):
        # Only verifies that machine learning doesn't throw an error
        log = LogisticRegression(penalty='l1', solver='liblinear', random_state=201)

        tmle = StochasticTMLE(df, exposure='art', outcome='dead')
        tmle.exposure_model('male + age0 + cd40 + cd4_rs1 + cd4_rs2 + dvl0 + male:dvl0', custom_model=log)
        tmle.outcome_model('art + male + age0 + dvl0  + cd40', custom_model=log)
        tmle.fit(p=0.4, samples=20)
