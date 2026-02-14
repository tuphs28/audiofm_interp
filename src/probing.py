from typing import Literal

import numpy as np
from sklearn.linear_model import RidgeCV, Lasso, LinearRegression, LassoCV
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler


class LinearProbe:

    def __init__(
        self,
        probe_id: Literal["ridge", "lasso", "linear"],
        alphas: list[float] = [1.0],
        scale_targets: bool = True,
        scale_hidden_states: bool = True
    ):

        self.scale_targets = scale_targets
        self.scale_hidden_states = scale_hidden_states

        if probe_id == "ridge":
            self.model = RidgeCV(alphas=alphas)
        elif probe_id == "lasso":
            print("this could be sow i need to double check before running")
            self.model = MultiOutputRegressor(LassoCV(alphas=alphas))
        elif probe_id == "linear":
            self.model = LinearRegression()
        else:
            raise ValueError(f"Unsupported probe_id: {probe_id}")

    
    def fit_and_score(
        self,
        hidden_states_train: np.ndarray,
        hidden_states_test: np.ndarray,
        targets_train: np.ndarray,
        targets_test: np.ndarray
    ) -> dict[str, float | np.ndarray]:

        if self.scale_hidden_states:
            scaler = StandardScaler()
            hidden_states_train = scaler.fit_transform(hidden_states_train)
            hidden_states_test = scaler.transform(hidden_states_test)

        if self.scale_targets:
            target_scaler = StandardScaler()
            targets_train = target_scaler.fit_transform(targets_train)
            targets_test = target_scaler.transform(targets_test)

        self.model.fit(hidden_states_train, targets_train)

        predictions = self.model.predict(hidden_states_test)
        global_r2 = r2_score(targets_test, predictions, multioutput='uniform_average')
        individual_r2 = r2_score(targets_test, predictions, multioutput='raw_values')

        return {
            "global_r2": global_r2,
            "individual_r2": individual_r2,
            "probe_coefs": self.model.coef_,
            "probe_intercepts": self.model.intercept_,
            "best_alpha": getattr(self.model, 'alpha_', None)
        }






    

