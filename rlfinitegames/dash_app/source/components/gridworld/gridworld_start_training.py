from dash import Dash, html, dcc
from rlfinitegames.dash_app.source.components import ids
import i18n
from typing import Union
from dash.dependencies import Input, Output
from dash_bootstrap_components import Button
from dash.exceptions import PreventUpdate
from rlfinitegames.dash_app.source.components.gridworld.grid_world_play_trained_agent import GRID_WORLD_POLICY_ITERATION


def render(app: Dash) -> html.Div:
    @app.callback(
        Output(ids.GRID_WORLD_PLACEHOLDER, "children"),
        [
            Input(ids.GRID_WORLD_START_TRAINING_BUTTON, "n_clicks"),
        ],
    )
    def update_grid_action_value(start_training: Union[None, int]) -> str:
        if start_training is None:
            raise PreventUpdate
        else:
            # print(f"policy before policy iteration {GRID_WORLD_POLICY_ITERATION.agent.policy}")
            GRID_WORLD_POLICY_ITERATION.policy_iteration()
            # print(f"policy after policy iteration {GRID_WORLD_POLICY_ITERATION.agent.policy}")
        return ""

    return html.Div(
        children=[
            Button(i18n.t("general.grid-world-start-training"), id=ids.GRID_WORLD_START_TRAINING_BUTTON,
                   class_name="me-2", n_clicks=None),
            html.Div("Placeholder", id=ids.GRID_WORLD_PLACEHOLDER,
                     style={'display': 'none'})
        ]
    )
