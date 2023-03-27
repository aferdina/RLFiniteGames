from dash import Dash, html, dcc
from rlfinitegames.dash_app.source.components import ids
import i18n
from dash.dependencies import Input, Output
from dash_bootstrap_components import Button
from typing import Union
from rlfinitegames.algorithms.policy_iteration import PolicyIteration
from rlfinitegames.policies.discrete_agents import FiniteAgent
from rlfinitegames.dash_app.source.components.gridworld.grid_world_game_trained_agent import ENVIRONMENT 
from dash.exceptions import PreventUpdate

GRID_WORLD_POLICY_ITERATION = PolicyIteration(environment=ENVIRONMENT, policy=FiniteAgent(env=ENVIRONMENT))

def render(app: Dash) -> html.Div:
    @app.callback(
        Output(ids.GRID_WORLD_TRAINED_AGENT_VALUE, "children"),
        [
        Input(ids.GRID_WORLD_START_TRAINED_AGENT_BUTTON, "n_clicks"),
        ],
    )
    def update_grid_action_value(n_clicks: Union[int,None]) -> str:
        if n_clicks is None:
            raise PreventUpdate
        else:
            action = GRID_WORLD_POLICY_ITERATION.agent.get_action(ENVIRONMENT.state)
        print(f"state was is {ENVIRONMENT.state}")
        print(f"action was is {action}")
        return str(action)
    
    return html.Div(
        children=[
            Button(i18n.t("general.grid-world-play-trained-agent"), id=ids.GRID_WORLD_START_TRAINED_AGENT_BUTTON,
                       class_name="me-2", n_clicks=None),
            html.Div(None,id=ids.GRID_WORLD_TRAINED_AGENT_VALUE, style={'display': 'none'})
        ]
    )