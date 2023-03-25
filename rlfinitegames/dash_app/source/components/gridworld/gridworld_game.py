from dash import Dash, html, dcc
from rlfinitegames.dash_app.source.components import ids
from dash.dependencies import Input, Output
from rlfinitegames.environments.grid_world import GridWorld
from rlfinitegames.dash_app.source.backend_gridworld import costum_render
from dash.exceptions import PreventUpdate

ENVIRONMENT = GridWorld(size=10)


def render(app: Dash) -> html.Div:
    @app.callback(
        [Output(ids.GRID_WORLD_GAMEFIELS, "figure"),
         Output(ids.GRID_WORLD_TEXT_BOX, "value"),
         Output(ids.GRID_WORLD_UPDATE_BUTTON, "n_clicks")],
        [
            Input(ids.GRID_WORLD_ACTION_VALUE, "children"),
            Input(ids.GRID_WORLD_UPDATE_BUTTON, "n_clicks"),
        ],
    )
    def update_figure(action: str, n_clicks: int):
        action = int(action)
        print(f"action: {action}, n_clicks: {n_clicks}")
        if n_clicks is None:
            raise PreventUpdate
        valid_actions = ENVIRONMENT.get_valid_actions(ENVIRONMENT.state)
        if action not in valid_actions:
            fig = costum_render(ENVIRONMENT.state, env=ENVIRONMENT)
            info = f"the action {action} is not valid"
        else:
            next_state, reward, done, _ = ENVIRONMENT.step(action)
            if done:
                ENVIRONMENT.reset()
                next_state = ENVIRONMENT.state
            info = info = f"reward: {reward}, next state: {next_state}, done: {done}"
                # Create figure from the new state
            fig = costum_render(state=next_state.tolist(),env=ENVIRONMENT)

        return fig, info, None

    return html.Div(
        children=[
            dcc.Graph(
                id=ids.GRID_WORLD_GAMEFIELS,
                figure=costum_render(
                    state=ENVIRONMENT.state.tolist(), env=ENVIRONMENT)
            )

        ]
    )
